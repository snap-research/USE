"""
    Below are the customized classes for RetNet.
    Implementation adapted from
    https://github.com/fkodom/yet-another-retnet and
    https://github.com/microsoft/torchscale/commit/bf65397b26469ac9c24d83a9b779b285c1ec640b#diff-8c0a56195606d489b702e9270ba269c24803354ff8e70056f66946353b070c2d
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import einsum, rearrange, repeat
from model import ContrastiveLoss
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Normalize
from transformers import PretrainedConfig, top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

RETNET_QK_RATIO = 1
RETNET_V_RATIO = 1
RETNET_FFN_RATIO = 4
DEFAULT_DEVICE = torch.device("cpu")

"""
    The configuration class of RetNet
"""


class RetNetConfig(PretrainedConfig):
    model_type = "retnet"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 3,
        qk_dim: int = 768,
        v_dim: int = 1532,
        ffn_proj_size: int = 1532,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        is_decoder: bool = True,
        pad_token_id: int = 50256,
        eos_token_id: int = 50256,
        output_retentions: bool = False,
        use_cache: bool = True,
        forward_impl: str = "parallel",
        chunk_size: int = 512,
        activation: str = "swish",
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        use_bias_in_mlp: bool = True,
        use_bias_in_retention: bool = True,
        tie_word_embeddings: bool = True,
        fep_loss_frequency: int = 1,
        fep_context_length: int = 0,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.ffn_proj_size = ffn_proj_size
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.output_retentions = output_retentions
        self.forward_impl = forward_impl
        self.chunk_size = chunk_size
        self.activation = activation
        self.device = device
        self.dtype = dtype
        self.use_bias_in_mlp = use_bias_in_mlp
        self.use_bias_in_retention = use_bias_in_retention
        self.tie_word_embeddings = tie_word_embeddings
        self.fep_loss_frequency = fep_loss_frequency
        self.fep_context_length = fep_context_length

        super().__init__(
            is_decoder=is_decoder,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string"""
    activation_functions = {"swish": F.silu, "gelu": F.gelu, "relu": F.relu}

    if activation in activation_functions:
        return activation_functions[activation]
    else:
        raise RuntimeError(
            f"Unsupported activation string '{activation}'. "
            f"Supported: {activation_functions.keys()}"
        )


def _build_decay_gammas(
    num_heads: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Decay values are different for each retention head, following the prescribed
    method in the paper.  Conceptually, I think of each head having a different
    "retention window", which is the effective number of steps back in time that
    the head can attend to.  Retention windows are effectively determined by
    these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)
    """
    xmin, xmax = math.log(1 / 32), math.log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - torch.exp(x)


def _build_decay_mask(
    query_length: int,
    key_length: int,
    decay_gammas: Tensor,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed
    and applied to the similarity matrix at once, rather than being applied to
    each element in the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5
    """
    query_pos = torch.arange(query_length, device=device, dtype=dtype)
    key_pos = torch.arange(key_length, device=device, dtype=dtype)

    distance = query_pos.unsqueeze(-1) - key_pos.unsqueeze(0)

    # Set the upper-triangular distances to infinity, so that only *past* keys
    # can affect the current query.  (Setting distance to infinity ensures that
    # the decay matrix is 0 for those positions, since x^(inf) = 0 when -1 < x < 1.
    distance_mask = torch.ones_like(distance, dtype=torch.bool).triu_(diagonal=1)
    distance = distance.masked_fill(distance_mask, float("inf"))

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas**distance


def _build_position_thetas(
    head_dim: int,
    scale: float = 10000,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Positional thetas are different for each value along head_dim, following the
    prescribed method in the paper.  These are used to update the positional
    embeddings in both the parallel and recurrent formulations of retention.
    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 2.1 (Retention)

    NOTE: The actual values for thetas are not specified in the paper, so I
    copied these values from the official implementation.
    See: https://github.com/microsoft/torchscale/blob/7d231743f4f96c460b7cf0aa0cf242bb192b34f8/torchscale/architecture/retnet.py#L27C1-L28C59
    """
    x = torch.linspace(0, 1, steps=head_dim // 2, device=device, dtype=dtype)
    thetas = 1 / (scale**x)
    return repeat(thetas, "d -> (d n)", n=2)


# NOTE: For the purposes of positional embeddings, we view query/key Tensors as
# complex-valued, where the even-numbered indices are the real part, and the
# odd-numbered indices are the imaginary part.  This makes it easy to compute
# complex values without *actually* using complex dtypes in PyTorch.
# (Complex dtypes have limited support compared to real dtypes.)
#
# I don't re-explain this in the functions below, but it's important to keep in
# mind when reading the code.


def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


def retention_parallel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    decay_gammas: Optional[Tensor] = None,
    output_retentions: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        decay_gammas=decay_gammas,
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    if output_retentions:
        return retention, similarity
    else:
        return retention, None


def retention_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
    decay_gammas: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    state = einsum(key, value, "b h d, b h m -> b h d m")
    if prev_state is not None:
        state = state + prev_state * rearrange(decay_gammas, "h -> () h () ()")
    retention = einsum(query, state, "b h d, b h d m -> b h m")

    return retention, state


def retention_chunkwise(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
    decay_gammas: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        decay_gammas=decay_gammas,
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    states = einsum(key, value, "b h n d1, b h n d2 -> b h n d1 d2")
    state_decays = decay_gammas ** (key.size(2) - inner_pos)
    state = einsum(states, state_decays, "b h n d1 d2, _ h n _ -> b h d1 d2")
    if prev_state is not None:
        # Update internal state to return to the user
        chunk_decay = decay_gammas ** key.size(2)
        state = state + prev_state * chunk_decay
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas**inner_pos
        retention = retention + (
            einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


class MultiScaleRetention(nn.Module):
    """Multi-scale retention (MSR) layer.  Intended to be (mostly) a drop-in replacement
    for nn.MultiheadAttention, but with the option to use either the parallel or
    recurrent formulation of retention. (Attention only has the parallel formulation.)

    NOTE: As presented in the paper, Multi-Scale Retention includes an explicit
    position embedding, which is based on xPos.  IMO, this is unnecessary and overly
    specific to language modeling, since other domains (e.g. computer vision,
    heterogeneous graphs) will have different positional semantics.

    Reference:
        "Retentive Network: A Successor to Transformer for Large Language Models"
        https://arxiv.org/pdf/2307.08621v3.pdf
    """

    def __init__(
        self,
        config: RetNetConfig,
    ):

        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.dropout = 0.0
        self.bias = config.use_bias_in_retention
        self.activation = _get_activation_fn(config.activation)

        device, dtype = config.device, config.dtype

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )

        self.head_dim = self.hidden_size // self.num_heads
        if not self.head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (hidden_size / num_heads = {self.head_dim}) must be divisible by 8"
            )

        # The q/k/v projection layers are the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )
        self.group_norm = nn.GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.num_heads,
            affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )
        # The output project is slightly different, due to the gated "swish" layer.
        self.g_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )

        # 'thetas' parameter for updating the relative position embeddings.
        thetas: Optional[Tensor] = None
        thetas = _build_position_thetas(
            head_dim=self.head_dim, device=device, dtype=dtype
        )
        self.thetas: Optional[Tensor]
        self.register_buffer("thetas", thetas)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Double-check that we're following the same initialization as in
        # the paper.  This is a generic initialization for MHA linear layers.
        for layer in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.g_proj,
        ]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward_parallel(
        self,
        X: Tensor,
        output_retentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(X)
        k: Tensor = self.k_proj(X)
        v: Tensor = self.v_proj(X)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        assert self.thetas is not None
        indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
        indices = rearrange(indices, "n -> () () n ()")
        thetas = rearrange(self.thetas, "d -> () () () d")
        angles = indices * thetas
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        q = _theta_shift(q, sin, cos)
        k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, weights = retention_parallel(
            q, k, v, output_retentions=output_retentions
        )
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) h d")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) h d -> b n (h d)", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'X' can equivalently be used as the input.
        gate = self.activation(self.g_proj(X))
        retention = self.out_proj(retention * gate)

        return retention, weights

    def forward_recurrent(
        self,
        X: Tensor,
        sequence_offset: int,
        prev_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # h - number of heads
        # d - embedding dimension
        #
        # input shape: (b, d)
        q: Tensor = self.q_proj(X)
        k: Tensor = self.k_proj(X)
        v: Tensor = self.v_proj(X)

        # Unfold 'd' dimension into 'h' separate retention heads.
        q = rearrange(q, "b (h d) -> b h d", h=self.num_heads)
        k = rearrange(k, "b (h d) -> b h d", h=self.num_heads)
        v = rearrange(v, "b (h d) -> b h d", h=self.num_heads)

        assert self.thetas is not None
        thetas = rearrange(self.thetas, "d -> () () d")
        angles = sequence_offset * thetas
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        q = _theta_shift(q, sin, cos)
        k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_recurrent(q, k, v, prev_state=prev_state)
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Fold heads back into the embedding dimension.
        retention = rearrange(retention, "b h d -> b (h d)")

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'X' can equivalently be used as the input.
        gate = self.activation(self.g_proj(X))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_chunkwise(
        self,
        X: Tensor,
        sequence_offset: int,
        prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(X)
        k: Tensor = self.k_proj(X)
        v: Tensor = self.v_proj(X)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # global (cross-chunk) relative position embedding
        assert self.thetas is not None
        thetas = rearrange(self.thetas, "d -> () () () d")
        angles = sequence_offset * thetas
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        q = _theta_shift(q, sin, cos)
        k = _theta_shift(k, sin, cos)

        # intra-chunk relative position encoding
        indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
        indices = rearrange(indices, "n -> () () n ()")
        thetas = rearrange(self.thetas, "d -> () () () d")
        angles = indices * thetas
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        q = _theta_shift(q, sin, cos)
        k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_chunkwise(q, k, v, prev_state=prev_state)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) h d")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) h d -> b n (h d)", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'X' can equivalently be used as the input.
        gate = self.activation(self.g_proj(X))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward(
        self,
        X: Tensor,
        retention_mask: Optional[Tensor] = None,
        prev_state: Optional[Tuple[Tensor]] = None,
        forward_impl: str = "parallel",
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ):

        if forward_impl == "parallel":
            return self.forward_parallel(X, output_retentions)

        elif forward_impl == "recurrent":
            return self.forward_recurrent(X.squeeze(1), sequence_offset, prev_state)

        elif forward_impl == "chunkwise":
            return self.forward_chunkwise(X, sequence_offset, prev_state)


class RetNetBlock(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.msr = MultiScaleRetention(config)

        self.ffn = nn.Sequential(
            nn.Linear(
                config.hidden_size, config.ffn_proj_size, bias=config.use_bias_in_mlp
            ),
            nn.GELU(),
            nn.Linear(
                config.ffn_proj_size, config.hidden_size, bias=config.use_bias_in_mlp
            ),
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        retention_mask: Optional[Tensor] = None,
        forward_impl: str = "parallel",
        prev_state: Optional[Tuple[Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:

        msr_outs = self.msr(
            self.ln1(hidden_states),
            retention_mask=retention_mask,
            prev_state=prev_state,
            forward_impl=forward_impl,
            sequence_offset=sequence_offset,
            chunk_size=chunk_size,
            output_retentions=output_retentions,
        )
        msr = msr_outs[0]
        curr_kv = msr_outs[1]
        y = hidden_states.squeeze(1) + self.dropout(msr)
        y = y + self.ffn(self.ln2(y))

        outputs = (y, curr_kv)

        if output_retentions:
            outputs += (msr_outs[2],)
        return outputs


class RetNetPreTrainedModel(PreTrainedModel):
    # copied from LlamaPretrainedModel
    config_class = RetNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RetNetBlock"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RetNetModel):
            module.gradient_checkpointing = value


@dataclass
class RetNetOutputWithPast(ModelOutput):
    """
    class for RetNet model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `prev_states` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        prev_states (`tuple(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape
            `(batch_size, num_heads, qk_dim, v_dim)`.

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `prev_states` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.
    """

    last_hidden_state: torch.FloatTensor = None
    prev_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None


class RetNetModel(RetNetPreTrainedModel):
    def __init__(self, config: RetNetConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.blocks = nn.ModuleList(
            [RetNetBlock(config) for _ in range(config.num_layers)]
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        prev_states: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_retentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = "parallel",
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
    ) -> Union[Tuple, RetNetOutputWithPast]:

        if not prev_states:
            prev_states = [None] * self.config.num_layers

        output_retentions = (
            output_retentions
            if output_retentions is not None
            else self.config.output_retentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if retention_mask is None:
            if attention_mask is not None:
                retention_mask = attention_mask
            else:
                # TODO: might not need this
                retention_mask = torch.ones(
                    (batch_size, seq_length),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, hidden_size]
        next_decoder_cache = () if use_cache else None

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            prev_state = prev_states[i] if prev_states is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, sequence_offset, chunk_size, output_retentions
                        )

                    return custom_forward

                block_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    retention_mask,
                    forward_impl,
                    prev_state,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    retention_mask=retention_mask,
                    forward_impl=forward_impl,
                    prev_state=prev_state,
                    sequence_offset=sequence_offset,
                    chunk_size=chunk_size,
                    output_retentions=output_retentions,
                )

            hidden_states = block_outputs[0]

            if use_cache:
                next_decoder_cache += (block_outputs[1],)

            if output_retentions:
                all_retentions += (block_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_retentions]
                if v is not None
            )
        return RetNetOutputWithPast(
            last_hidden_state=hidden_states,
            prev_states=next_cache,
            hidden_states=all_hidden_states,
            retentions=all_retentions,
        )


@dataclass
class RetNetCausalLMOutputWithPast(ModelOutput):
    """
    class for RetNet causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        prev_states (`tuple(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape
            `(batch_size, num_heads, qk_dim, v_dim)`.

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `prev_states` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    prev_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None
    fep_logits: torch.FloatTensor = None


# This is an implement of the SimSiam algorithm proposed in  https://arxiv.org/pdf/2011.10566.pdf
def SimSiamLoss(pred1, pred2, proj1, proj2):
    criterion = nn.CosineSimilarity(dim=1)
    return -(criterion(pred1, proj2).mean() + criterion(pred2, proj1).mean()) * 0.5


class RetNetModelWithLMHead(RetNetPreTrainedModel):
    def __init__(self, config: RetNetConfig) -> None:
        super().__init__(config)
        self.model = RetNetModel(config)

        # initialize hyperparameter if not specified
        if "contrastive_head_size" not in config.to_dict().keys():
            config.contrastive_head_size = 128
            config.contrastive_temperature = 0.05
            config.contrastive_type = "Orig"
            config.contrastive_embedding = "token"

        # casaul language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_clm_fct = torch.nn.CrossEntropyLoss()

        # future event prediction
        if "num_fep_events" not in config.to_dict().keys():
            config.num_fep_events = 1
            config.fep_context_length = 0
        self.num_fep_events = config.num_fep_events
        self.fep_context_length = config.fep_context_length
        self.fep_loss_frequency = config.fep_loss_frequency
        if self.num_fep_events == config.vocab_size:
            print("Using LM head as FEP head")
            self.fep_head = self.lm_head
        else:
            self.fep_head = nn.Linear(config.hidden_size, config.num_fep_events)

        # same user prediction
        self.contrastive_embedding = config.contrastive_embedding
        self.contrastive_type = config.contrastive_type

        # use the average of token embeddings as sequence embedding
        if self.contrastive_embedding == "token":
            pass
        # use a linear layer to project RetNet state as sequence embedding
        elif self.contrastive_embedding == "state":
            self.feature_extractor = nn.Linear(
                config.head_size * config.head_size, config.hidden_size
            )
        # concatenate the above "token" and "state" embedding, and project it back to hidden size
        elif self.contrastive_embedding == "joint":
            self.feature_extractor = nn.Linear(
                config.head_size * config.head_size, config.hidden_size
            )
            self.feature_projector = nn.Linear(
                2 * config.hidden_size, config.hidden_size
            )
        # use a multi-layer CNN to transfer RetNet state as sequence embedding
        elif self.contrastive_embedding == "state_cnn":
            self.feature_normalizer = Normalize(mean=0.5, std=1)
            # default parameters of CoAtNet-tiny (about 18M parameters)
            self.feature_extractor = CoAtNet(
                image_size=(config.head_size, config.head_size),
                in_channels=config.num_heads,
                num_blocks=[2, 2, 3, 5, 2],
                channels=[64, 96, 192, 384, 768],
                block_types=["C", "C", "T", "T"],
                num_classes=config.hidden_size,
            )

        else:
            raise ValueError(
                "Only support contrastive_embedding of [token, state, joint, state_cnn]"
            )

        # The "Orig" is the original implemenation of SimCLR loss https://arxiv.org/abs/2002.05709
        # The "HardNeg" modifies SimCLR by assigning higher weights to the samples that are more similar to the anchor
        if self.contrastive_type in ["Orig", "HardNeg"]:
            self.sup_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.contrastive_head_size, bias=False),
            )
            self.loss_contrastive_fct = ContrastiveLoss(
                temperature=config.contrastive_temperature,
                contrastive_type=config.contrastive_type,
            )

        # "simsiam" is an implementation of SimSiam Loss. https://arxiv.org/abs/2011.10566
        # Add it here as a baseline of not using negative sampling in contrastive learning
        # Maybe useful when we have to use smaller batch size
        # But we do not get good results out of it in preliminary experiments.
        elif self.contrastive_type == "simsiam":
            self.predictor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                nn.BatchNorm1d(config.hidden_size),
                nn.ReLU(inplace=True),  # hidden layer
                nn.Linear(config.hidden_size, config.hidden_size),
            )  # output layer
            self.loss_contrastive_fct = SimSiamLoss
        else:
            raise ValueError(
                "Only support contrastive_type of [Orig, HardNeg, simsiam]"
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_embedding_from_token(self, attention_mask, token_embedding):
        attention_mask = attention_mask.unsqueeze(-1)
        sequence_embedding = torch.sum(
            token_embedding * attention_mask, dim=1
        ) / torch.sum(
            attention_mask, dim=1
        )  # shape: (batch_size, hidden_size)
        assert sequence_embedding.shape[0] % 2 == 0
        sequence_embedding_1 = sequence_embedding[::2]  # select the odd rows
        sequence_embedding_2 = sequence_embedding[1::2]  # select the even rows
        return sequence_embedding_1, sequence_embedding_2

    def get_embedding_from_state(self, prev_states):
        prev_states_1 = prev_states[-1][::2]  # select the odd rows
        prev_states_2 = prev_states[-1][1::2]  # select the even rows
        num_pairs = prev_states_1.shape[0]
        prev_states_1 = prev_states_1.mean(dim=1).view(
            [num_pairs, -1]
        )  # head_wise average
        prev_states_2 = prev_states_2.mean(dim=1).view(
            [num_pairs, -1]
        )  # head_wise average
        sequence_embedding_1 = self.feature_extractor(prev_states_1)
        sequence_embedding_2 = self.feature_extractor(prev_states_2)
        return sequence_embedding_1, sequence_embedding_1

    def get_embedding_from_state_cnn(self, prev_states):
        prev_states_1 = prev_states[-1][::2]  # select the odd rows
        prev_states_2 = prev_states[-1][1::2]  # select the even rows
        sequence_embedding_1 = self.feature_extractor(prev_states_1)
        sequence_embedding_2 = self.feature_extractor(prev_states_2)
        return sequence_embedding_1, sequence_embedding_1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        prev_states: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_retentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = "parallel",
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        time_ids: Optional[Tensor] = None,
        is_sup_training: Optional[bool] = False,
        is_lm_training: Optional[bool] = False,
        is_fep_training: Optional[bool] = False,
        fep_labels: Optional[Tensor] = None,
        fep_weights: Optional[Tensor] = None,
        return_loss: Optional[bool] = True,
    ) -> Union[Tuple, RetNetCausalLMOutputWithPast]:
        output_retentions = (
            output_retentions
            if output_retentions is not None
            else self.config.output_retentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if is_sup_training:
            forward_impl = (
                "parallel" if self.contrastive_embedding == "token" else "chunkwise"
            )

        input_ids, attention_mask = input_ids.squeeze(1), attention_mask.squeeze(1)

        chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size

        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        outputs = self.model(
            input_ids,
            retention_mask=retention_mask,
            prev_states=prev_states,
            inputs_embeds=inputs_embeds,
            output_retentions=output_retentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forward_impl=forward_impl,
            use_cache=use_cache,
            sequence_offset=sequence_offset,
            chunk_size=chunk_size,
        )

        token_embedding = outputs[0]
        

        loss_clm = None
        logits = self.lm_head(token_embedding)
        if is_lm_training:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_clm = self.loss_clm_fct(shift_logits, shift_labels)

        loss_fep = None
        fep_logits = self.fep_head(token_embedding)
        if is_fep_training:
            fep_logits = self.fep_head(token_embedding)
            fep_weights = attention_mask.float()
            # only calculate loss on the tokens with enough context
            fep_logits = fep_logits[:, self.fep_context_length :, :]
            fep_labels = fep_labels[:, self.fep_context_length :, :]
            fep_weights = fep_weights[:, self.fep_context_length :]
            # select certain tokens for loss calculation
            # for example, if fep_loss_frequency=5, the 5-th, 10-th, 15-th, ..., n*5-th tokens are selected.
            fep_logits = fep_logits[
                :, self.fep_loss_frequency - 1 :: self.fep_loss_frequency, :
            ]
            fep_labels = fep_labels[
                :, self.fep_loss_frequency - 1 :: self.fep_loss_frequency, :
            ]
            fep_weights = fep_weights[
                :, self.fep_loss_frequency - 1 :: self.fep_loss_frequency
            ]
            fep_logits = fep_logits.reshape(-1, self.num_fep_events)
            fep_labels = fep_labels.reshape(-1, self.num_fep_events).float()
            fep_weights = fep_weights.reshape(-1, 1)
            # use weight to avoid calculating loss on pad tokens
            loss_fep_fct = torch.nn.BCEWithLogitsLoss(weight=fep_weights)
            loss_fep = loss_fep_fct(fep_logits, fep_labels)

        loss_contrastive = None
        if is_sup_training:
            # generate sequence embeddings from the encoder
            if self.contrastive_embedding == "token":
                (
                    sequence_embedding_1,
                    sequence_embedding_2,
                ) = self.get_embedding_from_token(attention_mask, token_embedding)

            elif self.contrastive_embedding == "state":
                (
                    sequence_embedding_1,
                    sequence_embedding_2,
                ) = self.get_embedding_from_state(outputs.prev_states)

            elif self.contrastive_embedding == "joint":
                # token
                (
                    sequence_embedding_token_1,
                    sequence_embedding_token_2,
                ) = self.get_embedding_from_token(attention_mask, token_embedding)
                (
                    sequence_embedding_state_1,
                    sequence_embedding_state_2,
                ) = self.get_embedding_from_state(outputs.prev_states)

                # joint embedding
                sequence_embedding_1 = self.feature_projector(
                    torch.concat(
                        (sequence_embedding_token_1, sequence_embedding_state_1), dim=-1
                    )
                )
                sequence_embedding_2 = self.feature_projector(
                    torch.concat(
                        (sequence_embedding_token_2, sequence_embedding_state_2), dim=-1
                    )
                )

            elif self.contrastive_embedding == "state_cnn":
                (
                    sequence_embedding_1,
                    sequence_embedding_2,
                ) = self.get_embedding_from_state_cnn(outputs.prev_states)

            # perform contrastive learning over the sequence embeddings
            if self.contrastive_type == "simsiam":
                projection_1 = sequence_embedding_1
                projection_2 = sequence_embedding_2
                prediction_1 = self.predictor(projection_1)
                prediction_2 = self.predictor(projection_2)
                loss_contrastive = self.loss_contrastive_fct(
                    prediction_1,
                    prediction_2,
                    projection_1.detach(),
                    projection_2.detach(),
                )
            else:
                sequence_embedding_1 = self.sup_head(sequence_embedding_1)
                sequence_embedding_2 = self.sup_head(sequence_embedding_2)
                sequence_embedding_1 = F.normalize(sequence_embedding_1, dim=1)
                sequence_embedding_2 = F.normalize(sequence_embedding_2, dim=1)
                loss_contrastive = self.loss_contrastive_fct(
                    sequence_embedding_1, sequence_embedding_2
                )

        # calculate total loss
        loss = None
        if is_lm_training:
            loss = loss_clm
        if is_sup_training:
            loss = loss + loss_contrastive if loss is not None else loss_contrastive
        if is_fep_training:
            loss = loss + loss_fep if loss is not None else loss_fep
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return RetNetCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            fep_logits=fep_logits,
            prev_states=outputs.prev_states,
            hidden_states=outputs.hidden_states,
            retentions=outputs.retentions,
        )

    def sample_token(self, logit, do_sample=False, top_k=1, top_p=1.0, temperature=1.0):
        if not do_sample:
            return torch.argmax(logit, dim=-1)
        filtered = top_k_top_p_filtering(logit / temperature, top_k=top_k, top_p=top_p)
        return torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens=100,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        num_beams=1,
    ):
        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        generated = []
        outputs_prompt = self(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            forward_impl="chunkwise",
            use_cache=True,
        )
        sequence_offset = input_ids.shape[1] - 1
        token = input_ids[:, -1].unsqueeze(1)
        attention_mask = attention_mask[:, -1].unsqueeze(1)
        prev_states = outputs_prompt.prev_states

        for i in range(max_new_tokens):
            outputs = self(
                input_ids=token,
                forward_impl="recurrent",
                prev_states=prev_states,
                use_cache=True,
                return_dict=True,
                sequence_offset=sequence_offset,
            )
            logit = outputs.logits
            prev_states = outputs.prev_states
            sequence_offset += 1
            token = self.sample_token(
                logit,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ).unsqueeze(1)
            generated.append(token)

        generated = torch.cat(generated, dim=-1)
        return generated
