import copy
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertIntermediate,
    BertModel,
    BertOnlyMLMHead,
    BertOutput,
    BertPreTrainedModel,
    BertSelfOutput,
)
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import ModelOutput


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrastive_type="HardNeg"):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrastive_type = contrastive_type
        self.eps = 1e-08

    def forward(self, features_1, features_2):
        batch_size = features_1.shape[0]

        mask = torch.eye(batch_size, dtype=torch.bool).to(features_1.device)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        all_sim = torch.mm(features_1, features_2.t().contiguous())
        neg = (
            torch.exp(all_sim / self.temperature)
            .masked_select(mask)
            .view(batch_size, -1)
        )

        if self.contrastive_type == "Orig":
            Ng = neg.sum(dim=-1)
            loss_pos = (-torch.log(pos / (Ng + pos))).sum() / (batch_size)
            return loss_pos

        elif self.contrastive_type == "HardNeg":
            negimp = neg.log().exp()
            Ng = (negimp * neg).sum(dim=-1) / negimp.mean(dim=-1)
            loss_pos = (-torch.log(pos / (Ng + pos))).sum() / (batch_size)
            return loss_pos

        else:
            raise Exception("Please specify the contrastive loss, Orig vs. HardNeg.")


@dataclass
class PreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_mlm: Optional[torch.FloatTensor] = None
    loss_contrastive: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_embedding: Optional[Tuple[torch.FloatTensor]] = None


class TimeEmbeddings(nn.Module):
    """
    The module generate time embeddings from the time_ids.
    """

    def __init__(self, time_embedding_size):
        super().__init__()
        # add one more embedding position for special tokens: [CLS], [SEP], and [PAD]
        self.holiday_embeddings = nn.Embedding(2 + 1, time_embedding_size["holiday"])
        self.month_embeddings = nn.Embedding(12 + 1, time_embedding_size["month"])
        self.weekday_embeddings = nn.Embedding(7 + 1, time_embedding_size["weekday"])

    """
    Assume the time_ids is in the shape of [batch_size, 5, sequence_length] with the 2nd dimension ordered as:
    0: whether the timestamp is a holiday
    1: month of the timestamp
    2: weekday of the timestamp
    3: sine value of the normalized hour_minute (e.g., 13:35 -> -0.4050)
    4: cosine value of the normalized hour_minute (e.g., 13:35 -> -0.9143)

    The shape of the output is [batch_size, sequence_length, num_embeddings]
    , where num_embeddings=holiday_embedding_size + month_embedding_size + weekday_embedding_size + 2 (sin and cos)
    """

    def forward(self, time_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:

        return torch.concat(
            (
                self.holiday_embeddings(time_ids[:, 0, :].long()),
                self.month_embeddings(time_ids[:, 1, :].long()),
                self.weekday_embeddings(time_ids[:, 2, :].long()),
                time_ids[:, 3, :].unsqueeze(-1),
                time_ids[:, 4, :].unsqueeze(-1),
            ),
            dim=-1,
        )


class EventBertEmbeddings(nn.Module):
    """
    Construct the embeddings for words and time(optional), ignoring position.

    There are no positional embeddings since we use ALiBi. We also get rid of token type embeddings since we
    are unlikely to handle the cases when multiple sequences are concatenated as input.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertEmbeddings`. The key change is
    that position embeddings are removed. Position information instead comes
    from attention biases that scale linearly with the position distance
    between query and key tokens.

    This module ignores the `position_ids` input to the `forward` method.

    This module also allows the future customization of time-awared embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.use_time = config.use_time
        self.use_alibi = config.use_alibi

        # initialize optional time embedding
        if self.use_time:
            self.time_embedding_size = config.time_embedding_size
            self.time_embeddings = TimeEmbeddings(self.time_embedding_size)
            # use a linear projector to map the concatenated word and time embedding back to hidden_size
            self.embeddings_proj = nn.Linear(
                config.hidden_size + sum(self.time_embedding_size.values()),
                config.hidden_size,
            )

        # initialize optional positional information
        if not self.use_alibi:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.position_embedding_type = getattr(
                config, "position_embedding_type", "absolute"
            )
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
                persistent=False,
            )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        time_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        word_embeddings = (
            self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        )

        # add optional time embedding
        if self.use_time:
            time_embeddings = self.time_embeddings(time_ids)
            embeddings = self.embeddings_proj(
                torch.concat((word_embeddings, time_embeddings), dim=-1)
            )
        else:
            embeddings = word_embeddings

        # add optional positional embedding
        if not self.use_alibi:
            if position_ids is None:
                position_ids = self.position_ids[
                    :,
                    past_key_values_length : input_ids.size()[1]
                    + past_key_values_length,
                ]
            embeddings += self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EventBertSelfAttention(nn.Module):
    """
    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertSelfAttention`,
    but with substantial modifications to implement ALiBi.

    Compared to the analogous Hugging Face BERT module, this module pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # apply attention_mask and ALiBi
        attention_scores = attention_scores + bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class EventBertAttention(nn.Module):
    """
    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertAttention`,
    but with substantial modifications to implement ALiBi.

    Compared to the analogous Hugging Face BERT module, this module pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = EventBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            bias,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class EventBertLayer(nn.Module):
    """
    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertLayer`,
    but with substantial modifications to implement ALiBi.

    Compared to the analogous Hugging Face BERT module, this module pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EventBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            bias=bias,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EventBertBertEncoder(nn.Module):
    """
    A stack of BERT layers with ALiBi.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertEncoder`,
    but with substantial modifications to implement ALiBi.

    Compared to the analogous Hugging Face BERT module, this module pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config):
        super().__init__()
        layer = EventBertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )
        self.use_alibi = config.use_alibi
        self.num_attention_heads = config.num_attention_heads

        if self.use_alibi:
            # The alibi mask will be dynamically expanded if it is too small for
            # the input the model receives. But it generally helps to initialize it
            # to a reasonably large size to help pre-allocate CUDA memory.
            # The default `alibi_starting_size` is 512.
            self._current_alibi_size = int(config.alibi_starting_size)
            # head-wise attention bias for attention calculate, with a shape of [batch_size, num_heads, seq_len, seq_len]
            self.alibi = torch.zeros(
                (
                    1,
                    self.num_attention_heads,
                    self._current_alibi_size,
                    self._current_alibi_size,
                )
            )
            self.rebuild_alibi_tensor(size=config.alibi_starting_size)

    def rebuild_alibi_tensor(
        self, size: int, device: Optional[Union[torch.device, str]] = None
    ):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:
            # generate a sequence of slopes as described in paper https://arxiv.org/abs/2108.12409
            # e.g., n_heads = 8, generate [1/2^1, 1/2^2, ..., 1/2^8]
            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        batch, seqlen = hidden_states.shape[:2]

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
        )  # fp16 compatibility
        # minus a large number to the attention scores to prevent attention on the masked area
        LARGE_NUMBER_TO_PREVENT_ATTENTION = -10000.0
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * LARGE_NUMBER_TO_PREVENT_ATTENTION

        if self.use_alibi:
            # Add alibi matrix to extended_attention_mask
            if self._current_alibi_size < seqlen:
                # Rebuild the alibi tensor when needed
                warnings.warn(
                    f"Increasing alibi size from {self._current_alibi_size} to {seqlen}"
                )
                self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
            elif self.alibi.device != hidden_states.device:
                # Device catch-up
                self.alibi = self.alibi.to(hidden_states.device)
            alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
            attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
            bias = attn_bias + alibi_bias
        else:
            bias = extended_attention_mask[:, :, :seqlen, :seqlen]

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                bias=bias,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class EventBertModel(BertPreTrainedModel):
    """
    The BERT model with ALiBi supports.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertModel`,
    but with substantial modifications to implement ALiBi.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.use_alibi = (
            config.use_alibi if "use_alibi" in config.to_dict().keys() else False
        )
        self.use_time = (
            config.use_time if "use_time" in config.to_dict().keys() else False
        )

        self.embeddings = EventBertEmbeddings(config)
        self.encoder = EventBertBertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        time_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            time_ids=time_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class EventBertForMaskedLM(BertPreTrainedModel):
    """
    This class extents BertForMaskedL to support same user prediction based on contrastive learning.
    When same user prediction is activated, we always assume the odd rows of the inputs represents the first event segment of all the users
        and the even rows represents the second user segments of the users,
        while every two continous rows represents the segment of the same user.
    If there are 3 users in the batch, inputs is ordered as: [u1_p1, u1_p2, u2_p1, u2_p2, u3_p1, u3_p2]
        where ui_sj represents the j-th event segment of the i-th user.
    """

    def __init__(self, config):
        super().__init__(config)

        # Specif the training objective
        self.do_mlm = config.training_objective in [
            "MLM",
            "Both",
        ]  # whether to do Masked Language Modeling
        self.do_sup = config.training_objective in [
            "SUP",
            "Both",
        ]  # whether to do Same User Pediction

        self.bert = EventBertModel(config, add_pooling_layer=False)

        # Masked Language Modeling head and loss
        self.cls = BertOnlyMLMHead(config)
        self.loss_mlm_fct = CrossEntropyLoss()

        # Same User Prediction head and loss
        self.sup_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.contrastive_head_size, bias=False),
        )
        self.loss_contrastive_fct = ContrastiveLoss(
            temperature=config.contrastive_temperature,
            contrastive_type=config.contrastive_type,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        time_ids: Optional[torch.LongTensor] = None,
        is_sup_training: Optional[bool] = False,
        is_lm_training: Optional[bool] = False,
        return_loss: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # calculate sequence embedding from BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            time_ids=time_ids,
        )

        token_embedding = outputs[0]  # shape: (batch_size, seq_len, hidden_size)
        attention_mask = attention_mask.unsqueeze(-1)
        sequence_embedding = torch.sum(
            token_embedding * attention_mask, dim=1
        ) / torch.sum(
            attention_mask, dim=1
        )  # shape: (batch_size, hidden_size)

        # calculate MLM loss on sequence
        loss_mlm = None
        mlm_scores = self.cls(token_embedding)
        if labels is not None and self.do_mlm:
            loss_mlm = self.loss_mlm_fct(
                mlm_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        # calculate contrastive loss between events of diffrent users
        # when contrastive loss is used, the input sequence contains two event sequences for each user, i.e., [u1_p1, u1_p2, u2_p1, u2_p2, u3_p1, u3_p2]
        loss_contrastive = None
        output_embedding = None
        if is_sup_training:
            assert sequence_embedding.shape[0] % 2 == 0
            sequence_embedding_1 = sequence_embedding[::2]  # select the odd rows
            sequence_embedding_2 = sequence_embedding[1::2]  # select the even rows
            sequence_embedding_1 = self.sup_head(sequence_embedding_1)
            sequence_embedding_2 = self.sup_head(sequence_embedding_2)
            sequence_embedding_1 = F.normalize(sequence_embedding_1, dim=1)
            sequence_embedding_2 = F.normalize(sequence_embedding_2, dim=1)
            loss_contrastive = self.loss_contrastive_fct(
                sequence_embedding_1, sequence_embedding_2
            )
        else:
            output_embedding = self.sup_head(sequence_embedding)

        # calculate total loss
        loss = None
        if is_sup_training and is_lm_training:
            loss = loss_mlm + loss_contrastive
        elif is_lm_training:
            loss = loss_mlm
        elif is_sup_training:
            loss = loss_contrastive

        if not return_dict:
            output = (mlm_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PreTrainingOutput(
            loss=loss,
            logits=mlm_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            output_embedding=output_embedding,
        )
