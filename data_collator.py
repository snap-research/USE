"""
Code credit: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
"""

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# This function implement two purposes:
# 1. Generate the future event prediction labels for a given inputs sequence
# 2. Return a re-padded version of ids. This is needed when the padding token appears in ids.
#    In this case, the fep label for the tokens before padding is not accurate
#    since there are less than "fep_window_size" non-padding tokens after it.
#    Therefore, we find the first padding token and set its previous "fep_window_size" also as padding
#    to aviod training model on invalid fep labels
#
# Input:
#     ids: a List "input_seq_length + fep_window_size" token.
#     eid2pos: a map with token id as key and its corresponding position in the fep label
#     fep_window_size: length of future we want to predict
#     pad_token_id: the id of pad token in the tokenizer
# this function generate a fep label for every token in the first "input_seq_length" positions
# for efficiently generate the label, we use a counter to keep track of the event existance information
# since the fep label tends to be very sparse, to save storage, we only record the position existence events here
# the real fep label will be generated in the following _decode_fep_labels function in the training process.
def generate_fep_labels(ids, eid2pos, fep_window_size=100, pad_token_id=683):
    input_seq_length = len(ids) - fep_window_size

    try:
        first_padding_position = ids.index(pad_token_id)
    # when pad_token_id not exist, will trigger a ValueError
    except ValueError:
        first_padding_position = -1
    if first_padding_position == -1:
        nonpadding_seq_length = input_seq_length
    else:
        nonpadding_seq_length = first_padding_position - 100

    counter = collections.Counter(ids[:fep_window_size])
    seq_labels = []
    for i in range(nonpadding_seq_length):
        cur_labels = []
        counter[ids[i]] -= 1
        counter[ids[i + fep_window_size]] += 1
        if counter[ids[i]] == 0:
            counter.pop(ids[i])

        for e in eid2pos:
            if e in counter:
                cur_labels.append(eid2pos[e])
        seq_labels.append(cur_labels)

    ids = torch.Tensor(ids).long()

    # for the padding tokens, just use empty list as its fep label since we would calculate loss on them
    if nonpadding_seq_length < input_seq_length:
        seq_labels.extend([[] for _ in range(input_seq_length - nonpadding_seq_length)])
        ids[nonpadding_seq_length:] = pad_token_id

    return seq_labels, ids


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        return self.torch_call(features)


def _decode_fep_labels(raw_label, num_fep_events):
    input_seq_length = len(raw_label)
    fep_label = torch.zeros([input_seq_length, num_fep_events])
    for i, idx in enumerate(raw_label):
        if len(idx) == 0:
            break
        fep_label[i][idx] = 1
    return fep_label


def _torch_collate_batch(
    raw_examples,
    tokenizer,
    same_user_prediction: bool = False,
    future_event_prediction: bool = False,
    num_fep_events: int = 686,
    pad_to_multiple_of: Optional[int] = None,
):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    ### For same user prediction, we need to order the two parts of the input sequences properly to ensure it does not mess up in multi-GPU training
    # the data collator here will collect data for all GPUs and then split them into multiple equal-sized blocks and send each block to one GPU
    # here we put the two segment of the same user continuous to make sure they will be send to the same GPU
    # e.g., support we have 2 users and 2 GPUs
    # If the sequence is ordered as [u1_p1, u2_p1, u1_p2, u2_p2], then GPU1 will receive [u1_p1, u2_p1] and GPU2 will receive [u1_p2, u2_p2], which is undesired
    # So we order the sequence as [u1_p1, u1_p2, u2_p1, u2_p2], then GPU1 will receive [u1_p1, u1_p2] and GPU2 will receive [u2_p1, u2_p2]

    if same_user_prediction:
        examples = []
        fep_labels = []
        for example in raw_examples:
            examples.append(example["input_ids_1"])
            examples.append(example["input_ids_2"])
            if future_event_prediction:
                fep_labels.append(
                    _decode_fep_labels(example["fep_labels_1"][0], num_fep_events)
                )
                fep_labels.append(
                    _decode_fep_labels(example["fep_labels_2"][0], num_fep_events)
                )
        if future_event_prediction:
            fep_labels = torch.stack(fep_labels, dim=0)
    else:
        examples = [example["input_ids"].squeeze(0) for example in raw_examples]
        if future_event_prediction:
            fep_labels = torch.stack(
                [
                    _decode_fep_labels(e["fep_labels"][0], num_fep_events)
                    for e in raw_examples
                ],
                dim=0,
            )
        else:
            fep_labels = []

    max_length = max(x.size(0) for x in examples)

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(examples, dim=0), fep_labels, max_length

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result, fep_labels, max_length





@dataclass
class DataCollatorForMLMAndSameUserPrediction(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        same_user_prediction (`bool`, *optional*, defaults to `False`):
            Whether or not to use same user prediction.
        use_time (`bool`, *optional*, defaults to `False`):
            Whether or not to generate time_ids from the timestamps. If set as 'True', where extract
            holiday, month, weekday, hour, and minute information from the raw timestamps and concat them in a
            torch tensor of size [batch_size, 5, sequence_length].
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    clm: bool = False
    mlm_probability: float = 0.15
    same_user_prediction: bool = False
    future_event_prediction: bool = False
    num_fep_events: int = 686
    use_time: bool = False
    pad_to_multiple_of: Optional[int] = None
    model_type: str = "retnet"
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:

        input_ids, fep_labels, max_length = _torch_collate_batch(
            examples,
            self.tokenizer,
            self.same_user_prediction,
            self.future_event_prediction,
            self.num_fep_events,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = {
            "input_ids": input_ids,
        }

        if self.future_event_prediction:
            batch["fep_labels"] = fep_labels

        if self.use_time:
            batch["time_ids"] = get_time_info(
                examples,
                same_user_prediction=self.same_user_prediction,
                max_length=max_length,
            )

        # mask tokens
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if self.mlm:
            batch["input_ids"], batch["labels"] = self._mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        elif self.clm:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        batch["attention_mask"] = batch["input_ids"] != self.tokenizer.pad_token_id
        if self.model_type == "retnet":
            batch["is_lm_training"] = True if (self.clm or self.mlm) else False
            batch["is_sup_training"] = self.same_user_prediction
            batch["is_fep_training"] = self.future_event_prediction

        return batch

    def _mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
