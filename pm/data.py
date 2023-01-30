from typing import Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_TYPES = {
    'EleutherAI/gpt-j-6B': 'EleutherAI/gpt-j-6B'
}

DATASET_TYPES = {
    'CarperAI/openai_summarize_comparisons': 'CarperAI/openai_summarize_comparisons'
}

SPLIT_TYPES = {'train', 'test'}


def _process_openai_summarize_comparisons(sample):
    """Preprocesses OpenAI Summarize Comparisons dataset.

    Args:
        sample: a raw item from dataset.
    """
    pair = {}
    prompt = sample["prompt"]
    chosen_summary = sample["chosen"]
    rejected_summary = sample["rejected"]
    if chosen_summary == rejected_summary:
        return None
    if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
        return None
    pair["chosen"] = prompt + "\n" + chosen_summary
    pair["rejected"] = prompt + "\n" + rejected_summary

    return pair


def get_tokenizer(tokenizer_type: str):
    """Gets tokenizer and add some possible missing tokens (padding, etc)."""
    assert tokenizer_type in TOKENIZER_TYPES

    if tokenizer_type == 'EleutherAI/gpt-j-6B':
        tokenizer_path = TOKENIZER_TYPES[tokenizer_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
    else:
        raise ValueError(tokenizer_type)


def preprocess_dataset(dataset_type, split) -> Sequence[dict[str, str]]:
    """Preprocesses dataset into expected format."""
    assert dataset_type in DATASET_TYPES
    assert split in SPLIT_TYPES

    dataset_path = DATASET_TYPES[dataset_type]

    if dataset_type == 'CarperAI/openai_summarize_comparisons':
        dataset = load_dataset(dataset_path, split=split)
        # TODO: replace this upfront cost with iterable option (only preprocess when loading).
        return _process_openai_summarize_comparisons(dataset)
    else:
        raise ValueError(dataset_type)


class PairwiseDataset(Dataset):
    """Dataset that yields each item as a dict of ['input_ids', 'mask']."""
    def __init__(self, pairs, tokenizer, max_length):
        """Initializes the dataset.

        Args:
            pairs: a dict of ['chosen', 'rejected'], whose values are string.
            tokenizer: a tokenizer object.
            max_length: an int that is the maximum length for tokenizer's padding operation.
        """
        def _process(item: str) -> dict:
            return tokenizer(
                f"{tokenizer.bos_token} {item} {tokenizer.eos_token}",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

        self.items = []

        for pair in tqdm(pairs):
            input_ids = {}
            mask = {}
            for item_type in ('chosen', 'rejected'):
                encoded_dict = _process(pair[item_type])
                input_ids[item_type] = encoded_dict['input_ids']
                mask[item_type] = encoded_dict['attention_mask']

            self.items.append({'input_ids': input_ids, 'mask': mask})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DataCollatorReward:
    def __call__(self, data):
        batch = {
            "input_ids": data['input_ids'],
            "mask": data['mask']
        }
        return batch
