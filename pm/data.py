from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

TOKENIZER_TYPES = {
    'EleutherAI/gpt-j-6B': 'EleutherAI/gpt-j-6B'
}

DATASET_TYPES = {
    'CarperAI/openai_summarize_comparisons': 'CarperAI/openai_summarize_comparisons'
}

SPLIT_TYPES = {'train', 'test'}


def _process_openai_summarize_comparisons(sample, tokenizer, max_length):
    """Preprocesses OpenAI Summarize Comparisons dataset.

    Args:
        sample: a raw item from dataset.
    """

    def _tokenize(item: str):
        """Tokenizes item into encoding dict with input_ids and mask."""
        return tokenizer(
            f"{tokenizer.bos_token} {item} {tokenizer.eos_token}",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Turns item into the pair format.
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

    # Tokenize and format into input_ids and mask for chosen and rejected string.
    input_ids = {}
    mask = {}
    for item_type in ('chosen', 'rejected'):
        encoded_dict = _tokenize(pair[item_type])
        input_ids[item_type] = encoded_dict['input_ids'][0]  # [1, max_length] so we index into 0 to get [max_length]
        mask[item_type] = encoded_dict['attention_mask'][0]  # [1, max_length] so we index into 0 to get [max_length]

    return {'input_ids': input_ids, 'mask': mask}


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


class PairwiseDataset(Dataset):
    """Dataset that yields each item as a dict of ['input_ids', 'mask']."""

    def __init__(self, dataset_type: str, tokenizer, max_length: int, split: str):
        """Initializes the dataset.

        Args:
            dataset_type: a str for type of dataset to load.
            # dict of ['chosen', 'rejected'], whose values are string.
            tokenizer: a tokenizer object.
            max_length: an int that is the maximum length for tokenizer's padding operation.
        """
        assert dataset_type in DATASET_TYPES
        assert split in SPLIT_TYPES

        self._tokenizer = tokenizer
        self._max_length = max_length
        self._split = split
        dataset_path = DATASET_TYPES[dataset_type]

        if dataset_type == 'CarperAI/openai_summarize_comparisons':
            self._dataset = load_dataset(dataset_path, split=split)
            self._process_fn = _process_openai_summarize_comparisons
        else:
            raise ValueError(dataset_type)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        processed = self._process_fn(sample,
                                     tokenizer=self._tokenizer,
                                     max_length=self._max_length)

        return processed


class DataCollatorReward:
    def __call__(self, data):
        batch = {
            "input_ids": data['input_ids'],
            "mask": data['mask']
        }
        return batch
