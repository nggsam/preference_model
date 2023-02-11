import copy
import dataclasses

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import TrainingArguments

Tensor = torch.Tensor


@dataclasses.dataclass
class HParams:
    """Hyperparameters class with default values."""
    pretrained_model: str = 'CarperAI/openai_summarize_tldr_sft'
    dataset_type: str = 'CarperAI/openai_summarize_comparisons'
    tokenizer_type: str = 'EleutherAI/gpt-j-6B'
    use_deepspeed: bool = False
    max_length: int = '300'
    root_dir: str = './'
    eval_fraction: float = 1.0
    train_fraction: float = 1.0


def get_args_parser():
    """Initializes args parser and add arguments."""
    parser = HfArgumentParser([HParams, TrainingArguments])

    return parser


def batch_get_mask_equal_or_larger_than_indices(matrix, indices):
    """Gets mask larger than indices in a batch fashion.

    Args:
      matrix: a 2D with [batch_size, dimension]
      indices: 1D index tensor, with values indicating the start of the threshold.
    Returns:
      A 2D matrix with mask of [0, 1], with 0 indicating values smaller than
        the index for each row.
    """
    assert len(matrix.shape) == 2
    batch_size, dim = matrix.shape

    assert len(indices.shape) == 1
    assert indices.shape[0] == batch_size

    # Turn indices into the same shape as matrix A.
    indices: Tensor = indices.unsqueeze(1).repeat(1, dim)
    # Get a 2D matrix that goes from 0 to dim-1 for each row of batch_size.
    arange: Tensor = torch.arange(dim).tile(batch_size, 1).to(matrix.device)
    # Calculate the mask
    return (arange >= indices).type(torch.long)


def make_tensor(values, dtype) -> torch.Tensor:
    """Create PyTorch tensors with dtype."""
    return torch.tensor(values, dtype=dtype)


def merge_training_args(source: TrainingArguments, to: TrainingArguments):
    """Merges values in dataclass fr into to and returns a copy of to."""
    assert type(source) == type(to)

    cp = copy.deepcopy(to)
    for key in source.__dataclass_fields__:
        cp.__dict__[key] = source.__dict__[key]

    return cp


def maybe_get_subset_dataset(dataset: Dataset, fraction: float):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Fraction has to be: 0.0 <= {fraction} <= 1.0.")

    if fraction < 1.0:
        subset_length = max(int(len(dataset) * float(fraction)), 1)
        ds = Subset(dataset, range(subset_length))
        return ds
    else:
        return dataset


def seed_everything(seed: int) -> None:
    """Seeds PyTorch, random and Numpy."""
    import torch
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
