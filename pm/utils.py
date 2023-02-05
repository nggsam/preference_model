import dataclasses

import torch
from transformers import HfArgumentParser


@dataclasses.dataclass
class HParams:
    """Hyperparameters class with default values."""
    pretrained_model: str = 'CarperAI/openai_summarize_tldr_sft'
    dataset_type: str = 'CarperAI/openai_summarize_comparisons'
    tokenizer_type: str = 'EleutherAI/gpt-j-6B'
    use_deepspeed: bool = False
    max_length: int = '300'
    root_dir: str = './'


def get_args_parser():
    """Initializes args parser and add arguments."""
    parser = HfArgumentParser(HParams)

    return parser


def batch_get_mask_equal_or_larger_than_indices(A, indices):
    """Gets mask larger than indices in a batch fashion.

    Args:
      A: a 2D with [batch_size, dimension]
      indices: 1D index tensor, with values indicating the start of the threshold.
    Returns:
      A 2D matrix with mask of [0, 1], with 0 indicating values smaller than
        the index for each row.
    """
    assert len(A.shape) == 2
    batch_size, dim = A.shape

    assert len(indices.shape) == 1
    assert indices.shape[0] == batch_size

    # Turn indices into the same shape as matrix A.
    indices = indices.unsqueeze(1).repeat(1, dim)
    # Get a 2D matrix that goes from 0 to dim-1 for each row of batch_size.
    arange = torch.arange(dim).tile(batch_size, 1)
    # Calculate the mask
    return (arange >= indices).type(torch.long)


def make_tensor(values, dtype) -> torch.Tensor:
    """Create PyTorch tensors with dtype."""
    return torch.tensor(values, dtype=dtype)
