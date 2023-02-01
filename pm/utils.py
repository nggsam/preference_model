import dataclasses

from transformers import HfArgumentParser


@dataclasses.dataclass
class HParams:
    """Hyperparameters class with default values."""
    pretrained_model: str = 'CarperAI/openai_summarize_tldr_sft'
    use_deepspeed: bool = False
    dataset_type: str = 'CarperAI/openai_summarize_comparisons'
    tokenizer_type: str = 'EleutherAI/gpt-j-6B'


def get_args_parser():
    """Initializes args parser and add arguments."""
    parser = HfArgumentParser(HParams)
    # parser.add_argument('--pretrained_model', help='Pretrained model to load from initially.')
    # parser.add_argument('--use_deepspeed', help='Whether to enable deepspeed.')
    # parser.add_argument('--dataset_type', help='Type of dataset to train with.')
    # parser.add_argument('--tokenizer_type', help='Type of tokenizer to process data with.')

    return parser
