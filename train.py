"""Trains a preference model.

This is the main entry point to train a preference model (or reward model). WIP.
"""

import pathlib

import transformers
from torch.utils.data import default_collate
from torch.utils.data import Subset

from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.model import RewardModel
from pm.utils import HParams
from pm.utils import get_args_parser
from pm.utils import merge_training_args

# TODO: Train to make sure that loss is going down.
# TODO: Add metrics to measure accuracy while training.
# TODO: Try different configs:
#          - Freeze some of the layers to avoid overfitting.
#          - Train first layer for 0.1 epoch. Then train the other layers.
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args_into_dataclasses()
    hparams: HParams = args[0]
    training_args: transformers.TrainingArguments = args[1]
    root_dir = pathlib.Path(hparams.root_dir)

    # Use hparams to override some options in training_args.
    if not hparams.use_deepspeed:
        training_args.deepspeed = None

    # Initialize the reward model.
    model = RewardModel(hparams)

    tokenizer = get_tokenizer(hparams.tokenizer_type)
    # TODO: What's a good max length?
    train_ds = PairwiseDataset(hparams.dataset_type,
                               tokenizer=tokenizer,
                               max_length=hparams.max_length,
                               split='train')
    eval_ds = PairwiseDataset(hparams.dataset_type,
                              tokenizer=tokenizer,
                              max_length=hparams.max_length,
                              split='test')
    subset_eval_ds = Subset(eval_ds, range(int(len(eval_ds) * hparams.eval_fraction)))
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=subset_eval_ds,
        data_collator=default_collate,
    )
    trainer.train()
