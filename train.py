"""Trains a preference model.

This is the main entry point to train a preference model (or reward model). WIP.
"""

import time
import pathlib

import transformers
from torch.utils.data import default_collate
import wandb

from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.loss import compute_reward_metrics
from pm.model import get_reward_model
from pm.utils import HParams
from pm.utils import get_args_parser
from pm.utils import maybe_get_subset_dataset
from pm.utils import seed_everything

if __name__ == "__main__":
    # Parse hparams and training_args.
    parser = get_args_parser()
    args = parser.parse_args_into_dataclasses()
    hparams: HParams = args[0]
    training_args: transformers.TrainingArguments = args[1]
    # Add timestamp and expt name to root_dir.
    expt_timestamp = int(time.time())
    expt_name = f'{hparams.expt}_{expt_timestamp}'
    expt_dir = pathlib.Path(hparams.root_dir) / expt_name
    expt_dir.mkdir(exist_ok=True, parents=True)

    # Enable WANDB.
    if hparams.wandb:
        wandb.init(project=hparams.expt, sync_tensorboard=True, dir=expt_dir, name=expt_name)
        wandb.config.update(hparams)
        wandb.config.update(training_args)

    # Seed.
    seed_everything(training_args.seed)

    # Initialize the reward model.
    model = get_reward_model(hparams)

    # Switch TrainingArgs dir to HParams expt_dir.
    setattr(training_args, 'output_dir',  str(expt_dir / training_args.output_dir))
    setattr(training_args, 'logging_dir',  str(expt_dir / training_args.logging_dir))

    tokenizer = get_tokenizer(hparams.tokenizer_type)
    train_ds = PairwiseDataset(hparams.dataset_type,
                               tokenizer=tokenizer,
                               max_length=hparams.max_length,
                               split='train')
    eval_ds = PairwiseDataset(hparams.dataset_type,
                              tokenizer=tokenizer,
                              max_length=hparams.max_length,
                              split='test')

    # Subset train and eval if train/eval_fraction < 1.0.
    train_ds = maybe_get_subset_dataset(train_ds, hparams.train_fraction)
    eval_ds = maybe_get_subset_dataset(eval_ds, hparams.eval_fraction)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_collate,
        compute_metrics=compute_reward_metrics
    )
    trainer.train()
