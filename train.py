"""Trains a preference model.

This is the main entry point to train a preference model (or reward model). WIP.
"""

import pathlib

import transformers
from torch.utils.data import default_collate

from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.model import GPTRewardModel
from pm.utils import HParams
from pm.utils import get_args_parser

# TODO: Train to make sure that loss is going down.
# TODO: Add metrics to measure accuracy while training.
if __name__ == "__main__":
    parser = get_args_parser()
    hparams: HParams = parser.parse_args_into_dataclasses()[0]
    root_dir = pathlib.Path(hparams.root_dir)

    training_args = transformers.TrainingArguments(
        output_dir=str(root_dir / "pm_checkpoint"),
        num_train_epochs=3,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_total_limit=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir=str(root_dir / "logs"),
        fp16=False,
        bf16=False,
        learning_rate=1e-5,
        deepspeed=str(root_dir / "deepspeed_config.json") if hparams.use_deepspeed else None,
    )

    # Initialize the reward model.
    model = GPTRewardModel(hparams)

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
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_collate,
    )
    trainer.train()
