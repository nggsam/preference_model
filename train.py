"""Trains a preference model.

This is the main entry point to train a preference model (or reward model). WIP.
"""

from transformers import TrainingArguments

from pm.model import GPTRewardModel
from pm.utils import HParams
from pm.utils import get_args_parser

if __name__ == "__main__":
    parser = get_args_parser()
    hparams: HParams = parser.parse_args_into_dataclasses()[0]
    print(hparams)

    training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=False,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="deepspeed_config.json" if hparams.use_deepspeed else None,
        save_total_limit=1,
    )

    print(hparams)
    # Initialize the reward model.
    model = GPTRewardModel(hparams)


