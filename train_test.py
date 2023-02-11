"""Tests for training module."""

import pathlib
import unittest

import torch.nn as nn
from torch.utils.data import default_collate
import transformers

from pm.data import get_tokenizer
from pm.data import PairwiseDataset
from pm.loss import compute_reward_metrics
from pm.model import RewardModel
from pm.utils import get_args_parser
from pm.utils import HParams, TrainingArguments
from pm.utils import maybe_get_subset_dataset


class DummyNN(nn.Module):
    def __init__(self):
        super(DummyNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Embedding(10, 3),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, mask=None, labels=None):
        chosen = input_ids['chosen']
        rejected = input_ids['rejected']

        # Limit to a small range for testing.
        chosen[chosen > 9] = 9
        rejected[rejected > 9] = 9

        ha = self.layers(chosen).mean()
        hb = self.layers(rejected).mean()

        loss = self.loss_fn(ha, hb)
        return {
            'loss': loss,
            'diff': (ha - hb).mean()
        }


class TestTrain(unittest.TestCase):
    def test_hf_trainer_dummy(self):
        def compute_metrics(eval_pred):
            import random
            del eval_pred  # Unused.

            # Dummy metric.
            return {'accuracy': random.random()}

        model = DummyNN()
        training_args = transformers.TrainingArguments(
            output_dir="/tmp/",
            num_train_epochs=1,
            logging_steps=5,
            learning_rate=1e-5,
            evaluation_strategy='steps',
            eval_steps=10,
            per_device_train_batch_size=2,
            torch_compile=False,
            save_total_limit=3,
            save_strategy="steps",
            save_steps=10,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True  # For loss only. Need to change if using accuracy.
        )
        tokenizer = get_tokenizer('EleutherAI/gpt-j-6B')
        train_ds = PairwiseDataset('CarperAI/openai_summarize_comparisons',
                                   tokenizer=tokenizer,
                                   max_length=10,
                                   split='train')
        eval_ds = PairwiseDataset('CarperAI/openai_summarize_comparisons',
                                  tokenizer=tokenizer,
                                  max_length=10,
                                  split='test')
        # Small train and eval set for testing.
        train_ds = maybe_get_subset_dataset(train_ds, 0.000001)
        eval_ds = maybe_get_subset_dataset(eval_ds, 0.0000001)

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=default_collate,
            compute_metrics=compute_metrics
        )
        trainer.train()
        done_training = True

        self.assertTrue(done_training)

    def test_hf_trainer_gpt2(self):
        training_args = transformers.TrainingArguments(
            output_dir="out",
            logging_dir="logs",
            num_train_epochs=1,
            logging_steps=1,
            learning_rate=1e-5,
            evaluation_strategy='steps',
            eval_steps=1,
            per_device_train_batch_size=2,
            torch_compile=False,
            save_total_limit=1,
            save_strategy="steps",
            save_steps=1,
            metric_for_best_model="eval_rank_accuracy",
            greater_is_better=True,  # For loss only. Need to change if using accuracy.
        )
        hparams = HParams(pretrained_model='gpt2',
                          tokenizer_type='gpt2',
                          dataset_type='CarperAI/openai_summarize_comparisons',
                          max_length=600,
                          eval_fraction=0.000001,
                          train_fraction=0.00001,
                          root_dir='/tmp/')

        # Switch TrainingArgs dir to HParams root_dir.
        setattr(training_args, 'output_dir',  str(pathlib.Path(hparams.root_dir) / training_args.output_dir))
        setattr(training_args, 'logging_dir',  str(pathlib.Path(hparams.root_dir) / training_args.logging_dir))

        # Initialize the reward model.
        model = RewardModel(hparams)
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
            compute_metrics=compute_reward_metrics,
        )
        trainer.train()
        done_training = True

        self.assertTrue(done_training)


if __name__ == '__main__':
    unittest.main()
