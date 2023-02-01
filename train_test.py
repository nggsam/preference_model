"""Tests for training module."""

import unittest

import torch.nn as nn
from torch.utils.data import default_collate
from torch.utils.data import Subset
import transformers

from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.utils import get_args_parser


class DummyNN(nn.Module):
    def __init__(self):
        super(DummyNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Embedding(10, 3),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, mask=None):
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
    def test_hparams_parser(self):
        parser = get_args_parser()
        hparams = parser.parse_args_into_dataclasses()[0]  # First index is HParams class.
        self.assertEqual(hparams.pretrained_model, 'CarperAI/openai_summarize_tldr_sft')
        self.assertEqual(hparams.tokenizer_type, 'EleutherAI/gpt-j-6B')

    def test_hf_trainer(self):

        model = DummyNN()
        training_args = transformers.TrainingArguments(
            output_dir="/tmp/",
            num_train_epochs=1,
            logging_steps=1,
            learning_rate=1e-5,
            max_steps=5,
            evaluation_strategy='epoch',
            per_device_train_batch_size=2
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
        # Small eval set for testing.
        eval_ds = Subset(eval_ds, range(10))
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=default_collate
        )
        trainer.train()
        done_training = True

        self.assertTrue(done_training)



if __name__ == '__main__':
    unittest.main()
