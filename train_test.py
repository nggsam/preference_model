"""Tests for training module."""

import unittest
import dataclasses

from transformers import TrainingArguments, HfArgumentParser

from train import HParams


class TestTrain(unittest.TestCase):
    def test_hparams_parser(self):
        parser = HfArgumentParser(HParams)
        hparams = parser.parse_args_into_dataclasses()[0]  # First index is HParams class.
        self.assertEqual(hparams.pretrained_model, 'CarperAI/openai_summarize_tldr_sft')

if __name__ == '__main__':
    unittest.main()
