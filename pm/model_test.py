"""Tests for data module."""

import unittest

from torch.utils.data import Subset, DataLoader

from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.model import BaseRewardModel, PerTokenRewardModel, PoolRewardModel
from pm.utils import HParams


class RewardModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hparams = HParams(pretrained_model='gpt2', tokenizer_type='gpt2')

        # Load a subset of dataset for some dummy data.
        tokenizer = get_tokenizer(self.hparams.tokenizer_type)
        ds = PairwiseDataset(self.hparams.dataset_type, tokenizer=tokenizer, max_length=400, split='test')
        self.ds = Subset(ds, range(16))  # Subset of 16 items.
        self.dl = DataLoader(self.ds, batch_size=4, shuffle=False)

    def test_model_init(self):
        model = BaseRewardModel(self.hparams)
        self.assertIsNotNone(model)

    def test_per_token_model_forward(self):
        model = PerTokenRewardModel(self.hparams)
        batch = next(iter(self.dl))
        output = model(**batch)
        self.assertIsNotNone(output)

    def test_pool_model_forward(self):
        model = PoolRewardModel(self.hparams)
        batch = next(iter(self.dl))
        output = model(**batch)
        self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main()
