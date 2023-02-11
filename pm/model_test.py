"""Tests for data module."""

import unittest

from torch.utils.data import Subset, DataLoader

from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.model import RewardModel
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
        model = RewardModel(self.hparams)
        self.assertIsNotNone(model)

    # TODO: Check this test.
    def test_model_forward(self):
        model = RewardModel(self.hparams)
        batch = next(iter(self.dl))

        model(**batch)



if __name__ == '__main__':
    unittest.main()
