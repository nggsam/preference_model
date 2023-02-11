"""Tests for utils module."""

import unittest

from pm.utils import HParams
from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.utils import maybe_get_subset_dataset


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hparams = HParams(pretrained_model='gpt2', tokenizer_type='gpt2')

        # Load a subset of dataset for some dummy data.
        tokenizer = get_tokenizer(self.hparams.tokenizer_type)
        self.dataset = PairwiseDataset(self.hparams.dataset_type, tokenizer=tokenizer, max_length=400, split='test')

    # TODO: Check this test.
    def test_maybe_get_subset_dataset(self):
        for fraction in (1.0, 0.1, 0.0):
            with self.subTest(fraction=fraction):
                ds = maybe_get_subset_dataset(self.dataset, fraction)
                self.assertEqual(len(ds), int(fraction * len(self.dataset)))

    def test_maybe_get_subset_dataset_wrong_fraction(self):
        for fraction in (-0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(Exception) as context:
                    maybe_get_subset_dataset(self.dataset, fraction)

                self.assertIsNotNone(context.exception)


if __name__ == '__main__':
    unittest.main()
