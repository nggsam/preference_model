"""Tests for utils module."""

import unittest

import torch.nn as nn

from pm.utils import HParams
from pm.data import PairwiseDataset
from pm.data import get_tokenizer
from pm.utils import maybe_freeze_layers
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
                expected_length = int(fraction * len(self.dataset))
                expected_length = max(expected_length, 1)
                self.assertEqual(len(ds), expected_length)

    def test_maybe_get_subset_dataset_wrong_fraction(self):
        for fraction in (-0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(Exception) as context:
                    maybe_get_subset_dataset(self.dataset, fraction)

                self.assertIsNotNone(context.exception)

    def test_maybe_freeze_layers_wrong_ratio(self):
        for freeze_ratio in (-0.1, 1.1):
            with self.subTest(freeze_ratio=freeze_ratio):
                with self.assertRaises(Exception) as context:
                    maybe_freeze_layers(None, freeze_ratio)

                self.assertIsNotNone(context.exception)

    def test_maybe_freeze_layers(self):
        for freeze_ratio in (0.0, 0.5, 1.0):
            with self.subTest(freeze_ratio=freeze_ratio):
                fake_hidden_layers = nn.Sequential(
                            nn.Linear(3, 7),
                            nn.Dropout(0.1),
                            nn.Linear(7, 5)
                        )
                mock_model = unittest.mock.Mock()
                mock_model.configure_mock(**{'transformer.h': fake_hidden_layers})
                model = maybe_freeze_layers(mock_model, freeze_ratio=freeze_ratio)

                num_frozen = int(freeze_ratio * len(model.transformer.h))
                for layer in model.transformer.h[:num_frozen]:
                    # TODO: Make this recursive.
                    if hasattr(layer, 'weight'):
                        self.assertFalse(layer.weight.requires_grad)

                for layer in model.transformer.h[num_frozen:]:
                    if hasattr(layer, 'weight'):
                        self.assertTrue(layer.weight.requires_grad)




if __name__ == '__main__':
    unittest.main()
