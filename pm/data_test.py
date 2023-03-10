"""Tests for data module."""

import unittest

import torch
import torch.utils.data as torch_data

from pm.data import load_dataset
from pm.data import process_openai_summarize_comparisons
from pm.data import get_tokenizer
from pm.data import PairwiseDataset


class TestPairwiseDataset(unittest.TestCase):
    def test_get_tokenizer(self):
        for tokenizer_type in ('EleutherAI/gpt-j-6B',):
            with self.subTest(tokenizer_type=tokenizer_type):
                tokenizer = get_tokenizer(tokenizer_type)
                self.assertIsNotNone(tokenizer)
                self.assertTrue(getattr(tokenizer, 'pad_token'))
                # TODO: Add more tests with tokenize results.

    def test_create_summarize_comparison_dataset_train(self):
        dataset = load_dataset('CarperAI/openai_summarize_comparisons', split='train')
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 92534)

    def test_create_summarize_comparison_dataset_test(self):
        dataset = load_dataset('CarperAI/openai_summarize_comparisons', split='test')
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 83629)

    def test_dataloader(self):
        batch_size = 16
        max_length = 520
        tokenizer = get_tokenizer('EleutherAI/gpt-j-6B')

        # Make pairwise datasets for training
        ds = PairwiseDataset('CarperAI/openai_summarize_comparisons',
                             tokenizer,
                             max_length=max_length,
                             split='test')

        dl = torch_data.DataLoader(ds, batch_size=batch_size, shuffle=False)

        batch = next(iter(dl))
        self.assertIsNotNone(batch)
        # Expected keys.
        self.assertSetEqual(set(batch.keys()), {'input_ids', 'mask', 'divergence_index', 'labels'})
        # Expected shapes.
        self.assertEqual(batch['input_ids']['chosen'].shape,
                         (batch_size, max_length))
        self.assertEqual(batch['mask']['chosen'].shape,
                         (batch_size, max_length))
        self.assertEqual(batch['divergence_index'].shape,
                         (batch_size,))

        # Expected true lengths (before padding).
        lengths = batch['mask']['chosen'].sum(dim=1)
        lengths = lengths.unsqueeze(-1)  # Unsqueeze for torch.gather.
        input_ids = batch['input_ids']['chosen']
        # Indexing into lengths should result in padding tokens only.
        gathered_indices = torch.gather(input_ids, dim=1, index=lengths).squeeze(-1)
        all_pad_token_ids = torch.tensor(batch_size * [tokenizer.pad_token_id], dtype=gathered_indices.dtype)
        self.assertTrue(torch.equal(gathered_indices, all_pad_token_ids))


if __name__ == '__main__':
    unittest.main()
