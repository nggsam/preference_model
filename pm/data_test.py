"""Tests for data module."""

import unittest

import torch.utils.data as torch_data
from pm import data as data_module



class TestPairwiseDataset(unittest.TestCase):
    def test_get_tokenizer(self):
        for tokenizer_type in ('EleutherAI/gpt-j-6B',):
            with self.subTest(tokenizer_type=tokenizer_type):
                tokenizer = data_module.get_tokenizer(tokenizer_type)
                self.assertIsNotNone(tokenizer)
                self.assertTrue(getattr(tokenizer, 'pad_token'))
                # TODO: Add more tests with tokenize results.

    def test_init(self):
        data = [{'chosen': 'prompt item0', 'rejected': 'prompt item1'},
                {'chosen': 'prompt item2', 'rejected': 'prompt item3'}]

        tokenizer = data_module.get_tokenizer('EleutherAI/gpt-j-6B')
        ds = data_module.PairwiseDataset(data, tokenizer, 32)

        self.assertIsNotNone(ds)

    def test_create_summarize_comparison_dataset(self):
        ds = data_module.preprocess_dataset('CarperAI/openai_summarize_comparisons', split='test')
        self.assertIsNotNone(ds)
        self.assertEqual(len(ds), 92534)

    def test_data_collator(self):
        preprocessed = data_module.preprocess_dataset('CarperAI/openai_summarize_comparisons', split='test')
        tokenizer = data_module.get_tokenizer('EleutherAI/gpt-j-6B')

        # Make pairwise datasets for training
        ds = data_module.PairwiseDataset(preprocessed, tokenizer, max_length=30)

        # Create the collator to gather batches of pairwise comparisons
        data_collator = data_module.DataCollatorReward()

        dl = torch_data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=data_collator)

        batch = next(iter(dl))
        self.assertIsNotNone(batch)

if __name__ == '__main__':
    unittest.main()
