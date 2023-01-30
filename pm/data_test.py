import unittest

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
        ds = data_module.create_comparison_dataset('CarperAI/openai_summarize_comparisons', split='train')
        self.assertIsNotNone(ds)
        self.assertEqual(len(ds), 92534)


if __name__ == '__main__':
    unittest.main()
