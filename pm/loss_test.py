"""Tests for training module."""

import unittest

import torch

from pm.loss import batch_pairwise_loss
from pm.loss import pairwise_loss
from pm.utils import make_tensor


class TestLosses(unittest.TestCase):
    def test_pairwise_loss(self):
        a_rewards = make_tensor([[1.2, 1.3, 1.4], [1.1, 1.2, 1.1]], torch.float)
        a_mask = make_tensor([[1, 1, 1], [1, 1, 1]], torch.long)

        b_rewards = make_tensor([[0.1, 0.2, 1.4], [0.4, 1.5, 1.6]], torch.float)
        b_mask = make_tensor([[1, 1, 0], [1, 1, 1]], torch.long)

        divergence_index = make_tensor([1, 2], torch.long)

        loss_dict = pairwise_loss(a_rewards, b_rewards, a_mask, b_mask, divergence_index)

        expected_a_last_rewards = make_tensor([1.4, 1.1], torch.float)
        expected_b_last_rewards = make_tensor([1.4, 1.6], torch.float)
        first_row = -torch.log(torch.sigmoid(a_rewards[0][1:3] - b_rewards[0][1:3])).mean()
        second_row = -torch.log(torch.sigmoid(a_rewards[1][2:3] - b_rewards[1][2:3])).mean()
        expected_loss = (first_row + second_row) / 2.

        self.assertTrue(torch.equal(loss_dict['chosen_last_rewards'], expected_a_last_rewards))
        self.assertTrue(torch.equal(loss_dict['rejected_last_rewards'], expected_b_last_rewards))
        self.assertEqual(loss_dict['loss'], expected_loss)

    def test_batch_pairwise_loss(self):
        a_rewards = make_tensor([[1.2, 1.3, 1.4], [1.1, 1.2, 1.1]], torch.float)
        a_mask = make_tensor([[1, 1, 1], [1, 1, 1]], torch.long)

        b_rewards = make_tensor([[0.1, 0.2, 1.4], [0.4, 1.5, 1.6]], torch.float)
        b_mask = make_tensor([[1, 1, 0], [1, 1, 1]], torch.long)

        divergence_index = make_tensor([1, 2], torch.long)

        loss_dict = batch_pairwise_loss(a_rewards, b_rewards, a_mask, b_mask, divergence_index)

        expected_a_last_rewards = make_tensor([1.4, 1.1], torch.float)
        expected_b_last_rewards = make_tensor([1.4, 1.6], torch.float)
        first_row = -torch.log(torch.sigmoid(a_rewards[0][1:3] - b_rewards[0][1:3])).mean()
        second_row = -torch.log(torch.sigmoid(a_rewards[1][2:3] - b_rewards[1][2:3])).mean()
        expected_loss = (first_row + second_row) / 2.

        self.assertTrue(torch.equal(loss_dict['chosen_last_rewards'], expected_a_last_rewards))
        self.assertTrue(torch.equal(loss_dict['rejected_last_rewards'], expected_b_last_rewards))
        self.assertEqual(loss_dict['loss'], expected_loss)


if __name__ == '__main__':
    unittest.main()
