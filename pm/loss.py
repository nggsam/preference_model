import numpy as np
import torch
import transformers

from pm.utils import batch_get_mask_equal_or_larger_than_indices

Tensor = torch.Tensor


def batch_pairwise_loss(c_rewards, r_rewards, c_mask, r_mask, divergence_index):
    """Calculates batch-style pairwise loss between rewards for chosen and rejected phrases.

    This loss is calculated in a batch-style (without looping) to speed up the code.
    """
    or_mask = torch.logical_or(c_mask, r_mask).type(torch.long)
    or_lengths = or_mask.sum(axis=1)
    or_indices = or_lengths - 1  # To gather the value at the last index of each row.

    d_rewards = (c_rewards - r_rewards)
    c_last_rewards = torch.gather(c_rewards, dim=1, index=or_indices.unsqueeze(-1))
    r_last_rewards = torch.gather(r_rewards, dim=1, index=or_indices.unsqueeze(-1))
    divergence_mask = batch_get_mask_equal_or_larger_than_indices(
        d_rewards, divergence_index
    )
    weights = divergence_mask * or_mask

    loss = -torch.log(torch.sigmoid(d_rewards)) * weights
    # Sum over each row first.
    loss = loss.sum(dim=-1)
    # Normalize row-wise using weights first.
    loss = loss / weights.sum(dim=-1)
    # Normalize with batch size.
    loss = loss.sum() / weights.shape[0]

    return {'loss': loss,
            'chosen_last_rewards': c_last_rewards.squeeze(-1),
            'rejected_last_rewards': r_last_rewards.squeeze(-1)}


def pairwise_loss(c_rewards, r_rewards, c_mask, r_mask, divergence_index):
    """Calculates pairwise loss between rewards for chosen and rejected phrases.

    This loops through each batch item so it might not be as efficient as batch-style.
    """
    assert c_rewards.shape == r_rewards.shape

    c_lengths = c_mask.sum(axis=1)
    r_lengths = r_mask.sum(axis=1)

    batch_size, seq_length = c_rewards.shape

    loss = 0.
    c_last_rewards = []
    r_last_rewards = []
    for i in range(batch_size):
        c_ind = c_lengths[i]
        r_ind = r_lengths[i]
        end_ind = max(c_ind, r_ind)

        divergence_ind = divergence_index[i]
        assert divergence_ind > 0

        # Index into the correct rewards.
        c_truncated_reward = c_rewards[i][divergence_ind:end_ind]
        r_truncated_reward = r_rewards[i][divergence_ind:end_ind]

        # Append the last rewards to the list of end scores.
        c_last_rewards.append(c_truncated_reward[-1])
        r_last_rewards.append(r_truncated_reward[-1])

        # Compute loss.
        loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

    loss = loss / batch_size

    return {'loss': loss,
            'chosen_last_rewards': torch.tensor(c_last_rewards),
            'rejected_last_rewards': torch.tensor(r_last_rewards)}


def binary_crossentropy_ranking_loss(a_rewards, b_rewards, labels):
    """Calculate ranking loss of a vs b based on labels.

    We calculate the ranking loss based on binary crossentropy of difference between a_rewards and b_rewards.
    Label of 1 means the chosen one while 0 means rejected.

    Args:
        a_rewards: 1D tensor of float.
        b_rewards: 1D tensor of float.
        labels: 1D tensor of 0 or 1.
    Returns:
        Binary cross entropy ranking loss of a_rewards and b_rewards based on labels.
    """
    assert a_rewards.shape == b_rewards.shape and a_rewards.shape == labels.shape

    logits = a_rewards - b_rewards
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-logits)

    losses = -1. * (labels * log_p + (1 - labels) * log_not_p)
    # TODO: mean() or sum()?
    return losses.mean()


def reward_ranking_accuracy_metric(a_rewards: Tensor, b_rewards: Tensor):
    """Calculates the average of number of the times where a rewards are higher than b rewards.

    Here we assume a rewards should be larger than b rewards.
    Args:
        a_rewards: 1D tensor of rewards for A.
        b_rewards: 1D tensor of rewards for B.
    Returns:
        Tensor of type float for the average number of the times where a rewards are larger than b rewards.
    """

    if isinstance(a_rewards, np.ndarray):
        a_rewards = torch.tensor(a_rewards)
    if isinstance(b_rewards, np.ndarray):
        b_rewards = torch.tensor(b_rewards)

    assert len(a_rewards.shape) == 1
    assert a_rewards.shape == b_rewards.shape

    a_rewards: Tensor = a_rewards.type(torch.float)
    b_rewards: Tensor = b_rewards.type(torch.float)

    return (a_rewards > b_rewards).type(torch.float).mean()


def compute_reward_metrics(eval_pred: transformers.trainer_utils.EvalPrediction):
    """Computes metrics during evaluation for RewardModel.

    Args:
        eval_pred: EvalPrediction that has `predictions`, `inputs`, `label_ids`.
    Returns:
        A dict of metrics[str, value].
    """

    predictions = eval_pred.predictions
    # RewardModel returns a_rewards (or chosen rewards) first and then b_rewards (rejected rewards).
    # This is hard-coded as 0 and 1 index.
    a_rewards = predictions[0]
    b_rewards = predictions[1]

    return {
        'rank_accuracy': reward_ranking_accuracy_metric(a_rewards, b_rewards)
    }
