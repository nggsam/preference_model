import torch

from pm.utils import batch_get_mask_equal_or_larger_than_indices


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
