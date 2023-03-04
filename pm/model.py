import logging
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from pm.data import get_tokenizer
from pm.loss import batch_pairwise_loss, binary_crossentropy_ranking_loss
from pm.utils import HParams
from pm.utils import maybe_freeze_layers

# Pretrained model name -> Path to model weights (online or offline loading).
PRETRAINED_MODEL_TYPES = {
    # TODO: Add GPT2-X*
    'gpt2': 'gpt2',
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
    'CarperAI/openai_summarize_tldr_sft': 'CarperAI/openai_summarize_tldr_sft'
}

REWARD_MODEL_TYPES = {
    'pool': 'pool',
    'per_token': 'per_token'
}


class BaseRewardModel(nn.Module):

    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        assert hparams.pretrained_model in PRETRAINED_MODEL_TYPES
        model = self._load_pretrained_model()

        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = get_tokenizer(hparams.tokenizer_type)

    def _load_pretrained_model(self):
        """Loads pretrained model based on hparams."""
        assert self.hparams.pretrained_model in PRETRAINED_MODEL_TYPES
        model = AutoModelForCausalLM.from_pretrained(self.hparams.pretrained_model)
        self.config = model.config

        if self.hparams.pretrained_model == 'CarperAI/openai_summarize_tldr_sft':
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = (
                self.config.hidden_size
                if hasattr(self.config, "hidden_size")
                else self.config.n_embd
            )

        return model

    def forward(self, input_ids, mask=None, divergence_index=None, labels=None):
        raise NotImplementedError()


class PerTokenRewardModel(BaseRewardModel):
    """This reward model calculates rewards for each token. The rewards for the last token is taken when inference."""

    def forward(self, input_ids, mask=None, divergence_index=None, labels=None):
        if labels is None:
            # Inference mode. Run only on 'chosen'.
            hidden_states = self.transformer(input_ids['chosen'], attention_mask=mask['chosen']).last_hidden_state
            rewards = self.v_head(hidden_states).squeeze(-1)
            return rewards

        # Concat 'chosen' and 'rejected' for batch computation of both.
        chosen_input_ids, rejected_input_ids = input_ids['chosen'], input_ids['rejected']
        c_num = chosen_input_ids.shape[0]
        r_num = rejected_input_ids.shape[0]
        assert c_num == r_num, "We assume number of chosen samples == number of rejected samples."
        # input_ids[chosen], input_id[rejected]: [batch_size, seq_len].
        # all_input_ids: [batch_size * 2, seq_len].
        all_input_ids = torch.concat((chosen_input_ids, rejected_input_ids), dim=0)
        all_mask = torch.concat((mask['chosen'], mask['rejected']), dim=0)
        # hidden_states: [batch_size * 2, seq_len, h_dim].
        hidden_states = self.transformer(all_input_ids, attention_mask=all_mask).last_hidden_state
        # rewards: [batch_size * 2, seq_len].
        rewards = self.v_head(hidden_states).squeeze(-1)
        c_rewards = rewards[:c_num]
        r_rewards = rewards[c_num:]

        # Calculate loss.
        loss_dict = batch_pairwise_loss(c_rewards=c_rewards, r_rewards=r_rewards,
                                        c_mask=mask['chosen'], r_mask=mask['rejected'],
                                        divergence_index=divergence_index)
        return loss_dict


class PoolRewardModel(BaseRewardModel):
    """This reward model pools the rewards into the last token (a special token) of each sequence."""

    def __init__(self, hparams: HParams):
        super(PoolRewardModel, self).__init__(hparams)
        self.pool_token_id = self.tokenizer.eos_token_id

    def forward(self, input_ids, mask=None, divergence_index=None, labels=None):
        if labels is None:
            # Inference mode. Run only on 'chosen'.
            this_input_ids = input_ids['chosen']
            this_mask = mask['chosen']
            # Replace the last input_ids to be a special token to pool rewards.
            this_input_ids[:, -1] = self.pool_token_id
            # Enable attention on this token.
            this_mask[:, -1] = 1

            hidden_states = self.transformer(this_input_ids, attention_mask=this_mask).last_hidden_state
            rewards = self.v_head(hidden_states[:, -1, :]).squeeze(-1)
            return rewards

        # Concat 'chosen' and 'rejected' for batch computation of both.
        chosen_input_ids, rejected_input_ids = input_ids['chosen'], input_ids['rejected']
        c_num = chosen_input_ids.shape[0]
        r_num = rejected_input_ids.shape[0]
        assert c_num == r_num, "We assume number of chosen samples == number of rejected samples."
        # input_ids[chosen], input_id[rejected]: [batch_size, seq_len].
        # all_input_ids: [batch_size * 2, seq_len].
        all_input_ids = torch.concat((chosen_input_ids, rejected_input_ids), dim=0)
        all_mask = torch.concat((mask['chosen'], mask['rejected']), dim=0)

        # Replace the last input_ids to be a special token to pool rewards.
        all_input_ids[:, -1] = self.pool_token_id
        # Enable attention on this token.
        all_mask[:, -1] = 1

        # hidden_states: [batch_size * 2, seq_len, h_dim].
        hidden_states = self.transformer(all_input_ids, attention_mask=all_mask).last_hidden_state
        # rewards: [batch_size * 2, 1].
        rewards = self.v_head(hidden_states[:, -1, :]).squeeze(-1)
        c_rewards = rewards[:c_num]
        r_rewards = rewards[c_num:]

        # Calculate loss.
        # We know that ranking labels are all 0 as chosen rewards are the one that should get higher rewards.
        ranking_labels = torch.zeros_like(c_rewards).to(chosen_input_ids.device)
        loss = binary_crossentropy_ranking_loss(c_rewards, r_rewards, ranking_labels)

        return {'loss': loss,
                'c_rewards': c_rewards,
                'r_rewards': r_rewards}


    
def get_reward_model(hparams: HParams):
    """Gets reward model."""
    reward_model_type = hparams.reward_model_type
    assert reward_model_type in REWARD_MODEL_TYPES

    model = None
    if reward_model_type == 'pool':
        model = PoolRewardModel(hparams)
    elif reward_model_type == 'per_token':
        model = PerTokenRewardModel(hparams)
    else:
        raise ValueError(reward_model_type, f'Unexpected reward_model_type: {reward_model_type}')
    

    model = maybe_freeze_layers(model, hparams.freeze_backbone_layers_ratio)

    return model
    
    

