import torch
from torch import nn
from transformers import AutoModelForCausalLM

from pm.data import get_tokenizer
from pm.utils import batch_get_mask_equal_or_larger_than_indices
from pm.utils import HParams

# Model name -> Path to model weights (online or offline loading).
MODEL_TYPES = {
    # TODO: Add GPT2-X*
    'gpt2': 'gpt2',
    'CarperAI/openai_summarize_tldr_sft': 'CarperAI/openai_summarize_tldr_sft'
}


class GPTRewardModel(nn.Module):

    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        assert hparams.pretrained_model in MODEL_TYPES
        model = self._load_pretrained_model()

        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = get_tokenizer(hparams.tokenizer_type)

    def _load_pretrained_model(self):
        """Loads pretrained model based on hparams."""
        assert self.hparams.pretrained_model in MODEL_TYPES
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

    def forward(self, input_ids, mask=None, inference=False, divergence_index=None):
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
        c_lengths = mask['chosen'].sum(axis=1)
        r_lengths = mask['rejected'].sum(axis=1)
        or_mask = torch.logical_or(mask['chosen'], mask['rejected']).type(torch.long)
        or_lengths = or_mask.sum(axis=1)

        d_rewards = (c_rewards - r_rewards) * or_mask
        r_last_reward = torch.gather(r_rewards, or_lengths, dim=1)
        c_last_reward = torch.gather(c_rewards, or_lengths, dim=1)
        divergence_mask = batch_get_mask_equal_or_larger_than_indices(
            d_rewards, divergence_index
        )
        d_rewards = d_rewards * divergence_mask
        weights = divergence_mask * or_mask

        loss = -torch.log(torch.sigmoid(d_rewards))

        loss = loss / weights.sum()

        return {
            "loss": loss,
        }
