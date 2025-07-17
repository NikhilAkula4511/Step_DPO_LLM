from trl import DPOTrainer
from transformers import PreTrainedModel
from typing import Dict, Union
import torch
import torch.nn.functional as F


class StepwiseDPOTrainer(DPOTrainer):
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # Ensure all tensors are on the correct device
        device = model.device
        chosen_input_ids = inputs["chosen_input_ids"].to(device)
        rejected_input_ids = inputs["rejected_input_ids"].to(device)

        # Compute attention masks if needed
        attention_mask_chosen = (chosen_input_ids != self.tokenizer.pad_token_id).to(device)
        attention_mask_rejected = (rejected_input_ids != self.tokenizer.pad_token_id).to(device)

        # Forward pass
        chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=attention_mask_chosen)
        rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=attention_mask_rejected)

        # Extract logits
        chosen_logits = chosen_outputs.logits  # [B, T, V]
        rejected_logits = rejected_outputs.logits

        # Log-probs
        chosen_logprobs = self._get_batch_log_probs(chosen_logits, chosen_input_ids)
        rejected_logprobs = self._get_batch_log_probs(rejected_logits, rejected_input_ids)

        # Sequence-level rewards
        chosen_rewards = chosen_logprobs.mean(dim=1)  # [B]
        rejected_rewards = rejected_logprobs.mean(dim=1)

        logits_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits_diff).mean()

        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
                "logits_diff": logits_diff,
            }
        else:
            return loss

    def _get_batch_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probabilities of the labels under the logits.
        Args:
            logits: Tensor [B, T, V]
            labels: Tensor [B, T]
        Returns:
            log_probs: Tensor [B, T]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        label_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        return label_log_probs
