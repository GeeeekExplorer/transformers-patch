import torch
import torch.nn.functional as F
try:
    from transformers.models.qwen2 import modeling_qwen2
except:
    modeling_qwen2 = None
try:
    from transformers.models.qwen3 import modeling_qwen3
except:
    modeling_qwen3 = None
try:
    from transformers.loss.loss_utils import LOSS_MAPPING
except:
    LOSS_MAPPING = {}

from .rms_norm import RMSNorm
from .silu_mul import SiLUMul
from .rope import RotaryEmbedding
from .cross_entropy import CrossEntropyLoss


def rms_norm_forward(self, hidden_states):
    return RMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)


def mlp_forward(self, x):
    return self.down_proj(SiLUMul.apply(self.gate_proj(x), self.up_proj(x)))


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    return RotaryEmbedding.apply(q, k, cos, sin, unsqueeze_dim)


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size,
    num_items_in_batch = None,
    ignore_index = -100,
    shift_labels = None,
    **kwargs,
) -> torch.Tensor:
    assert ignore_index == -100
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    loss = CrossEntropyLoss.apply(logits, shift_labels)
    return loss.sum() / num_items_in_batch if num_items_in_batch is not None else loss.mean()


if modeling_qwen2 is not None:
    modeling_qwen2.Qwen2RMSNorm.forward = rms_norm_forward
    modeling_qwen2.Qwen2MLP.forward = mlp_forward
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
if modeling_qwen3 is not None:
    modeling_qwen3.Qwen3RMSNorm.forward = rms_norm_forward
    modeling_qwen3.Qwen3MLP.forward = mlp_forward
    modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb
LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss