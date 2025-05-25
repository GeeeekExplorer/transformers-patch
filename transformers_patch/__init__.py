import sys
import inspect
import torch
import torch.nn.functional as F
import transformers
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


try:
    old_func = transformers.modeling_flash_attention_utils._flash_attention_forward
    source = inspect.getsource(old_func)
    source = source.replace("(torch.diff(position_ids, dim=-1) >= 0).all()", "check_once(position_ids)")
    source += """
def check_once(position_ids):
    if not hasattr(check_once, "cache"):
        check_once.cache = (-1, False)
    k = id(position_ids)
    if k == check_once.cache[0]:
        return check_once.cache[1]
    v = (torch.diff(position_ids, dim=-1) >= 0).all().item()
    check_once.cache = (k, v)
    return v
"""
    exec(source, transformers.modeling_flash_attention_utils.__dict__)
    new_func = transformers.modeling_flash_attention_utils._flash_attention_forward
    for module in sys.modules.values():
        if getattr(module, "_flash_attention_forward", None) is old_func:
            setattr(module, "_flash_attention_forward", new_func)
except:
    pass