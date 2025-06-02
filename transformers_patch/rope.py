import torch


def rotate_half(x, fwd=True):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    if fwd:
        return torch.cat((-x2, x1), dim=-1)
    else:
        return torch.cat((x2, -x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, fwd=True):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q, fwd) * sin)
    k_embed = (k * cos) + (rotate_half(k, fwd) * sin)
    return q_embed, k_embed


class RotaryEmbedding(torch.autograd.Function):
    @torch.profiler.record_function("rope_fwd")
    @torch.compile
    @staticmethod
    def forward(ctx, q, k, cos, sin, unsqueeze_dim=1):
        ctx.save_for_backward(cos, sin)
        ctx.unsqueeze_dim = unsqueeze_dim
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
        return q, k

    @torch.profiler.record_function("rope_bwd")
    @torch.compile
    @staticmethod
    def backward(ctx, dq, qk):
        cos, sin = ctx.saved_tensors
        dq, dk = apply_rotary_pos_emb(dq, qk, cos, sin, ctx.unsqueeze_dim, False)
        return dq, dk, None, None, None


if __name__ == "__main__":
    bs = 2
    seq_len = 1024
    num_head = 32
    dim = 128
    q = torch.randn(bs, num_head, seq_len, dim, requires_grad=True, device="cuda")
    k = torch.randn(bs, num_head, seq_len, dim, requires_grad=True, device="cuda")
    freqs = torch.randn(bs, seq_len, dim // 2, device="cuda")
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    t = apply_rotary_pos_emb(q, k, cos, sin)
    (t[0] + t[1]).sum().backward()
    q_ = q.clone().detach_().requires_grad_()
    k_ = k.clone().detach_().requires_grad_()
    t_ = RotaryEmbedding.apply(q_, k_, cos, sin)
    (t_[0] + t_[1]).sum().backward()
    torch.testing.assert_close(t[0], t_[0])
    torch.testing.assert_close(t[1], t_[1])
    torch.testing.assert_close(q.grad, q_.grad)
    torch.testing.assert_close(k.grad, k_.grad)