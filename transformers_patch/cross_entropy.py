import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_forward_kernel(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    label = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + row_idx * logits_row_stride + col_offsets, mask=mask, other = -float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    logsumexp = max_logits + tl.log(tl.sum(tl.exp(logits - max_logits), 0))
    if label != -100:
        x = tl.load(logits_ptr + row_idx * logits_row_stride + label).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(loss_ptr + row_idx, loss)
    tl.store(logsumexp_ptr + row_idx, logsumexp)


@triton.jit
def cross_entropy_backward_kernel(
    logits_ptr,
    logits_row_stride,
    dloss_ptr,
    dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    label = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + row_idx * logits_row_stride + col_offsets, mask=mask, other = -float("inf")).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    dlogits = tl.exp(logits - logsumexp)
    dlogits = tl.where(col_offsets == label, dlogits - 1.0, dlogits)
    if label != -100:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    tl.store(logits_ptr + row_idx * logits_row_stride + col_offsets, dloss * dlogits, mask=mask)


class CrossEntropyLoss(torch.autograd.Function):
    @torch.profiler.record_function("ce_fwd")
    @staticmethod
    def forward(ctx, logits, labels):
        n_rows, n_cols = logits.shape
        device = logits.device
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        losses = torch.empty(n_rows, dtype=torch.float32, device=device)
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device=device)

        with torch.cuda.device(device):
            cross_entropy_forward_kernel[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=1,
                num_warps=32,
            )

        ctx.save_for_backward(logits, logsumexp, labels)
        return losses

    @torch.profiler.record_function("ce_bwd")
    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        with torch.cuda.device(dlosses.device):
            cross_entropy_backward_kernel[(n_rows,)](
                logits, logits.stride(0),
                dlosses, dlosses.stride(0),
                logsumexp,
                labels,
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=1,
                num_warps=32,
            )

        return logits, None


if __name__ == "__main__":
    n = 16384
    vocab_size = 151936
    x = torch.randn(n, vocab_size, device="cuda", requires_grad=True)
    y = torch.randint(0, vocab_size, (n,), device="cuda")
    coff = torch.randn(n, device="cuda")
    loss = torch.nn.functional.cross_entropy(x, y, reduction="none")
    (coff * loss).sum().backward()
    x_ = x.clone().detach_().requires_grad_()
    loss_ = CrossEntropyLoss.apply(x_, y)
    (coff * loss_).sum().backward()
    torch.testing.assert_close(loss, loss_)
    torch.testing.assert_close(x.grad, x_.grad)