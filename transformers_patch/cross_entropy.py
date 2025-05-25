import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * tl.num_programs(1) + block_idx
    labels_ptr += row_idx

    label = tl.load(labels_ptr)
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other = -float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    logsumexp = max_logits + tl.log(tl.sum(tl.exp(logits - max_logits), 0))
    if block_idx == 0:
        if label != -100:
            x = tl.load(logits_ptr + label).to(tl.float32)
            loss = -x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)


@triton.jit
def _cross_entropy_backward(
    logits_ptr,
    logits_row_stride,
    dloss_ptr,
    dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    dloss_ptr += row_idx * dloss_row_stride
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label = tl.load(labels_ptr)
    if label != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    x = tl.load(logits_ptr + col_offsets, mask=mask, other = -float("inf")).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr)
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label, y - 1.0, y)
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


class CrossEntropyLoss(torch.autograd.Function):
    @torch.profiler.record_function("ce_fwd")
    @staticmethod
    def forward(ctx, logits, labels):
        n, vocab_size = logits.shape
        device = logits.device
        BLOCK_SIZE = 65536
        grid = (n, triton.cdiv(vocab_size, BLOCK_SIZE))
        losses = torch.empty(n, dtype=torch.float32, device=device)
        logsumexp = torch.empty(grid, dtype=torch.float32, device=device)

        with torch.cuda.device(device):
            _cross_entropy_forward[grid](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)

        ctx.save_for_backward(logits, logsumexp, labels)
        return losses

    @torch.profiler.record_function("ce_bwd")
    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n, vocab_size = logits.shape
        BLOCK_SIZE = 16384
        grid = (n, triton.cdiv(vocab_size, BLOCK_SIZE))

        with torch.cuda.device(dlosses.device):
            _cross_entropy_backward[grid](
                logits, logits.stride(0),
                dlosses, dlosses.stride(0),
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return logits, None


if __name__ == "__main__":
    n = 16384
    vocab_size = 128*1024
    x = torch.randn(n, vocab_size, device="cuda", requires_grad=True)
    y = torch.randint(0, vocab_size, (n,), device="cuda")
    loss = torch.nn.functional.cross_entropy(x, y, reduction="none")
    loss.sum().backward()
    x_ = x.clone().detach_().requires_grad_()
    loss_ = CrossEntropyLoss.apply(x_, y)
    loss_.sum().backward()
    torch.testing.assert_close(loss, loss_)
    torch.testing.assert_close(x.grad, x_.grad)