import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx
    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other = -float("inf")).to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            loss = -1.0 * x
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
    dloss_ptr  += row_idx *  dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    x = tl.load(logits_ptr + col_offsets, mask=mask, other = -float("inf")).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


MAX_FUSED_SIZE = 65536
class CrossEntropyLoss(torch.autograd.Function):
    @torch.profiler.record_function("ce_fwd")
    @staticmethod
    def forward(ctx, logits, labels):
        n_rows, vocab_size = logits.shape
        device = logits.device
        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device=device)
        logsumexp = torch.empty((n_rows, n_chunks,), dtype=torch.float32, device=device)

        with torch.cuda.device(device):
            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                num_warps=32,
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
        n_rows, vocab_size = logits.shape
        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        with torch.cuda.device(dlosses.device):
            _cross_entropy_backward[(n_rows, n_blocks,)](
                logits, logits.stride(0),
                dlosses, dlosses.stride(0),
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
            )
        return logits, None, None, None,


if __name__ == "__main__":
    x = torch.randn(1024, 128*1024, device="cuda", requires_grad=True)
    y = torch.randint(0, 128*1024, (1024,), device="cuda")
    loss = torch.nn.functional.cross_entropy(x, y, reduction="none")
    loss.sum().backward()
    x_ = x.clone().detach_().requires_grad_()
    loss_ = CrossEntropyLoss.apply(x_, y)
    loss_.sum().backward()
    torch.testing.assert_close(loss, loss_)
    torch.testing.assert_close(x.grad, x_.grad)