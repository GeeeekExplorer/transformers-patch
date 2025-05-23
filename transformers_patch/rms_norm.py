import torch
import torch.nn.functional as F


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        dtype = x.dtype
        y = x.to(torch.float32)
        var = y.pow(2).mean(-1, keepdim=True)
        y = y * torch.rsqrt(var + eps)
        z = w * y.to(dtype)
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        ctx.dtype = dtype
        return z

    @staticmethod
    def backward(ctx, dz):
        x, w = ctx.saved_tensors
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            y = x.to(torch.float32)
            var = y.pow(2).mean(-1, keepdim=True)
            y = y * torch.rsqrt(var + ctx.eps)
            y = y.to(ctx.dtype)
        dy = dz * w
        torch.autograd.backward([y], [dy])
        dx = x.grad
        dw = dz * y
        return dx, dw, None


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, 4096, device="cuda", requires_grad=True)
    w = torch.randn(4096, device="cuda", requires_grad=True)
    y = F.rms_norm(x, (4096,), w, 1e-6)
    y.sum().backward()
    x_ = x.clone().detach_().requires_grad_()
    w_ = w.clone().detach_().requires_grad_()
    y_ = RMSNorm.apply(x_, w_, 1e-6)
    y_.sum().backward()
    assert torch.allclose(y, y_)
    assert torch.allclose(x.grad, x_.grad)
    assert torch.allclose(w.grad, w_.grad)