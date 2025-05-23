import torch
import torch.nn.functional as F


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        y = x * rstd
        z = w * y.to(dtype)
        ctx.save_for_backward(z, w, rstd)
        return z

    @staticmethod
    def backward(ctx, dz):
        z, w, rstd = ctx.saved_tensors
        w = w.float()
        y = z / w
        dy = dz * w
        dx = rstd * (dy - y * (y * dy).mean(-1, keepdim=True))
        dw = (dz * y).view(-1, w.size(-1)).sum(0)
        return dx, dw, None


if __name__ == "__main__":
    x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.ones(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = F.rms_norm(x, (4096,), w, 1e-6)
    y.sum().backward()
    x_ = x.clone().detach_().requires_grad_()
    w_ = w.clone().detach_().requires_grad_()
    y_ = RMSNorm.apply(x_, w_, 1e-6)
    y_.sum().backward()
    torch.testing.assert_close(y, y_)
    torch.testing.assert_close(x.grad, x_.grad)
    torch.testing.assert_close(w.grad, w_.grad)