import torch
import torch.nn.functional as F


class SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        z = F.silu(x) * y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, dz):
        x, y = ctx.saved_tensors
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            t = F.silu(x)
        dt = dz * y
        torch.autograd.backward([t], [dt])
        dx = x.grad
        dy = dz * t
        return dx, dy


if __name__ == "__main__":
    x = torch.randn(1024, 2048, device="cuda", requires_grad=True)
    y = torch.randn(1024, 2048, device="cuda", requires_grad=True)
    z = F.silu(x) * y
    z.sum().backward()
    x_ = x.clone().detach_().requires_grad_()
    y_ = y.clone().detach_().requires_grad_()
    z_ = SiLUMul.apply(x_, y_)
    z_.sum().backward()
    assert torch.allclose(z, z_)
    assert torch.allclose(x.grad, x_.grad), (x_.grad / x.grad).mean()
    assert torch.allclose(y.grad, y_.grad)