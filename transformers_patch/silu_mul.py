import torch
import torch.nn.functional as F


class SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        t = F.silu(x)
        z = t * y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, dz):
        x, y = ctx.saved_tensors
        t = F.silu(x)
        dt = dz * y
        s = torch.sigmoid(x)
        dx = dt * (s * (1 + x * (1 - s)))
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
    torch.testing.assert_close(z, z_)
    torch.testing.assert_close(x.grad, x_.grad)
    torch.testing.assert_close(y.grad, y_.grad)