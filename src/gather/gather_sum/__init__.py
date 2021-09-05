import torch
import gather_sum._C as core
from pkg_resources import get_distribution

__version__ = get_distribution('gather_sum').version


class _GatherSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xs, ys, lx, ly):
        gathered_sum = core.gather_sum_forward(xs, ys, lx, ly)
        ctx.save_for_backward(lx, ly)
        return gathered_sum

    @staticmethod
    def backward(ctx, grad_sum):

        lx, ly = ctx.saved_tensors
        grad_x, grad_y = core.gather_sum_backward(
            grad_sum.contiguous(), lx, ly)
        return grad_x, grad_y, None, None


def gathersum(xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
    """ Sum the two 'gathered' tensors xs and ys.

    Args:
        xs (torch.FloatTensor): of size (lx0+lx1+..., *)
        ys (torch.FloatTensor): of size (ly0+ly1+..., *)
        lx (torch.LongTensor): of size (N, )
        ly (torch.LongTensor): of size (N, )

    Return:
        gathered_sum (torch.FloatTensor): size (lx0ly0+lx1ly1+..., *)
    """
    return _GatherSum.apply(xs, ys, lx.to(device=xs.device, dtype=torch.int32), ly.to(device=xs.device, dtype=torch.int32))
