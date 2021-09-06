
from gather._C import gather_sum_forward, gather_sum_backward
from gather import gathersum
import torch
import time

"""
xs [1,2,3,1,1,2]
lx [3,1,2]

ys [1,1,2,1]
ly [1,2,1]


[1+1, 2+1, 3+1, 1+1, 1+2, 1+1, 2+1]
[1., 1., 1., 1. + 1., 1., 1.]
[1. + 1. + 1., 1., 1., 1.+ 1.] 
"""


def vis(msg: str, L: int = 40):
    if len(msg) >= L:
        print(msg)
    else:
        pad_l = (L-len(msg))//2
        pad_r = (L-len(msg)) - pad_l
        print("{} {} {}".format(pad_l*'=', msg, pad_r*'='))


class Anything(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.new_ones((5, ))

    @staticmethod
    def backward(ctx, grad_outputs):

        print("Is gradient w.r.t. sum of b contiguous? ",
              grad_outputs.is_contiguous())
        return grad_outputs.new_zeros((1,))


def test(seed: int):
    vis(f'Test process with seed={seed}', 60)
    torch.manual_seed(seed)

    N = torch.randint(1, 4, (1,)).item()
    T = torch.randint(2, 4, (1,)).item()
    U = torch.randint(2, 4, (1,)).item()
    V = torch.randint(1, 1024, (1,)).item()
    lx = torch.randint(T//2, T, (N, ), dtype=torch.int, device=0)
    xs = torch.randn((lx.sum(), V), dtype=torch.float, device=0)
    ly = torch.randint(U//2, U, (N, ), dtype=torch.int, device=0)
    ys = torch.randn((ly.sum(), V), dtype=torch.float, device=0)

    # xs = torch.tensor([[1], [2], [3], [1], [1], [2]],
    #                   dtype=torch.float, device=0)
    # ys = torch.tensor([[1], [1], [2], [1]], dtype=torch.float, device=0)
    # lx = torch.tensor([3, 1, 2], dtype=torch.int, device=0)
    # ly = torch.tensor([1, 2, 1], dtype=torch.int, device=0)

    # xs = torch.randn((13, 5), device=0)
    # ys = torch.randn((4, 5), device=0)
    # lx = torch.tensor([3, 4, 3, 3], dtype=torch.int, device=0)
    # ly = torch.tensor([1, 1, 1, 1], dtype=torch.int, device=0)

    # vis('Test loaded sample')
    # checkpoint = torch.load('CheckForTestGatherSum.pt', map_location='cuda:0')
    # xs, ys, lx, ly = tuple(checkpoint.values())

    lx, ly = lx.to(dtype=torch.int, device=0), ly.to(dtype=torch.int, device=0)
    print("xs size: ", xs.size())
    print("ys size: ", ys.size())
    print("lx size: ", lx.size())
    print("ly size: ", ly.size())

    xs.requires_grad = True
    ys.requires_grad = True

    def manual_sum(xs, ys, lx, ly):
        out = []
        lx_cumsun = lx.cumsum(0)
        ly_cumsun = ly.cumsum(0)

        for n in range(lx.size(0)):
            Ti = xs[lx_cumsun[n]-lx[n]:lx_cumsun[n], :]
            Ui = ys[ly_cumsun[n]-ly[n]:ly_cumsun[n], :]
            out.append(Ti[:, None, :] + Ui[None, :, :])

        return torch.cat([x.view(-1, xs.size(-1)) for x in out], dim=0)

    def test_forward():
        vis('Test forward/backward computation')
        gather_x = gathersum(xs, ys, lx, ly)

        # manually cal
        manual = manual_sum(xs, ys, lx, ly)

        if not torch.all(manual == gather_x):
            print(manual)
            print(gather_x)
        else:
            print("Forward correct.")

        weighted_w = torch.randn_like(gather_x)
        (gather_x*weighted_w).sum().backward()
        tx_grad = xs.grad.data.detach()
        ty_grad = ys.grad.data.detach()
        xs.grad = None
        ys.grad = None

        (manual*weighted_w).sum().backward()
        mx_grad = xs.grad.data.detach()
        my_grad = ys.grad.data.detach()
        xs.grad = None
        ys.grad = None

        cmp = torch.all(tx_grad == mx_grad) and torch.all(ty_grad == my_grad)
        if not cmp:
            if not torch.all(tx_grad == mx_grad):
                print("xs backward mismatch.")
                print(torch.sum(torch.abs(tx_grad-mx_grad)))
                print(tx_grad[torch.logical_not(cmp)])
                print(mx_grad[torch.logical_not(cmp)])
            else:
                print("ys backward mismatch.")
                print(torch.sum(torch.abs(ty_grad-my_grad)))
                print(ty_grad[torch.logical_not(cmp)])
                print(my_grad[torch.logical_not(cmp)])

        else:
            print("Backward correct.")

    def test_backward_speed():
        vis('Test backward speed')
        gathered_sum = gather_sum_forward(xs, ys, lx, ly)

        cnt = 20
        t_beg = time.time()
        for _ in range(cnt):
            gather_sum_backward(torch.ones_like(gathered_sum), lx, ly)

        t = time.time() - t_beg
        print("Run {} times of backward function, takes {:.2f} ms on average.".format(
            cnt, t/cnt*1000))

    def test_autogradcheck():
        vis('Test autograd with torch')
        try:
            torch.autograd.gradcheck(gathersum, (xs, ys, lx, ly))
        except Exception as e:
            print(e)
            print("Maybe limit the (N, T, U, V) to smaller number and re-test.")
            exit(1)

    # test_autogradcheck()
    test_forward()

    print('')


if __name__ == "__main__":
    # funct = Anything.apply
    # a = torch.zeros((1,), requires_grad=True)
    # b = funct(a)
    # bs = b.sum()
    # print("Is sum of b contiguous? ", bs.is_contiguous())
    # bs.backward()
    # print(a.grad)
    # exit(0)

    # test(2)
    # exit(0)
    for i in range(1):
        test(i)
