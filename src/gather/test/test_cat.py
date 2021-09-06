
from gather._C import gather_cat_forward, gather_cat_backward
from gather import gathercat
import torch
import time


def vis(msg: str, L: int = 40):
    if len(msg) >= L:
        print(msg)
    else:
        pad_l = (L-len(msg))//2
        pad_r = (L-len(msg)) - pad_l
        print("{} {} {}".format(pad_l*'=', msg, pad_r*'='))


def test(seed: int):
    vis(f'Test process with seed={seed}', 60)
    torch.manual_seed(seed)

    N = torch.randint(1, 20, (1,)).item()
    T = torch.randint(2, 512, (1,)).item()
    V = torch.randint(1, 1024, (1,)).item()
    lx = torch.randint(T//2, T, (N, ), dtype=torch.int, device=0)
    xs = torch.randn((N, lx.max(), V), dtype=torch.float, device=0)

    lx = lx.to(dtype=torch.int, device=0)
    print("xs size: ", xs.size())
    print("lx size: ", lx.size())

    xs.requires_grad = True

    def manual_cat(xs, lx):
        return torch.cat([xs[i, :lx[i]].view(-1, xs.size(-1)) for i in range(lx.size(0))], dim=0)

    def test_forward():
        vis('Test forward/backward computation')
        gather_x = gathercat(xs, lx)

        # manually cal
        manual = manual_cat(xs, lx)
        print(manual.size())

        if not torch.all(manual == gather_x):
            print("Forward mismatch")
            print(manual)
            print(gather_x)
            raise RuntimeError
        else:
            print("Forward correct.")

        weighted_w = torch.randn_like(gather_x)
        (gather_x*weighted_w).sum().backward()
        tx_grad = xs.grad.data.detach()
        xs.grad = None

        (manual*weighted_w).sum().backward()
        mx_grad = xs.grad.data.detach()
        xs.grad = None

        cmp = tx_grad == mx_grad
        if not torch.all(cmp):
            print("Backward mismatch.")
            print(torch.sum(torch.abs(tx_grad-mx_grad)))
            print(tx_grad[torch.logical_not(cmp)])
            print(mx_grad[torch.logical_not(cmp)])
            raise RuntimeError

        else:
            print("Backward correct.")

    def test_autogradcheck():
        vis('Test autograd with torch')
        try:
            torch.autograd.gradcheck(gathercat, (xs, lx))
        except Exception as e:
            print(e)
            print("Maybe limit the (N, T, V) to smaller number and re-test.")
            exit(1)

    test_forward()
    # test_autogradcheck()

    print('')


if __name__ == "__main__":

    for i in range(5):
        test(i)
