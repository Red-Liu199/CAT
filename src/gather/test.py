
import torch
import sys
sys.path.pop(0)
# should be here
from gather_sum import gathersum
from gather_sum._C import gather_sum_forward, gather_sum_backward

"""
xs [1,2,3,1,1,2]
lx [3,1,2]

ys [1,1,2,1]
ly [1,2,1]


[1+1, 2+1, 3+1, 1+1, 1+2, 1+1, 2+1]
[1., 1., 1., 1. + 1., 1., 1.]
[1. + 1. + 1., 1., 1., 1.+ 1.] 
"""

if __name__ == "__main__":

    N = 15
    V = 7
    xn = torch.randint(2, 5, (N, ), dtype=torch.int, device=0)
    enc = torch.randn((xn.sum(), V), dtype=torch.float, device=0)
    # yn = torch.randint(2, 5, (N, ), dtype=torch.int, device=0)
    yn = torch.ones((N,), dtype=torch.int, device=0)
    dec = torch.randn((yn.sum(), V), dtype=torch.float, device=0)

    # enc = torch.tensor([[1], [2], [3], [1], [1], [2]],
    #                    dtype=torch.float, device=0)
    # dec = torch.tensor([[1], [1], [2], [1]], dtype=torch.float, device=0)
    # xn = torch.tensor([3, 1, 2], dtype=torch.int, device=0)
    # yn = torch.tensor([1, 2, 1], dtype=torch.int, device=0)

    enc = torch.randn((13, 5), device=0)
    dec = torch.randn((4, 5), device=0)
    xn = torch.tensor([3, 4, 3, 3], dtype=torch.int, device=0)
    yn = torch.tensor([1, 1, 1, 1], dtype=torch.int, device=0)

    print(enc.size(), dec.size())
    print(xn)
    print(yn)
    print("{0} Test forward computation {0}".format("="*10))
    gather_x = gather_sum_forward(enc, dec, xn, yn)

    # manually cal
    out = []
    xn_cumsun = xn.cumsum(0)
    yn_cumsun = yn.cumsum(0)

    for n in range(xn.size(0)):
        Ti = enc[xn_cumsun[n]-xn[n]:xn_cumsun[n], :]
        Ui = dec[yn_cumsun[n]-yn[n]:yn_cumsun[n], :]
        out.append(Ti[:, None, :] + Ui[None, :, :])

    manual = torch.cat([x.view(-1, enc.size(-1)) for x in out], dim=0)

    if not torch.all(manual == gather_x):
        print(manual)
        print(gather_x)
    else:
        print("Forward correct.")

    print("{0} Test forward/backward {0}".format("="*10))
    enc.requires_grad = True
    dec.requires_grad = True
    sum_ = gathersum(enc, dec, xn, yn)

    (sum_.mean()).backward()
