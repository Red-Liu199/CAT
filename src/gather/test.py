import torch
from gather_sum._C import gather_sum


if __name__ == "__main__":
    N = 3
    V = 3

    xn = torch.randint(2, 5, (N, ), dtype=torch.int, device=0)
    enc = torch.randn((xn.sum(), V), dtype=torch.float, device=0)

    yn = torch.randint(2, 5, (N, ), dtype=torch.int, device=0)
    dec = torch.randn((yn.sum(), V), dtype=torch.float, device=0)

    print(enc.size(), dec.size())
    print(xn)
    print(yn)
    gather_x = gather_sum(enc, dec, xn, yn)

    # manually cal
    out = []
    xn_cumsun = xn.cumsum(0)
    yn_cumsun = yn.cumsum(0)

    for n in range(xn.size(0)):
        Ti = enc[xn_cumsun[n]-xn[n]:xn_cumsun[n], :]
        Ui = dec[yn_cumsun[n]-yn[n]:yn_cumsun[n], :]
        out.append(Ti[:, None, :] + Ui[None, :, :])

    manual = torch.cat([x.view(-1, V) for x in out], dim=0)

    print(torch.all(manual == gather_x))
