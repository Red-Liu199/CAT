import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def example(rank, world_size, use_zero):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # create local model
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
    print_peak_memory("Max memory allocated after creating local model", rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    print_peak_memory("Max memory allocated after creating DDP", rank)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optim=torch.optim.Adam,
            lr=0.01
        )
    else:
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

    for _ in range(3):
        # forward pass
        outputs = ddp_model(torch.randn(20, 2000).to(rank))
        labels = torch.randn(20, 2000).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()

        # update parameters
        print_peak_memory("Max memory allocated before optimizer step()", rank)
        optimizer.step()
        optimizer.zero_grad()
        print_peak_memory("Max memory allocated after optimizer step()", rank)
        if rank == 0:
            print(torch.cuda.memory_summary())
        if use_zero:
            optimizer.consolidate_state_dict(0)
        if rank == 0:
            torch.save(optimizer.state_dict(), "tmp.pt")
        dist.barrier()
        optimizer.load_state_dict(torch.load("tmp.pt"))

        print(f"params sum is: {sum(model.parameters()).sum()}")
        if rank == 0:
            print(torch.cuda.memory_summary())
        exit(1)


def main():
    world_size = 2
    print("=== Using ZeroRedundancyOptimizer ===")
    mp.spawn(example,
        args=(world_size, True),
        nprocs=world_size,
        join=True)

    print("=== Not Using ZeroRedundancyOptimizer ===")
    mp.spawn(example,
        args=(world_size, False),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    main()
