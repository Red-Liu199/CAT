# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Optimizer scheduler impl"""

import math
import torch
import numpy as np
from collections import OrderedDict
from typing import *

from torch.distributed.optim import ZeroRedundancyOptimizer


def SetupOptim(type_optim: str, paramlist: Iterable[torch.nn.parameter.Parameter], use_zero: bool = False, **kwargs) -> Union[torch.optim.Optimizer, ZeroRedundancyOptimizer]:
    """Setup the optimizer.

    Args:
        type_optim (str): name of optimizer, should be an attribute of `torch.optim`, like `Adam`, `SGD`.
        paramlist (Iterable[Parameter]): a iterator or generator that returns parameters to be optimized.
        use_zero (bool, default False): a flag to determinte whether use `ZeroRedundancyOptimizer` or not,
            ref to https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html, which is supported since
            torch 1.8.0
        **kwargs: any keyword arguments can be passed into optimizer initializatio.
    
    Return:
        optimizer (torch.optim.Optimizer | ZeroRedundancyOptimizer)

    Example:
        >>> # With `use_zero=False`
        >>> model = nn.Linear(3,4)
        >>> optimizer = SetupOptim('Adam', model.parameters(), lr=1e-3, betas=(0.9,0.99))
        >>> # With `use_zero=True`
        >>> # ... (init of DDP)
        >>> model = torch.nn.parallel.DistributedDataParallel(model)
        >>> optimizer = SetupOptim('Adam', model.parameters(), use_zero=True, lr=1e-3, betas=(0.9,0.99))
    """
    if not use_zero:
        return getattr(torch.optim, type_optim)(paramlist, **kwargs)
    else:
        # raise NotImplementedError(f"Still on testing.")
        if torch.__version__ < '1.8.0':
            raise NotImplementedError

        # NOTE: This is still a experimental function in torch 1.9.0
        if torch.__version__ < '1.9.0':
            zerooptimizer = ZeroRedundancyOptimizer(
                params=paramlist, optim=getattr(torch.optim, type_optim), **kwargs)
        else:
            zerooptimizer = ZeroRedundancyOptimizer(
                params=paramlist, optimizer_class=getattr(torch.optim, type_optim), **kwargs)
        return zerooptimizer


class Scheduler(object):
    def __init__(self, optimizer_configs: dict, paramlist: Iterable[torch.nn.parameter.Parameter], reverse_metric_direc: bool = True):
        super().__init__()
        if 'type_optim' in optimizer_configs:   # for compablility of previous versions
            optimizer_configs['type'] = optimizer_configs['type_optim']

        self.optimizer = SetupOptim(
            optimizer_configs['type'], paramlist, ('use_zero' in optimizer_configs) and optimizer_configs['use_zero'], **optimizer_configs['kwargs'])
        self.iter_cur = 0
        self.best_metric = None
        self._reverse_ = reverse_metric_direc
        self.lr_init = self.lr_cur

    @property
    def lr_cur(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _adjust_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        output = OrderedDict()
        for name, attr in vars(self).items():
            if name == 'optimizer':
                output['optimizer'] = attr.state_dict()
            else:
                output[name] = attr
        return output

    def load_state_dict(self, ckpt: OrderedDict, optim_only: bool = False):
        """Load state dict from checkpoint.

        By setting `optim_only`, it allows to update the configurations of scheduler
        """
        if optim_only:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            return None

        for name in vars(self).keys():
            if name not in ckpt:
                continue
            if name == "optimizer":
                self.optimizer.load_state_dict(ckpt[name])
            else:
                setattr(self, name, ckpt[name])

    def update_lr_step(self, global_steps: int):
        """Method for updating the LR by steps. Defaultly do nothing."""
        return None

    def impl_step(self, metric) -> Literal[0, 1, 2]:
        """Implementation of scheduler updating according to metric from evaluation.
        This function will be invoked by `scheduler.step()`
        """
        raise NotImplementedError

    def step(self, cnt_iter: int, metric) -> Tuple[Literal[0, 1, 2], str]:
        """Optimizer step.
        Update the scheduler by every evaluation, useful for early-stop.

        Args:
            cnt_iter (int): # of iterations (begins from 1)
            metric (obj): the metric for evaluate the performance

        Returns: (state, info)
            state (int): choice of `[0, 1, 2]`, meaning
                0: continue training by the prior condition
                1: continue training for metric is improving
                2: stop training.
            
            info (str): information
        """
        if self.best_metric is None:
            self.best_metric = metric

        self.iter_cur = cnt_iter
        return self.impl_step(metric)


class SchedulerEarlyStop(Scheduler):
    """A scheduler wrapper for early-stop ones."""

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            iter_min: int,
            lr_stop: float = 1e-5,
            num_ahead: int = 1,
            gamma: float = 0.1,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, reverse_metric_direc)
        self.lr_stop = lr_stop
        self.iter_min = iter_min
        self.num_ahead = num_ahead
        self.gamma = gamma
        self.cnt_worse = 0

    def impl_step(self, metric) -> Literal[0, 1, 2]:
        state = 0
        if self.iter_cur <= self.iter_min:
            if not (self._reverse_ ^ (metric < self.best_metric)):
                self.best_metric = metric
                state = 1
        elif not (self._reverse_ ^ (metric < self.best_metric)):
            self.best_metric = metric
            self.cnt_worse = 0
            state = 1
        else:
            self.cnt_worse += 1
            if self.cnt_worse >= self.num_ahead:
                lr = self.lr_cur
                lr *= self.gamma
                if lr < self.lr_stop:
                    state = 2
                else:
                    self._adjust_lr(lr)
                    self.cnt_worse = 0
        return state


class SchedulerFixedStop(Scheduler):
    """A scheduler wrapper for ones stopping at fixed iterations."""

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            iter_max: int,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, reverse_metric_direc)
        self.iter_max = iter_max

    def _impl_update_lr_iter(self):
        """Implementation of your custom LR update method. Defaultly do nothing."""
        return None

    def impl_step(self, metric):
        state = 0
        if self.iter_cur >= self.iter_max:
            state = 2
        elif not (self._reverse_ ^ (metric < self.best_metric)):
            self.best_metric = metric
            state = 1

        self._impl_update_lr_iter()

        return state


class SchedulerWarmupMileStone(SchedulerEarlyStop):
    """MileStone scheduler with warmup
        
    Combine the linear warmup and mile stone decreasing up
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            total_batch_size: int,
            iter_warmup: int,
            refer_batch: int,
            refer_lr: float = 0.,
            lr_stop: float = 1e-5,
            num_ahead: int = 1,
            gamma: float = 0.1,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, 0, lr_stop,
                         num_ahead, gamma, reverse_metric_direc)
        if refer_lr == 0.:
            refer_lr = self.lr_init

        assert total_batch_size > 0
        assert iter_warmup > 0
        assert refer_batch > 0
        assert refer_lr > 0

        self.max_lr = max(total_batch_size/refer_batch * refer_lr, refer_lr)
        if self.lr_init != refer_lr:
            print("Warning: the learning set in optimizer and `refer_lr` are different.")
            self.lr_init = refer_lr
            self._adjust_lr(refer_lr)

        self.iter_warmup = iter_warmup
        self.lr_addon = (self.max_lr-self.lr_init)/iter_warmup

    def impl_step(self, metric):
        if self.iter_cur <= self.iter_warmup:
            state = 0
            if not (self._reverse_ ^ (metric < self.best_metric)):
                self.best_metric = metric
                state = 1
            cur_lr = self.lr_cur
            self._adjust_lr(cur_lr+self.lr_addon)
            return state
        else:
            return super().impl_step(metric)


class SchedulerTransformer(SchedulerFixedStop):
    """
    The standard scheduler of "Attention is all you need"
    peak learning rate peak_factor / sqrt(warmup_steps * d_model)
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            d_model: int,
            warmup_steps: int,
            iter_max: int,
            peak_factor: float = 1.0,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, iter_max, reverse_metric_direc)
        assert d_model > 0
        assert warmup_steps > 0
        self.lr_init = peak_factor/math.sqrt(d_model)
        self._div_warmup_steps = 1./math.sqrt(warmup_steps)/warmup_steps
        self.update_lr_step(1)

    def update_lr_step(self, global_steps: int):
        step = float(global_steps)
        lr = self.lr_init * min(1./math.sqrt(step),
                                step*self._div_warmup_steps)
        self._adjust_lr(lr)


class SchedulerTransformerEarlyStop(SchedulerEarlyStop):
    """
    Linear warmup by step + decay by step + early stop by iteration
    peak lr = peak_factor / sqrt(d_model * warpup_steps)
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            d_model: int,
            warmup_steps: int,
            peak_factor: float = 1.0,
            lr_stop: float = 1e-5,
            num_ahead: int = 1,
            gamma: float = 0.1,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, 0,
                         lr_stop, num_ahead, gamma, reverse_metric_direc)
        assert d_model > 0
        assert warmup_steps > 0
        self.lr_init = peak_factor/math.sqrt(d_model)
        self._div_warmup_steps = 1./math.sqrt(warmup_steps)/warmup_steps
        self.step_cur = 0
        self.warmup_steps = warmup_steps
        self.update_lr_step(1)

    def update_lr_step(self, global_steps: int):
        self.step_cur = global_steps
        step = float(global_steps)
        lr = self.lr_init * min(1./math.sqrt(step),
                                step*self._div_warmup_steps)
        self._adjust_lr(lr)

    def impl_step(self, metric):
        if self.step_cur <= self.warmup_steps:
            if not (self._reverse_ ^ (metric < self.best_metric)):
                self.best_metric = metric

            return 0
        else:
            lr0 = self.lr_cur
            states = super().impl_step(metric)
            lr1 = self.lr_cur
            self.lr_init *= lr1 / lr0
            return states


class SchedulerIterAnnealing(SchedulerFixedStop):
    """
    (Linear) annealing the LR by every evaluation time.
    """

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            decay_factor: float,
            iter_max: int,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, iter_max, reverse_metric_direc)
        assert decay_factor > 0. and decay_factor < 1. and iter_max > 0
        self.decay = decay_factor

    def _impl_update_lr_iter(self):
        lr = self.lr_init * (self.decay ** self.iter_cur)
        self._adjust_lr(lr)


class SchedulerCosineAnnealing(SchedulerFixedStop):
    """Annealing the LR with cosine function (and period) in iteration level."""

    def __init__(
            self,
            optimizer_configs,
            paramlist: Iterable[torch.nn.parameter.Parameter],
            lr_min: float,
            iter_max: int,
            period: int = 0,
            decay_factor: float = 1.,
            reverse_metric_direc: bool = True):
        super().__init__(optimizer_configs, paramlist, iter_max, reverse_metric_direc)
        assert period >= 0 and lr_min >= 0 and iter_max > 0
        assert decay_factor > 0. and decay_factor <= 1.
        if period == 0:
            period = iter_max

        self.period = period
        self.decay = decay_factor
        self.lr_min = lr_min
        self.lr_max = self.lr_init

    def _impl_update_lr_iter(self):
        iter_idx = self.iter_cur - 1
        lr_max = (self.lr_max *
                  self.decay**(iter_idx//self.period))

        lr = self.lr_min + 0.5 * (lr_max - self.lr_min) * (
            1 + np.cos((iter_idx % self.period)/self.period * np.pi))
        self._adjust_lr(lr)
