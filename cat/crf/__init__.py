# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""CTC-CRF-related modules
"""


from .train import build_model as am_builder

__all__ = [am_builder]
