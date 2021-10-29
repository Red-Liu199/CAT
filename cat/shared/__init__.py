# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Zheng Huahuan (maxwellzh@outlook.com)

"""Common files and variables for different trainer.
"""

FMT_CHECKPOINT = r"checkpoint.{:03}.pt"
FILE_WRITER = r"training.summary"


from .manager import Manager
from ._specaug import SpecAug

