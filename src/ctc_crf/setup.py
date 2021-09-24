#!/usr/bin/env python3
'''
Copyright 2018-2019 Tsinghua University, Author: Hu Juntao (hujuntao_123@outlook.com)
Apache 2.0.
This script is used to install ctc_crf_base_1_0 which depends on the ctc_crf native codes.
In this script we use cpp codes binding_1_0.h, binding_1_0.cpp to integrate the ctc fuctions.
This install script is used for the pytorch version 1.0.0 or later.
'''

import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


if not torch.cuda.is_available():
    raise Exception("CPU version is not implemented")


den_dir = os.path.realpath("./gpu_den/build")
ctc_dir = os.path.realpath("./gpu_ctc/build")

if __name__ == "__main__":
    setup(name='ctc_crf',
          version="0.1.0",
          packages=find_packages(),
          ext_modules=[
              CppExtension(
                  name='ctc_crf._C',
                  language='c++',
                  sources=['binding.cpp'],
                  library_dirs=[ctc_dir, den_dir],
                  libraries=['fst_den', 'warpctc'],
                  extra_link_args=['-Wl,-rpath,' +
                                   ctc_dir, '-Wl,-rpath,'+den_dir],
                  extra_compile_args=['-std=c++14',
                                      '-fPIC', '-I/usr/local/cuda/include']
              ),
          ],
          cmdclass={
              'build_ext': BuildExtension})
