# Transducer toolkit for speech recognition

## Results

- [Librispeech](egs/libri)
- [AIshell-1](egs/aishell)

## Installation

1. Install main dependencies
   - PyTorch: `>=1.9.0` is recommended
   - CAT: https://github.com/thu-spmi/CAT only for data preparation. After installing the CAT, please refers to the details in tools/README.md
   and link directory.
      ```bash
      ln -s <CAT> tools/
      ```

2. Python packages

   ```bash
   git clone git@github.com:maxwellzh/Transducer-dev.git
   cd Transducer-dev/
   pip install -r requirements.txt
   ```

3. Building packages from source

   ```bash
   cd src/
   git submodule init && git submodule update
   
   # warp-rnnt >= 0.7.0
   cd warp-rnnt/pytorch_binding/
   git checkout -t origin/dev
   python setup.py install
   cd ../../
   
   # gather >= 0.2.1
   cd torch-gather/
   python setup.py install
   cd ../
   ```

