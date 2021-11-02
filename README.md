# Transducer toolkit for speech recognition

## Usage

**Data preparation:**

Currently, this repository is relied on [CAT](https://github.com/thu-spmi/CAT) for data pre-processing. Basically the audio features and some of the text normalization.

So, before going into the task, you should do data preparation with CAT first. For more information, please refer to [tools/README.md](tools/README.md).

**Training/inference:**

In this repo, we support training RNN-T, Language model and CTC/CTC-CRF model training as well as the inference/decoding.

- **RNN-T:** refer to `egs/<recipe>/template/run_rnnt.sh`
- **LM:** refer to `egs/<recipe>/template/run_lm.sh`
- **CTC/CTC-CRF:** this can be regarded as a replica of CAT, but with better and pretty training procedure monitoring. Unfortunately, there's no available `run_ctc.sh` like RNN-T and LM training. I'll add it in the future, but that is not on my current schedule. And training CTC/CTC-CRF requires the `ctc_crf` to be installed. Refer to CAT installation for more details.

## In-house SOTA Results 

- [Librispeech](egs/libri): 2.37/5.46 WER% for test-clean/test-other
- [AIshell-1](egs/aishell): 4.77 CER%

## Installation

1. Install main dependencies
  
   - CUDA compatible machine, NVIDIA driver installed and NVIDIA toolkit available.
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
   ```

