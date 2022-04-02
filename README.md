# Transducer toolkit for speech recognition

## TODO

- [ ] Internal language model estimation (ongoing)
    - [x] evaluate the ILME method on RNN-T RNA decode
    - [ ] get RNN-T decode methods all ready for ILME
- [x] rename `*_process.*` -> `*_pipe.*`

## Installation

0. Clone the repo

    ```bash
    # clone main repo only, for LM training
    git clone git@github.com:maxwellzh/Transducer-dev.git
    # for all functions
    git clone --recurse-submodules git@github.com:maxwellzh/Transducer-dev.git
    ```

1. Main dependencies

   I test the codes with `cudatoolkit==11.3 torch==1.11`.
  
   - CUDA compatible machine, NVIDIA driver installed and CUDA available.
   - PyTorch: `>=1.9.0` is required.
   - KenLM: [installation guide](tools/README.md#kenlm)
   - [CAT](https://github.com/thu-spmi/CAT)**\[optional\]**: used for speech data preparation based on kaldi tool, [installation guide](tools/README.md#cat).
      
      Use `egs/[task]/local/data.sh` to process data with `torchaudio`.
   
2. Python packages

   ```bash
   cd Transducer-dev/
   pip install -r requirements.txt
   ```

3. Building packages from source:

   **LM task only:**
   
   ```bash
   cd src/
   git submodule init & git submodule update torch-gather/
   
   # gather >= 0.2.2
   cd torch-gather/
   python setup.py install
   ```
   
   **ASR (All functions):**
   
   Please refer to [src/INSTALL](src/INSTALL) for installation instruction.

## Examples and Usage

To get started with this project, please refer to [TEMPLATE](egs/TEMPLATE/README.md) for tutorial.

Once you're more familiar with the project, please read [the document](configure_guide.md) for configuring the training.
