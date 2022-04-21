# Transducer toolkit for speech recognition

## TODO

see personal notes.

## Installation

0. Clone the repo

    ```bash
    git clone git@github.com:maxwellzh/Transducer-dev.git
    ```

1. Main dependencies

   I test the codes with `cudatoolkit==11.3 torch==1.11`.
  
   - CUDA compatible device, NVIDIA driver installed and CUDA available.
   - PyTorch: `>=1.9.0` is required. [Installation guide from PyTorch](https://pytorch.org/get-started/locally/#start-locally)
   - [CAT](https://github.com/thu-spmi/CAT)**\[optional\]**: used for speech data preparation based on kaldi tool, [installation guide](tools/README.md#cat).
      
      Or you can use `egs/[task]/local/data.sh` to process data with `torchaudio`.
   
2. Install basic packages

   ```bash
   ./install.sh
   ```

## Examples and Usage

To get started with this project, please refer to [TEMPLATE](egs/TEMPLATE/README.md) for tutorial.

Once you're more familiar with the project, please read [the document](configure_guide.md) for configuring the training.
