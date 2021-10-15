# Requirement of third-party tools

## CAT

- Repository: https://github.com/thu-spmi/CAT
- CAT is required for data pre-processing, including the feature  extraction and some of the text normalization. So if you want to change the feature extraction settings, you should set it in `tools/CAT/`, probably in `tools/CAT/egs/<task>/run.sh`.
- We assume the required train/dev/test files satisfy the path format:
   ```
   tools/CAT/egs/<task>/data/
    ├── all_ark
    │   ├── <train_1>.scp
    │   ├── <dev_1>.scp
    │   ├── <test_1>.scp
    │   └── <test_2>.scp
    ├── <test_1>
    │   └── text
    ├── <test_2>
    │   └── text
    ├── <dev_1>
    │   └── text
    └── <train_1>
        └── text
   ```
   Just ensure there are corresponding text files in the directories w.r.t. `.scp` files. Multiple train/dev/test is OK.

   **Notes:**
   
   The contents of `*.scp` files are indeed the path of binary files (usually `.ark` files). Some of the recipes of CAT only store relative path in the `.scp` file, in such case, programs in a different working space are not able to read the correct path of binary files.
   
   To address this issue, you can replace the relative path with absolute path via
   ```bash
   cat <src.scp> | sed "s|<rel path>|<abs path>|g" > <dst.scp>
   # or making an in-place modification
   sed -i "s|<rel path>|<abs path>|g" <src.scp>
   ```