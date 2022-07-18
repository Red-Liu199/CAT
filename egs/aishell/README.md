## Data
170 hour Mandarin speech data. Mostly reading speech.

**Data prepare**

1. Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data.sh -h
   ```

2. Prepare data with Kaldi: scripts not provided here.
   
   - If you're CAT user, take a look at `Transducer-dev/tools/README.md`.

   - If you have kaldi installed, prepare the files: `text, wav.scp, utt2spk, ...`, then follow

      ```bash
      bash utils/data/data_prep_old.sh <path/to/data> --kaldi-root <path/to/kaldi> \
         --feat-dir data/raw-fbank --not-apply-cmvn
      mkdir -p data/src
      ln -snf $(readlink -f <path/to/data>) data/src/
      python utils/data/resolvedata.py
      ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
python utils/data/resolvedata.py
```
to refresh the information. Manually modifying is also OK.

## Result

Summarize experiments here.

