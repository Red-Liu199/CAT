## Data
170 hour Mandarin speech data. Mostly reading speech.

**Data prepare**

1. Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data.sh -h
   ```

2. Prepare data with Kaldi:

   - You should first have Kaldi tool installed.
   
   - Get help about how to use Kaldi to prepare data:

      ```bash
      KALDI_ROOT=<path/to/kaldi> bash local/data_kaldi.sh -h
      ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
cd /path/to/aishell
python utils/data/resolvedata.py
```
to refresh the information. Manually modifying is also OK.

## Result

Summarize experiments here.

