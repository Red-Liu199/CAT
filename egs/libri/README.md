## Data
960 hour English speech data. Book reading speech.

**Data prepare**

- Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data.sh -h
   ```

- Prepare data with Kaldi

   ```bash
   bash local/data_kaldi.sh -h
   ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
cd /path/to/libri
python utils/data/resolvedata.py
```
to refresh the information. Manually modifying is also OK.

## Result

Summarize experiments here.

