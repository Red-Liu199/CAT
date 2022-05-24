## Data
170 hour Mandarin speech data. Mostly reading speech.

**Data prepare**

1. Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data.sh -h
   ```

2. Prepare data with Kaldi: scripts not provided here.
   
   - If you're CAT user, take a look at `Transducer-dev/tools/README.md`.

   - If you have kaldi installed, prepare the files:`feat.ark, feat.scp, text`, and place them as (`feat.ark` can be placed anywhere, just ensure path of it in `feat.scp` is correct.)

        ```
        data/
        └── src/
            ├── all_ark
            │   └── feat.scp
            └── train_sp
                └── text
        ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
python utils/data/resolvedata.py
```
to refresh the information.

## Result

Summarize experiments here.

