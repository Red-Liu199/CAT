# Train TRF-LM with DNCE
The training and testing process is basically consistent with [Train GN-ELM with DNCE](../GN-ELM-DNCE/). We only explain the differences here.
## Notes
* **In stage 2 (data packing)**, for training TRF, we need to calculate the length distribution after packing data and before training.
```
python -m cat.lm.trf.prep_feats exp/TRF-LM-DNCE/pkl/train.pkl exp/TRF-LM-DNCE/linfo.pkl
```

## Result
We also try 3 different energy functions, whose results are as follows:
|CER type     | SumTargetLogit |  Hidden2Scalar  | SumTokenLogit |
| -------     | -------- | ----------- | ----------- |
| in-domain   | 8.97     |  8.95       |  9.00       |
| cross-domain| 15.77     |  15.67       |  15.65       | 

The training curve of the best model (Hidden2Scalar) is shown below.
|     training curve    |
|:-----------------------:|
|![monitor](./monitor.png)|