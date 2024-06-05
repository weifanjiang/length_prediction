# LLM response length prediction

## DistilBert classification predictor

From [S3](https://openreview.net/forum?id=zUYfbdNl1m&referrer=%5Bthe%20profile%20of%20Chun-Feng%20Wu%5D(%2Fprofile%3Fid%3D~Chun-Feng_Wu1)).

Model: finetuned `DistilBert` model on `alpaca` dataset, using 90% for training and 10% for testing.

Label: actual output lengths, truncated by max sequence length value (512), into 10 bins.

Instructions:

```
python3 prepare_alpaca.py
python3 finetune_bert.py
python3 eval_bert.py
```

Train/test label distributions:
```
train : 0 (25740), 1 (13109), 2 (5625), 3 (1343), 4 (545), 5 (239), 6 (98), 7 (48), 8 (21), 9 (33)
test : 0 (2827), 1 (1465), 2 (658), 3 (138), 4 (62), 5 (22), 6 (13), 7 (9), 8 (4), 9 (3)
```

Results:
```
Distance: 0.35723899250144203
Accuracy: 0.7071716977504326
```

See `analysis.ipynb` for details.
