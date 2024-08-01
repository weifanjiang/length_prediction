# LLM response length prediction

## Llama-Last-Layer-Tensor dataset:

Prompt information file: `/data/weifan/lllt/profiling-Meta-Llama-3-8B-Instruct.json` can be parsed with

```python
with open('/data/weifan/lllt/profiling-Meta-Llama-3-8B-Instruct.json', 'r') as fin:
    instruction = json.load(fin)

records = instruction['records']
records = records[1:]  # first entry
```

The `record` object is a 10K sized list. Each entry is formatted as follows:
```
{'record_id': 0,
 'start_ts': '2024-06-27 14:55:33.616826',
 'record_prompt': ...,
 'output': ...,
 'output_ts': '2024-06-27 14:55:39.891310',
 'iteration_count': 230,
 'iterations': [{'iter_id': 0,
   'tensor_size': [1, 44, 4096],
   'iter_ts': '2024-06-27 14:55:33.730154'},
  {'iter_id': 1,
   'tensor_size': [1, 1, 4096],
   'iter_ts': '2024-06-27 14:55:33.744979'},
   ...
 ]
}
```

`record_id` is the alpaca data prompt id. `iteration_count` is the total number of iterations. The `iterations` field is a list of dictionaries. The `iter_ts` field is a timestamp, and the embedding is located at `f'/data/weifan/lllt/llama_last_layer_tensor/{iter_ts}+.pt'`.

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
