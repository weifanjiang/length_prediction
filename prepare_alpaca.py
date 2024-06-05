import numpy as np
import os
import time
import torch
from collections import Counter
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


num_labels = 10
test_size = 0.1

max_token_len = 512

if not os.path.isdir('dataset/alpaca'):
    os.system('mkdir -p dataset')
    os.system('git clone https://huggingface.co/datasets/tatsu-lab/alpaca')
    os.system('mv alpaca dataset/alpaca')

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bin_size = max_token_len // num_labels

def combine_instruction_and_input(sequence):
    # remove response from text
    return {"instruction_input_text": sequence['text'].split("\n\n### Response:")[0] + "\n\n### Response: ASSISTANT:"}

def get_label(sequence):
    token_len = len(tokenizer(sequence['output'], truncation=True)['input_ids'])
    if token_len == max_token_len:
        label = num_labels - 1
    else:
        label = (token_len - 1) // bin_size
    return {"output_token_len": token_len, "label": label}

dataset = load_dataset('dataset/alpaca')['train']
label_dataset = dataset.map(combine_instruction_and_input).map(get_label)
dataset_dict = label_dataset.train_test_split(test_size=test_size, shuffle=True, seed=10)
dataset_dict.save_to_disk('dataset/alpaca_processed')

# display statistics for dataset
print('Label distribution:')
for name in ['train', 'test']:
    ds = dataset_dict[name]
    bin_idx, bin_size = np.unique(ds['label'], return_counts=True)
    si = np.argsort(bin_idx)
    bin_idx, bin_size = bin_idx[si], bin_size[si]

    output = "{} :".format(name)
    for bi, bs in zip(bin_idx, bin_size):
        output += f" {bi} ({bs}),"
    output = output[:-1]
    print(output)
