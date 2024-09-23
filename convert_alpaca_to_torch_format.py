import fire
import json
import numpy as np
import os
import pickle as pkl
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch import nn, optim


def main(data_dir, output_dir, seed, num_labels=10):

    with open(os.path.join(data_dir, 'split', f'seed{seed}.json'), 'r') as fin:
        split = json.load(fin)
    train_ids = set(split['train'])
    test_ids = set(split['test'])

    with open(os.path.join(data_dir, 'llama3-10-15-5k.json'), 'r') as fin:
        records = json.load(fin)['records'][1:]

    ds = list()
    for r in tqdm.tqdm(records):
        raw_prompt = r['output']
        pstart = raw_prompt.index('Below is an instruction')
        pend = raw_prompt.index('### Response:')
        prompt = raw_prompt[pstart:pend] + '### Response:'
        ds.append(
            {
                "id": int(r['record_id']),
                "prompt": prompt,
                "groundtruth": r['iteration_count'],
                "label": int(np.digitize(r['iteration_count'], np.linspace(0, 512, num_labels + 1), right=True) - 1)
            }
        )
    
    all_labels = [x['label'] for x in ds]
    print(f'min label {np.amin(all_labels)}, max label {np.amax(all_labels)}')
    

    path = os.path.join(output_dir, f"seed{seed}")
    os.system(f'mkdir -p {path}')

    train_ds, test_ds =  [x for x in ds if x['id'] in train_ids], [x for x in ds if x['id'] in test_ids]

    with open(os.path.join(path, 'train.json'), 'w') as fout:
        json.dump(train_ds, fout, indent=2)
    
    with open(os.path.join(path, 'test.json'), 'w') as fout:
        json.dump(test_ds, fout, indent=2)


if __name__ == '__main__':
    fire.Fire(main)
