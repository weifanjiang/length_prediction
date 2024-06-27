import fire
import json
import numpy as np
import os
import time
import torch
import tqdm
from collections import Counter
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def prepare_alpaca(num_labels, test_size, max_token_len):
    data_path = "dataset/alpaca"
    if not os.path.isdir(data_path):
        os.system('mkdir -p dataset')
        os.system('git clone https://huggingface.co/datasets/tatsu-lab/alpaca')
        os.system('mv alpaca dataset/alpaca')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bin_size = max_token_len // num_labels

    def combine_instruction_and_input(sequence):
        # remove response from text
        # return {"instruction_input_text": sequence['text'].split("\n\n### Response:")[0] + "\n\n### Response: ASSISTANT:"}
        return {"instruction_input_text": sequence['instruction'] + sequence['input']}

    def get_label(sequence):
        # token_len = len(tokenizer(sequence['output'], truncation=True)['input_ids'])
        token_len = 0
        words = sequence['output'].split(" ")
        num_chunks = int(np.ceil(len(words) / 500))
        for chunk_idx in range(num_chunks):
            chunk = words[500*chunk_idx:min(500*chunk_idx + 500, len(words))]
            token_len += len(tokenizer(" ".join(chunk), truncation=True)['input_ids'])

        if token_len >= bin_size * num_labels:
            label = num_labels - 1
        else:
            label = (token_len - 1) // bin_size
        return {"output_token_len": token_len, "label": label}

    dataset = load_dataset('dataset/alpaca')['train']
    label_dataset = dataset.map(combine_instruction_and_input).map(get_label)
    dataset_dict = label_dataset.train_test_split(test_size=test_size, shuffle=True, seed=10)
    dataset_dict.save_to_disk('dataset/alpaca_processed')


def prepare_ggqa(num_labels, test_size, max_token_len):
    # Download gz file from
    # https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz
    # then `gzip -d dataset/google-qa/v1.0-simplified_simplified-nq-train.jsonl.gz`
    data_path = "dataset/google-qa"
    assert(os.path.isfile(os.path.join(data_path, "v1.0-simplified_simplified-nq-train.jsonl")))
    if not os.path.isfile(os.path.join(data_path, 'ggqa.json')):
        pbar = tqdm.tqdm(total=307373)  # magic number
        parsed_qa = list()
        f = open(os.path.join(data_path, "v1.0-simplified_simplified-nq-train.jsonl"), 'r')
        line = f.readline()
        total, invalid, no_answer = 0, 0, 0
        while line:
            pbar.update(1)
            total += 1
            try:
                dat = json.loads(line)
            except Exception:
                dat = None
                invalid += 1
            
            if dat is not None:
                words = dat['document_text'].split(" ")
                la = dat['annotations'][0]['long_answer']
                st, et = la['start_token'], la['end_token']
                if (st != -1) and (et != -1):
                    parsed_qa.append({
                        "id": dat['example_id'],
                        "question": dat['question_text'],
                        "answer": " ".join(words[st:et])
                    })
                else:
                    no_answer += 1
            line = f.readline()
        f.close()
        pbar.close()

        with open(os.path.join(data_path, 'ggqa.json'), 'w') as fout:
            json.dump(parsed_qa, fout, indent=2)
        
        print(f'total samples {total}; invalid format {invalid}; no answer {no_answer}')
        
    if not os.path.isdir(os.path.join('dataset', 'google-qa_processed')):
        dataset = load_dataset('json', data_files=os.path.join(data_path, 'ggqa.json'))['train']
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        bin_size = max_token_len // num_labels
        
        def get_label(sequence):
            token_len = 0
            words = sequence['answer'].replace("<P>", "").replace("</P>", "").strip().split(" ")
            num_chunks = int(np.ceil(len(words) / 500))
            for chunk_idx in range(num_chunks):
                chunk = words[500*chunk_idx:min(500*chunk_idx + 500, len(words))]
                token_len += len(tokenizer(" ".join(chunk), truncation=True)['input_ids'])

            if token_len >= bin_size * num_labels:
                label = num_labels - 1
            else:
                label = (token_len - 1) // bin_size
            return {"output_token_len": token_len, "label": label}
        
        label_dataset = dataset.map(get_label)
        dataset_dict = label_dataset.train_test_split(test_size=test_size, shuffle=True, seed=10)
        dataset_dict.save_to_disk(os.path.join('dataset', 'google-qa_processed'))


def main(
    num_labels=10,
    test_size=0.1,
    max_token_len=2048,
    dataset_name="alpaca"
):
    os.system("mkdir -p dataset")

    match dataset_name:
        case "alpaca":
            prepare_alpaca(num_labels, test_size, max_token_len)
        case "google-qa":
            prepare_ggqa(num_labels, test_size, max_token_len)


if __name__ == '__main__':
    fire.Fire(main)
