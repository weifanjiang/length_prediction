import json
import numpy as np
import os
import time
import torch
import tqdm
from collections import Counter
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# configurations
model_dir = "model/alpaca_length_predictor"
data_dir = "dataset/alpaca_processed"
num_labels = 10

# evaluate on test data
os.system("mkdir -p eval")
pred_list, error_list, time_list = list(), list(), list()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels).to('cuda')
model.eval()
test_data = DatasetDict.load_from_disk(data_dir)['test']
texts = test_data['instruction_input_text']
eval_results = list()
pbar = tqdm.tqdm(total=len(texts))
with torch.no_grad():
    for i, seq in enumerate(texts):
        query = tokenizer(seq, truncation=True, padding='max_length', return_tensors='pt').to('cuda')
        start_time = time.time()
        logits = model(**query).logits
        torch.cuda.synchronize()
        pred_time = time.time() - start_time
        pred_label = logits.argmax().item()
        pred_list.append(pred_label)
        logits_np = logits.detach().cpu().numpy()[0]
        pred_probas = np.exp(logits_np)
        pred_probas /= np.sum(pred_probas)
        pred_probas = [float(round(x, 3)) for x in pred_probas]
        error_list.append(abs(test_data['label'][i] - logits.argmax().item()))
        time_list.append(pred_time)

        eval_results.append({
            "instruction_input_text": seq,
            "output": test_data['output'][i],
            "output_token_len": test_data['output_token_len'][i],
            "label": test_data['label'][i],
            "pred_label": pred_label,
            "pred_proba": pred_probas,
            "pred_time": pred_time
        })

        pbar.update(1)

pbar.close()

cnt = Counter(error_list)
print(f"Distance: {sum([k * v for k, v in cnt.items()]) / len(error_list)}")
print(f"Accuracy: {cnt[0] / len(error_list)}")

with open('eval/alpaca_test_eval.json', 'w') as fout:
    json.dump(eval_results, fout, indent=2)
