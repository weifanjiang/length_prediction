import evaluate
import fire
import numpy as np
import os
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



# labeled alpaca dataset with 10 bins
model_name = "alpaca-chunwei_length_predictor"
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_set = load_dataset('json', data_files='dataset/alpaca_chunwei/train.json')['train']
test_set = load_dataset('json', data_files='dataset/alpaca_chunwei/test.json')['train']

def tokenize_function(sequence):
    return tokenizer(sequence['prompt'], padding='max_length', truncation=True)

encoded_train = train_set.map(tokenize_function, batched=True)
encoded_test = test_set.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=9)
metric = evaluate.load('mse')

def compute_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

os.system("mkdir -p model/cache")
args = TrainingArguments(
    os.path.join('model', 'cache', model_name),
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='mse',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test,
    compute_metrics=compute_metric,
)

trainer.train()
model.save_pretrained(os.path.join('model', model_name))

os.system('rm -rf wandb')