import evaluate
import fire
import numpy as np
import os
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def main(
    num_labels=10,
    batch_size=32,
    metric_name='mse',
    dataset_name='alpaca',
    input_column_name='instruction_input_text',
    model_name=None
):
    # labeled alpaca dataset with 10 bins
    data_dir = f"dataset/{dataset_name}_processed"
    model_name = f"{dataset_name}_length_predictor"


    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = DatasetDict.load_from_disk(data_dir)

    def tokenize_function(sequence):
        return tokenizer(sequence[input_column_name], padding='max_length', truncation=True)

    encoded_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    metric = evaluate.load(metric_name)

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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['test'],
        compute_metrics=compute_metric,
    )

    trainer.train()
    model.save_pretrained(os.path.join('model', model_name))

    os.system('rm -rf wandb')


if __name__ =="__main__":
    fire.Fire(main)
