import uuid
import argparse
import itertools

import evaluate
import numpy as np
from sklearn.metrics import classification_report

from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model path')
parser.add_argument('--name', type=str, help='Name directory')
args_input = parser.parse_args()

task = "ner"

batch_size = 8

EPOCHS = 100

# dataset = load_dataset("Dr-BERT/QUAERO", "emea")
dataset = load_dataset("Dr-BERT/QUAERO", "medline")

train_dataset = dataset["train"]
print(train_dataset)

dev_dataset = dataset["validation"]
print(dev_dataset)

test_dataset = dataset["test"]
print(test_dataset)

label_list = train_dataset.features[f"{task}_tags"].feature.names
print(label_list)

def getConfig(raw_labels):

    label2id = {}
    id2label = {}

    for i, class_name in enumerate(raw_labels):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name

    return label2id, id2label

label2id, id2label = getConfig(label_list)

tokenizer = AutoTokenizer.from_pretrained("Dr-BERT/DrBERT-7GB")
model = AutoModelForTokenClassification.from_pretrained("Dr-BERT/DrBERT-7GB", num_labels=len(label_list))
model.config.label2id = label2id
model.config.id2label = id2label

def tokenize_and_align_labels(examples):

    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True, max_length=512)

    labels = []

    for i, label in enumerate(examples[f"{task}_tags"]):

        label_ids = []
        previous_word_idx = None
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            
            previous_word_idx = word_idx

        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels

    return tokenized_inputs

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
dev_tokenized_datasets = dev_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)

output_name = f"DrBERT-QUAERO-{task}-{str(uuid.uuid4().hex)}"

args = TrainingArguments(
    output_name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    greater_is_better=True,
)

print('Load Metrics')
metric  = evaluate.load("seqeval", experiment_id=output_name)
data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    macro_values = [results[r]["f1"] for r in results if "overall_" not in r]
    macro_f1 = sum(macro_values) / len(macro_values)

    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"], "macro_f1": macro_f1}

trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=dev_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.evaluate()
trainer.save_model(f"BEST-{output_name}")

# ------------------ EVALUATION ------------------

predictions, labels, _ = trainer.predict(test_tokenized_datasets)
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

f1_score = classification_report(
    list(itertools.chain.from_iterable(true_labels)),
    list(itertools.chain.from_iterable(true_predictions)),
    digits=4,
)
print(f1_score)