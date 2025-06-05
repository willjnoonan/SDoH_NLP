import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load held-out 50% balanced test set 
df = pd.read_csv("SDOH_MIMICIII_balanced_test50.csv")
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

sdoh_categories = ["housing", "employment", "transportation", "relationship", "support", "parent"]
label_columns = [f"{cat}_label" for cat in sdoh_categories]

# Load tokenizer and model 
model_path = "./model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Dataset class 
class SDOHDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

test_dataset = SDOHDataset(df["text"].tolist(), df[label_columns].values, tokenizer)

# Evaluate 
training_args = TrainingArguments(
    output_dir="./tmp_eval_train50",
    per_device_eval_batch_size=16,
    logging_dir="./logs_eval_train50",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

print("Running evaluation on held-out 50% MIMIC test set...")
pred_outputs = trainer.predict(test_dataset)
logits = pred_outputs.predictions
true_labels = df[label_columns].values

# Compute predictions 
probabilities = 1 / (1 + np.exp(-logits))
pred_labels = (probabilities >= 0.5).astype(int)

# Per-label metrics 
print("\nPer-label Precision, Recall, F1:")
for i, cat in enumerate(sdoh_categories):
    p = precision_score(true_labels[:, i], pred_labels[:, i], zero_division=0)
    r = recall_score(true_labels[:, i], pred_labels[:, i], zero_division=0)
    f = f1_score(true_labels[:, i], pred_labels[:, i], zero_division=0)
    print(f" - {cat.capitalize():<14} P={p:.3f}  R={r:.3f}  F1={f:.3f}")

# Overall metrics 
print("\nOverall Micro / Macro / Weighted Averages:")
micro_p = precision_score(true_labels, pred_labels, average="micro", zero_division=0)
micro_r = recall_score(true_labels, pred_labels, average="micro", zero_division=0)
micro_f = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
macro_p = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
macro_r = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
macro_f = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
weighted_f = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

print(f" - Micro     P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f:.3f}")
print(f" - Macro     P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f:.3f}")
print(f" - Weighted           F1={weighted_f:.3f}")

# Subset Accuracy 
subset_acc = accuracy_score(true_labels, pred_labels)
print(f"\nSubset Accuracy (Exact Match on all 6 labels): {subset_acc:.3f}")
