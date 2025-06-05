import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score

# Load new 50% balanced training data 
df = pd.read_csv("SDOH_MIMICIII_balanced_train50.csv")

# Drop empty text and define label columns
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

sdoh_categories = ["housing", "employment", "transportation", "relationship", "support", "parent"]
label_columns = [f"{cat}_label" for cat in sdoh_categories]

# Tokenizer 
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_synthetic_original")

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

train_dataset = SDOHDataset(df["text"].tolist(), df[label_columns].values, tokenizer)

# Load model pretrained on synthetic data 
model = DistilBertForSequenceClassification.from_pretrained("./distilbert_synthetic_original")

# Training arguments 
training_args = TrainingArguments(
    output_dir="./distilbert_synthetic_finetuned_train50",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs_train50",
    logging_steps=10,
    save_strategy="epoch"
)

# Metrics 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds)
    }

# Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train 
print("Starting fine-tuning (6 epochs) on 50% balanced MIMIC...")
trainer.train()

# Save 
print("Saving model to ./distilbert_synthetic_finetuned_train50")
trainer.save_model("./distilbert_synthetic_finetuned_train50")
tokenizer.save_pretrained("./distilbert_synthetic_finetuned_train50")