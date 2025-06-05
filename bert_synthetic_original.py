import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os

#  Load Original Synthetic CSV Files 
synthetic_paths = [
    "SyntheticSentences_Round1.csv",
    "SyntheticSentencs_Round2.csv",
    "ManuallyAnnotatedSyntheticSentences.csv"
]

sdoh_categories = ["housing", "employment", "transportation", "relationship", "support", "parent"]
label_columns = [f"{cat}_label" for cat in sdoh_categories]

#  Convert Long-Format to Wide-Format 
def load_and_format(path):
    df = pd.read_csv(path).dropna(subset=["text", "label"]).drop_duplicates(subset=["text", "label"])
    df["label"] = df["label"].str.lower().str.strip()
    df_wide = pd.DataFrame(0, index=df["text"].unique(), columns=label_columns)
    for cat in sdoh_categories:
        mask = df["label"] == cat
        matched = df[mask]["text"].unique()
        df_wide.loc[matched, f"{cat}_label"] = 1
    df_wide = df_wide.reset_index().rename(columns={"index": "text"})
    return df_wide

synthetic_dfs = [load_and_format(p) for p in synthetic_paths]
df = pd.concat(synthetic_dfs, ignore_index=True)
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

# Split 
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training on {len(train_df)}, validating on {len(val_df)}")

#  Tokenizer 
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

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

#  Datasets 
train_dataset = SDOHDataset(train_df["text"].tolist(), train_df[label_columns].values, tokenizer)
val_dataset = SDOHDataset(val_df["text"].tolist(), val_df[label_columns].values, tokenizer)

#  Model 
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6, problem_type="multi_label_classification")

training_args = TrainingArguments(
    output_dir="./distilbert_synthetic_original",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#  Train 
print("Training only on original synthetic data...")
trainer.train()

#  Save 
trainer.save_model("./distilbert_synthetic_original")
tokenizer.save_pretrained("./distilbert_synthetic_original")