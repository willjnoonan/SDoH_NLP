import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model
model_dir = "./distilbert_synthetic_finetuned_train50"
model = DistilBertForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model.eval()

# Input sentence
sentence = "Patient is supported by family and friends."

# Tokenize
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of attention matrices

# Use last layer's attention from head 0 (can average across heads too)
last_layer_attention = attentions[-1][0][0].numpy()  # shape: [seq_len, seq_len]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Plot token-level attention
plt.figure(figsize=(10, 8))
sns.heatmap(last_layer_attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.title("Attention Heatmap (Last Layer, Head 0)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
