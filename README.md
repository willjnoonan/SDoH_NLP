MIT License

Copyright (c) 2025 Will Noonan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell    
copies of the Software, and to permit persons to whom the Software is         
furnished to do so, subject to the following conditions:                      

The above copyright notice and this permission notice shall be included in    
all copies or substantial portions of the Software.                           

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN     
THE SOFTWARE.

This repository uses public data from the MIMIC-III Clinical Care Database (via PhysioNet) and associated annotations released under PhysioNet’s data license. Users must comply with the relevant PhysioNet data usage agreement:
https://physionet.org/content/sdoh-mimic/1.0.1/

Overview:
This project applies Natural Language Processing (NLP) to automatically identify Social Determinants of Health (SDoH) in unstructured clinical text. It uses a two-stage approach: pretraining a DistilBERT model on synthetic sentences and fine-tuning it on annotated real data (from MIMIC-III), with additional evaluation tools and attention visualizations.

- bert_synthetic_original.py — Pretrains DistilBERT on synthetic sentences
- bert_finetune_train50.py — Fine-tunes the pretrained model on 50% balanced MIMIC-III SDoH data
- evaluate_real_data.py — Evaluates the fine-tuned model on a held-out clinical test set
- heatmap.py — Visualizes attention maps for qualitative inspection
- *.csv — Includes synthetic and real SDoH-labeled datasets

Information:
- Multi-label classification for 6 SDoH factors:
    - housing, employment, transportation, relationship, support, parent
- Binary prediction for each category using sigmoid outputs
- Preprocessing and tokenization using HuggingFace Transformers
- Training with HuggingFace Trainer API and PyTorch
- Performance metrics: F1-score (micro/macro), accuracy, and attention heatmaps

Install dependencies with:
  pip install transformers pandas scikit-learn torch matplotlib seaborn

To Run:
  1. Pretrain on synthetic data:
  python bert_synthetic_original.py
  
  2. Fine-tune on real clinical data:
  python bert_finetune_train50.py
  
  3. Evaluate on test data:
  python evaluate_real_data.py
  
  4. Visualize attention:
  python heatmap.py

Additional Notes:
- All models are multi-label: each sentence can trigger multiple SDoH predictions.
- Synthetic data improves generalization for underrepresented categories.
- Best performance observed in employment, relationship, and support labels
- Ensure "model" is in the same directory

