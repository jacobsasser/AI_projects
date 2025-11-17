# Hate Speech Classification: TF-IDF + Logistic Regression vs BERT
This project compares two approaches for detecting hate speech in text:
- TF-IDF + Logistic Regression (baseline)
- Fine-tuned BERT model (deep learning)  

The goal is to evaluate how classical NLP methods compare to transformer-based models on a real hate-speech classification dataset.

---
# Dataset
Dataset used:
Hate Speech Dataset — Mendeley Data
`https://data.mendeley.com/datasets/9sxpkmm8xn/1`

The dataset contains labeled text samples annotated as Hateful (1) or Not Hateful (0).

---
# Preprocessing (Baseline Model)
For the TF-IDF + Logistic Regression baseline:
- Clean text (remove punctuation, URLs, numbers)
- Remove stopwords
- Apply either lemmatization or stemming (Snowball stemmer selected)
- Convert cleaned text into TF-IDF features
- Train a Logistic Regression classifier

This establishes the classical NLP baseline.

---
# BERT Model
A custom PyTorch dataset and tokenizer pipeline was built using `class HateSpeechDataset(Dataset)`.
Each sample is tokenized with:
- BertTokenizer (bert-base-uncased)
- Max length = 128
- Attention masks included  

A custom classifier is fine-tuned on top of BERT (`class BERTClassifier(nn.Module)`):
- Uses pretrained “bert-base-uncased”
- Only top layers are unfrozen (layers 10+, pooler)
- Hidden layer → ReLU → Dropout → Linear → Output
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (2e-5)
- Learning rate scheduler
- Gradient clipping (1.0)
- Early stopping (patience = 2)

---
# Training Pipeline
Both models use:
- Train/validation split
- Accuracy, precision, recall, F1
- Confusion matrix
- Early stopping based on validation loss

---
# Model Interpretability (SHAP)
The fine-tuned BERT model is interpreted using SHAP text explainers:
- Generates token-level importance scores
- Highlights which words contribute most to hateful vs non-hateful predictions
- Visualization saved as shap_plot.png

---
# Results and Conclusion
Across all experiments, the TF-IDF + Logistic Regression baseline performed very well on smaller sample sizes, showing strong accuracy even with limited data. However, as the dataset grew, the fine-tuned BERT model became the stronger performer, ultimately achieving higher overall accuracy and more reliable predictions. In addition, SHAP interpretability analysis demonstrated that the BERT classifier was able to capture meaningful token-level cues associated with hateful or non-hateful language, providing transparent insight into the model’s decision-making process.
