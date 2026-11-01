
# Moroccan Darija Hate Speech Detection

A comparative NLP study for detecting hate speech and abusive language in Moroccan Darija using traditional Machine Learning, Deep Learning, and pretrained Transformer models.

This project was developed as part of a Natural Language Processing practical assignment. The objective is to classify Moroccan Darija tweets into three categories: **Abuse, Hate, and Normal**, while investigating the challenges of class imbalance, code-switching, Arabizi, and orthographic variation.

## 1. Project Overview

Moroccan Darija is a challenging language for NLP due to its non-standardized spelling, Arabic and Latin scripts, frequent code-switching, and limited annotated resources.

In this project, I explored different NLP approaches to understand how well they handle these linguistic characteristics and the difficulty of distinguishing hate speech from general abusive language.

The main objectives were to:

- Explore and preprocess Moroccan Darija tweets.
- Develop classical Machine Learning baselines.
- Train and compare Deep Learning architectures.
- Evaluate pretrained language models using frozen embeddings and fine-tuning.
- Investigate class imbalance, Arabizi, and cross-dialect generalization.
- Analyze model errors, robustness, and computational efficiency.

## 2. Dataset

**Source:** [AfriHate — Hugging Face](https://huggingface.co/datasets/afrihate/afrihate)

**Configuration:** Moroccan Arabic (`ary`)

The dataset contains **4,634 annotated tweets** divided into three official splits.

| Split | Samples |
|---|---:|
| Train | 3,240 |
| Validation | 695 |
| Test | 699 |
| **Total** | **4,634** |

### Classification Labels

- **Abuse:** Offensive or insulting language.
- **Hate:** Hate speech targeting individuals or groups based on identity-related characteristics.
- **Normal:** Non-hateful and non-abusive content.

### Training Class Distribution

| Class | Samples | Percentage |
|---|---:|---:|
| Abuse | 1,588 | 49.01% |
| Normal | 1,182 | 36.48% |
| Hate | 470 | 14.51% |

The dataset is imbalanced, with Hate representing the minority class.

The official train, validation, and test splits were preserved throughout the experiments.

## 3. Data Exploration and Preprocessing

I performed exploratory data analysis to examine class distribution, tweet length, missing values, duplicate records, and writing systems.

The preprocessing pipeline includes:

- URL and user mention replacement.
- Arabic diacritic removal.
- Light Arabic character normalization.
- Reduction of excessive character repetitions.
- Whitespace normalization.
- Preservation of Arabic, Latin, Arabizi, and code-switched text.

I also investigated out-of-vocabulary (OOV) rates using a vocabulary built exclusively from the training data.

| Split | OOV Rate |
|---|---:|
| Train | 27.61% |
| Validation | 40.59% |
| Test | 39.47% |

These results highlight the lexical diversity and spelling variation found in Moroccan Darija.

## 4. Models and Experiments

### A. Traditional Machine Learning

I implemented TF-IDF-based classifiers using word n-grams (1–2) and character n-grams (2–5).

Models:

- Logistic Regression
- Linear SVM
- Class-weighted configurations
- Rule-based baseline using an insult lexicon

### B. Zero-Shot Classification

I evaluated multilingual Natural Language Inference (NLI) with hypotheses written in Arabic, French, and English.

I also tested a pretrained Darija toxicity classifier for binary offensive language detection.

### C. Deep Learning

I trained the following architectures:

- BiLSTM
- BiLSTM with Attention
- Mini-Transformer
- Character-level CNN

Additional experiments included:

- Random vs Word2Vec vs FastText embeddings
- Positional Encoding ablation
- Vocabulary and OOV analysis
- Multiple random seeds

### D. Frozen DarijaBERT

I used pretrained DarijaBERT as a frozen contextual feature extractor and compared:

- CLS embeddings + Logistic Regression
- Mean Pooling + Logistic Regression
- Token embeddings + BiLSTM + Attention

### E. Transformer Fine-Tuning

I performed full fine-tuning of:

- DarijaBERT
- MARBERTv2

I also implemented a Multi-Task Learning experiment combining hate speech classification with auxiliary Arabizi detection.

### F. Robustness and Generalization

The final experiments investigated:

- Validation-based decision adjustment.
- Cross-dialect generalization from Moroccan to Algerian Arabic.
- Performance differences across Arabic, Latin, and mixed writing systems.
- Robustness to text perturbations.
- Bootstrap confidence intervals.
- Qualitative error analysis.
- McNemar statistical testing.

## 5. Evaluation Metrics

Due to class imbalance, Accuracy was not used as the main model-selection criterion.

The primary evaluation metrics were:

- **Macro-F1:** Main metric across the three classes.
- **Hate F1:** Performance on the minority class.
- **PR-AUC:** Area under the Precision–Recall curve for each class.

Confusion matrices and classification reports were also used to analyze prediction errors.

## 6. Experimental Results

### Validation Results

| Model | Macro-F1 | Hate F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.8031 | 0.7059 |
| TF-IDF + Linear SVM | 0.8079 | 0.6740 |
| Zero-Shot NLI (French) | 0.4675 | 0.3316 |
| BiLSTM | 0.7380 | 0.5936 |
| BiLSTM + Attention | 0.7152 | 0.5484 |
| Mini-Transformer (with PE) | 0.6962 | 0.5182 |
| Character CNN | 0.7324 | 0.5852 |
| DarijaBERT + CLS | 0.6850 | 0.4979 |
| DarijaBERT + Mean Pooling | 0.6875 | 0.5145 |
| DarijaBERT + BiLSTM + Attention | 0.7848 | 0.6897 |
| DarijaBERT Fine-Tuning | 0.7973 | 0.6893 |
| **MARBERTv2 Fine-Tuning** | **0.8265** | **0.7283** |

The results correspond to the official validation split. The fine-tuning results shown here use random seed 42.

### Final Test Results

The final MARBERTv2 model was evaluated on 699 Moroccan Darija tweets using a decision adjustment selected on the validation set.

| Metric | Result |
|---|---:|
| Macro-F1 | 0.7668 |
| Hate F1 | 0.6354 |
| PR-AUC Abuse | 0.9137 |
| PR-AUC Hate | 0.7046 |
| PR-AUC Normal | 0.9085 |

### Final Model Comparison

| Metric | Linear SVM | MARBERTv2 |
|---|---:|---:|
| Macro-F1 | 0.7620 | 0.7668 |
| Hate F1 | 0.6082 | 0.6354 |
| Inference Latency | 0.268 ms/tweet | 7.838 ms/tweet |
| Model Size | 3.592 MB | 621.222 MB |

Inference latency was measured on the available Kaggle environment, with SVM running on CPU and MARBERTv2 on a Tesla T4 GPU. The measurements are therefore hardware-dependent.

An exact McNemar test produced a p-value of **0.9196**, indicating no statistically significant difference in the two models' overall proportions of correct predictions.

## 7. Additional Findings

### Embedding Comparison

FastText achieved the best performance among the tested BiLSTM embedding configurations.

| Embedding | Macro-F1 |
|---|---:|
| Random | 0.7380 |
| Word2Vec | 0.7543 |
| FastText | **0.7682** |

### Multi-Task Learning

The auxiliary Arabizi detection task did not improve the overall Macro-F1 in the tested configuration.

The best result was obtained with an auxiliary loss weight of 0, corresponding to no contribution from the auxiliary task.

### Cross-Dialect Generalization

I evaluated MARBERTv2 on Algerian Arabic (`arq`) to investigate domain shift.

| Configuration | Macro-F1 |
|---|---:|
| Before Adaptation | 0.6197 |
| After Adaptation | **0.6632** |

Additional training with Moroccan and Algerian Arabic improved performance on the Algerian test set.

### Model Robustness

MARBERTv2 showed relatively small changes in Macro-F1 after removing emojis, repeated punctuation, or selected insulting words.

### Error Analysis

I examined classification errors and prepared a qualitative annotation sample to investigate:

- Context ambiguity
- Class overlap
- Arabizi and orthographic variation
- Code-switching
- Implicit hate speech
- Potential annotation inconsistencies

The qualitative error annotations are exploratory and require manual validation.

## 8. Technologies

**Programming:** Python

**Machine Learning:** Scikit-learn, TF-IDF, Logistic Regression, Linear SVM

**Deep Learning:** PyTorch, BiLSTM, Attention, CNN, Transformer Encoder

**NLP:** Hugging Face Transformers, DarijaBERT, MARBERTv2, Word2Vec, FastText

**Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn

**Environment:** Kaggle Notebooks, NVIDIA Tesla T4 GPU

## 9. How to Run

1. Open the notebook in Kaggle or a GPU-enabled Jupyter environment.
2. Install the required Python dependencies.
3. Request access to the AfriHate dataset on Hugging Face and authenticate securely.
4. Load the Moroccan Arabic (`ary`) configuration.
5. Execute the notebook sections in order.

Some experiments require GPU acceleration and access to pretrained models from Hugging Face.

The AfriHate dataset is access-restricted. This repository does not redistribute the original dataset.

## 10. Conclusion

This project allowed me to explore several approaches to hate speech detection in Moroccan Darija, from traditional TF-IDF classifiers to pretrained language models and full Transformer fine-tuning.

MARBERTv2 achieved the highest validation Macro-F1 among the main evaluated models. However, its final test performance was close to Linear SVM, despite having significantly higher computational and storage requirements.

The experiments also highlighted the importance of class imbalance, script variation, domain shift, and contextual ambiguity.

Overall, the project demonstrates that advanced Transformers can improve hate speech classification, but simpler Machine Learning approaches remain competitive, especially when computational efficiency is important.

---

**Project Type:** Academic NLP Practical Assignment  
**Language:** Moroccan Darija (ary)  
**Dataset:** AfriHate  
**Environment:** Kaggle / Python / PyTorch
