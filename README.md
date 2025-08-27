# BERT-Sentiment-Analysis

Deep learning project using **BERT** for **sentiment classification** (positive, negative, neutral) and **emotion detection** (happy, sad, angry, surprise, disgust) on social-media style text.

---

## Goal / Objective
To analyze and classify sentiments and emotions in social media text using BERT, enabling applications in content moderation, mental health monitoring, and public sentiment analysis.

---

## Key Questions Answered
- Which sentiments and emotions are most prevalent in the dataset?
- How accurately can BERT classify short, noisy, social-media text?
- How does performance vary across emotion classes?
- What patterns emerge in distributions and confusion matrices?

---

## Steps Followed
- **Data Preparation:** Cleaned and preprocessed Twitter-style datasets (e.g., Sentiment140, SMILE): removed links, special characters, stopwords; filtered irrelevant labels.
- **Data Analysis:** Tokenized with BERT tokenizer; encoded labels; split train/validation.
- **Modeling:** Fine-tuned **bert-base-uncased** for sentiment/emotion classification using PyTorch + Transformers.
- **Visualization:** Produced confusion matrix, class distribution, and trend/length plots.
- **DAX Measures:** _N/A (Python-based project); metrics computed via scikit-learn._

---

## Key Insights
- **Weighted F1 Score:** **86%**
- **Class-wise Accuracy (example run):**
  - Happy: 95.9%  
  - Angry: 77.8%  
  - Not-Relevant: 62.5%  
  - Sad: 40%  
  - Surprise: 40%  
  - Disgust: 0%
- **Strengths:** Context-aware classification of short, noisy text; robust vs. traditional ML baselines.
- **Limitations:** Lower performance on underrepresented classes; sarcasm and domain shifts can reduce accuracy.

> _Note: Metrics reflect a specific training/evaluation run and may vary with data splits, seeds, and hardware._

---

## Tools Used
- **Python:** pandas, numpy, regex (preprocessing)
- **Modeling:** PyTorch, HuggingFace Transformers (BERT)
- **Evaluation:** scikit-learn (accuracy, precision, recall, F1)
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook

---

## How to Run
1. **Install dependencies**
   ```bash
   pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
