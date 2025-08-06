# Email Spam Classification (Week 3 – DEN DS Internship)

Classifies messages as **Spam** or **Ham** using classic NLP + ML.
Best model: **Linear SVM** with **TF–IDF (1–2 grams)**.

##  Project Structure
- `notebook/` – Final Jupyter notebook with end-to-end workflow
- `results/` – Saved metrics and plots

## Dataset
- Name: `spam.csv` (~5.5k messages; columns: `Category`, `Message`)
- Source: Public SMS/Email spam dataset (Kaggle/UCI variants exist)
- **Not included** in repo for size/license hygiene. 

##  Tech Stack
- Python, scikit-learn, NLTK
- TF–IDF Vectorizer: `ngram_range=(1, 2)`, `lowercase=True`, `min_df=1`
- Models tried: Multinomial NB, Logistic Regression, Linear SVM

##  Results (hold-out test)
| Model                  | Acc  | Prec(Spam) | Rec(Spam) | F1(Spam) |
|------------------------|------|------------|-----------|----------|
| Multinomial NB         | 0.971| 0.962      | 0.905     | 0.933    |
| Logistic Regression    | 0.982| 0.977      | 0.934     | 0.955    |
| **Linear SVM (Best)**  | 0.984| 0.981      | 0.934     | 0.957    |


##  How to Run
```bash
# 1) create env
python -m venv .venv && . .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate                   # macOS/Linux

pip install -r requirements.txt  # (generate this from your notebook)

# 2) place spam.csv in data/ 
# 3) run notebook or scripts
jupyter lab  # open notebook/spam_classification.ipynb
