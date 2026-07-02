# 📧 Email Spam Classification using Machine Learning

A Machine Learning and Natural Language Processing (NLP) project that classifies email/SMS messages as **Spam** or **Ham (Not Spam)** using multiple classification algorithms. The project compares different models and selects the best-performing classifier based on evaluation metrics.

---

## 📌 Project Overview

Spam detection is one of the most common applications of Natural Language Processing (NLP). This project builds a text classification model capable of automatically identifying whether a message is **Spam** or **Ham**.

The workflow includes:

- Data Cleaning & Preprocessing
- Text Vectorization using TF-IDF
- Model Training
- Model Evaluation
- Performance Comparison
- Predictions on Unseen Messages

---

## 📂 Dataset

- **Dataset Name:** `spam.csv`
- **Total Records:** Approximately 5,500 messages

### Dataset Columns

| Column | Description |
|----------|-------------|
| Category | Target label (Spam or Ham) |
| Message | Email/SMS text |

---

## 🛠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

---

## 📚 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
```

---

# 📖 Project Workflow

## 1️⃣ Data Cleaning & Preprocessing

The following preprocessing steps were performed:

- Converted text to lowercase
- Removed punctuation
- Removed numbers
- Removed special characters
- Tokenized text
- Removed stopwords using NLTK
- Applied Porter Stemmer
- Rejoined cleaned words into sentences

---

## 2️⃣ Feature Engineering

Text data was converted into numerical vectors using **TF-IDF Vectorizer**.

### TF-IDF Parameters

- lowercase = True
- ngram_range = (1,2)
- min_df = 1

---

## 3️⃣ Machine Learning Models

The following models were trained and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

---

# 📊 Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Multinomial Naive Bayes | 97.1% | 96.2% | 90.5% | 93.3% |
| Logistic Regression | 98.2% | 97.7% | 93.4% | 95.5% |
| **Linear SVM** ⭐ | **98.4%** | **98.1%** | **93.4%** | **95.7%** |

---

# 🏆 Best Model

**Linear Support Vector Machine (SVM)** achieved the highest overall performance.

**Accuracy:** 98.4%

**F1-Score:** 95.7%

---

# 🔍 Sample Predictions

| Message | Prediction |
|----------|------------|
| Congratulations! You have won a $1000 Walmart gift card. Click here... | Spam |
| Are we still meeting at 3pm today? | Ham |
| URGENT: Your account is locked. Verify your details immediately. | Spam |
| Ok, see you soon! | Ham |
| WINNER! Free vacation to Bahamas... | Spam |

---

# 📊 Visualizations

The project includes the following visualizations:

- Spam vs Ham Class Distribution
- Confusion Matrix Heatmap
- Model Performance Comparison

> Screenshots of these visualizations are available in the **images/** folder.

---

# 📁 Project Structure

```
Email-Spam-Classification/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── Email_Spam_Classification.ipynb
│
├── images/
│   ├── spam_vs_ham.png
│   └── confusion_matrix.png
│
├── reports/
│   └── Email_Spam_Classification_Report.pdf
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 🚀 How to Run

### Clone the Repository

```bash
git clone https://github.com/yourusername/Email-Spam-Classification.git
```

### Navigate to the Project

```bash
cd Email-Spam-Classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
Email_Spam_Classification.ipynb
```

and run all cells.

---

# 💡 Future Improvements

- Train on larger datasets
- Use Lemmatization instead of Stemming
- Experiment with Word Embeddings
- Apply Deep Learning models (LSTM, GRU)
- Fine-tune Transformer models such as BERT

---

# 👨‍💻 Author

**Aleem Shoukat**

Data Science Intern – Digital Empowerment Network (DEN)

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Data Analytics

---

# 📜 Internship Information

**Organization:** Digital Empowerment Network (DEN)

**Internship:** Data Science Internship

**Task:** Task 03 – Email Spam Classification

---

## ⭐ If you found this project helpful, consider giving it a star!
