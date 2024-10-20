
# Spam Detection using SVM

This project implements a machine learning pipeline to detect spam messages using Support Vector Machines (SVM). The model leverages techniques such as TF-IDF Vectorization and SMOTE for handling class imbalance to improve prediction accuracy.



## Table of Contents

- Introduction
- Dataset
- Installation
- Project Workflow
- Results
- Future Work
- Contributing
- License

# Introduction

Spam detection is an important task in natural language processing (NLP) to filter out unwanted messages. This project demonstrates how to build a spam classifier using SVM and various preprocessing techniques to improve model performance.

# Dataset

The dataset used in this project is a CSV file named spam.csv, which contains two main columns:

- label: Indicates whether the message is "spam" or "ham" (not spam).
- message: The actual text message.
Ensure that the dataset file is uploaded before running the code.

# Installation

#### Clone this repository

```bash
  git clone https://github.com/NateChris14/SpamClassification-using-TFIDF-Vectorizer.git

```
#### Navigate to project directory

```bash
  cd SpamClassification-using-TFIDF-Vectorizer

```

#### Install the required dependencies

```bash
pip install pandas scikit-learn imbalanced-learn seaborn numpy

```
# Project Workflow

- Importing Libraries: Import necessary Python libraries for data manipulation, model training, and evaluation.
- Loading the Dataset: Load and preprocess the dataset to prepare it for training.
- Data Preparation: Keep only the relevant columns (label and message),Visualize and analyze class distribution
- Splitting Data: Perform a stratified train-test split to maintain the balance of classes across training and testing datasets.
- Text Vectorization: Convert text data to numerical format using TF-IDF Vectorizer.
- Handling Class Imbalance: Use SMOTE to balance the dataset by generating synthetic samples of the minority class.
- Training the Model: Train an SVM model on the balanced dataset.
- Evaluating the Model: Measure model performance using classification reports and confusion matrices.

## Example Code Snippet


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_df=0.2)
X_train_count = tfidf_vectorizer.fit_transform(X_train)
X_test_count = tfidf_vectorizer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_count, y_train)

# Train the SVM Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_smote, y_train_smote)

# Evaluate the model
y_pred = svm_model.predict(X_test_count)
print("Classification Report:\n", classification_report(y_test, y_pred))

```
    
# Results

The model achieved the following performance metrics:


## Results

The model achieved the following performance metrics:

| Metric            | Precision | Recall  | F1-Score | Support |
|-------------------|-----------|---------|----------|---------|
| Ham               | 0.9901    | 0.9967  | 0.9934   | 1206    |
| Spam              | 0.9777    | 0.9358  | 0.9563   | 187     |
| **Accuracy**      |           |         | **0.9885**| 1393   |
| **Macro Avg**     | 0.9839    | 0.9663  | 0.9748   | 1393    |
| **Weighted Avg**  | 0.9884    | 0.9885  | 0.9884   | 1393    |

### Confusion matrix

```lua
[[1202    4]
 [  12  175]]

```
# Future Work

- Experiment with different machine learning models (e.g., Random Forest, Logistic Regression).
- Explore more advanced NLP techniques, such as word embeddings.
- Enhance preprocessing by adding more feature engineering.

# Contributing

Contributions are welcome! Please open an issue or submit a pull request to improve the project.