# Spam Detection using SVM

This project implements a machine learning pipeline to detect spam messages using Support Vector Machines (SVM). The model leverages techniques such as TF-IDF Vectorization and SMOTE for handling class imbalance to improve prediction accuracy.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Deployment](#deployment)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Spam detection is an essential task in natural language processing (NLP) to filter out unwanted messages. This project demonstrates how to build and deploy a spam classifier using SVM and various preprocessing techniques to improve model performance.

## Dataset

The dataset used in this project is a CSV file named `spam.csv`, which contains two main columns:
- `label`: Indicates whether the message is "spam" or "ham" (not spam).
- `message`: The actual text message.

Ensure that the dataset file is uploaded before running the code.

## Installation

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
pip install -r requirements.txt
```

## Project Workflow

1. **Import Libraries:** Import necessary Python libraries for data manipulation, model training, and evaluation.
2. **Load Dataset:** Load and preprocess the dataset to prepare it for training.
3. **Data Preparation:** Keep only the relevant columns (`label` and `message`). Visualize and analyze class distribution.
4. **Split Data:** Perform a stratified train-test split to maintain the balance of classes across training and testing datasets.
5. **Text Vectorization:** Convert text data to numerical format using TF-IDF Vectorizer.
6. **Handle Class Imbalance:** Use SMOTE to balance the dataset by generating synthetic samples of the minority class.
7. **Train Model:** Train an SVM model on the balanced dataset.
8. **Evaluate Model:** Measure model performance using classification reports and confusion matrices.

## Deployment

This project has been deployed using Flask and Heroku to make the spam detection model accessible as a web application.

### Steps to Deploy
1. **Create a Flask Application:**
   - Implement endpoints for predictions and integrate the trained SVM model.

2. **Procfile:** Include a `Procfile` to define the entry point for the Heroku app.

3. **requirements.txt:** Ensure all necessary dependencies are listed in the `requirements.txt` file.

4. **Heroku Deployment:**
   - Use the following commands to deploy:
     ```bash
     heroku create
     git push heroku main
     ```

5. **Access Application:**
   - Once deployed, access the application via the Heroku URL provided.

## Results

The model achieved the following performance metrics:

| Metric            | Precision | Recall  | F1-Score | Support |
|-------------------|-----------|---------|----------|---------|
| Ham               | 0.9901    | 0.9967  | 0.9934   | 1206    |
| Spam              | 0.9777    | 0.9358  | 0.9563   | 187     |
| **Accuracy**      |           |         | **0.9885**| 1393   |
| **Macro Avg**     | 0.9839    | 0.9663  | 0.9748   | 1393    |
| **Weighted Avg**  | 0.9884    | 0.9885  | 0.9884   | 1393    |

### Confusion Matrix
```plaintext
[[1202    4]
 [  12  175]]
```

## Future Work

- Experiment with different machine learning models (e.g., Random Forest, Logistic Regression).
- Explore more advanced NLP techniques, such as word embeddings.
- Enhance preprocessing by adding more feature engineering.
- Optimize the web application for scalability and performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

