# Gaussian Naive Bayes Classifier on "Play Tennis" Dataset

## Overview

This repository demonstrates the implementation of the **Gaussian Naive Bayes** classification algorithm on a classical machine learning dataset—**"Play Tennis"**. The model is designed to predict whether a game of tennis should be played based on weather conditions such as Outlook, Temperature, Humidity, and Wind.

---

## Algorithm Explanation

**Naive Bayes** is a family of probabilistic classifiers grounded in **Bayes’ Theorem**, with the simplifying (naive) assumption that features are conditionally independent given the class label. The **Gaussian variant** is used for continuous data and assumes features follow a normal distribution.

> **Bayes' Theorem**:  
> P(A|B) = (P(B|A) * P(A)) / P(B)

### Applications

Naive Bayes classifiers are widely used in:

- Email spam detection
- Sentiment analysis
- Medical diagnosis
- Document classification
- Recommendation systems

### 🎯 Goal

To accurately classify new instances into their respective categories by computing class probabilities and selecting the class with the highest posterior probability.

---

## Problem Statement

The task is to predict the value of the target variable `PlayTennis` (Yes/No) based on four categorical features:

| Feature     | Description                          |
|-------------|--------------------------------------|
| Outlook     | Sunny, Overcast, Rain                |
| Temperature | Hot, Mild, Cool                      |
| Humidity    | High, Normal                         |
| Wind        | Weak, Strong                         |

The goal is to develop a classifier that can recommend whether to play tennis under specific weather conditions.

---

## Implementation Workflow

### Data Preprocessing

- All categorical features are encoded using **One-Hot Encoding** via `pandas.get_dummies()`.
- The resulting dataset is fully numerical and suitable for input to the `GaussianNB` model.

### Model Training and Prediction

- A `GaussianNB` model is instantiated using Scikit-learn.
- The model is trained on the entire preprocessed dataset.
- Predictions are made on the same dataset to evaluate performance.

---

## Results

### Predicted Output

The model outputs binary predictions (`Yes` or `No`) for each observation in the dataset, representing whether tennis should be played.

### Accuracy

- **Accuracy Score:** `1.00` (100%)
- Computed using `sklearn.metrics.accuracy_score`

### Interpretation

- A 100% accuracy indicates perfect alignment between predicted and actual labels in the training dataset.
- However, **evaluating on training data only can cause overfitting**. For robust performance assessment, a train-test split or cross-validation is recommended.

---

## Key Takeaways

- Naive Bayes, despite its simplicity, can provide strong baseline models.
- The GaussianNB variant is suitable when features are continuous or transformed to numeric via encoding.
- High performance on small datasets may not reflect true generalization capacity. Evaluation on unseen data is critical.

---

## Recommendations

| Step                    | Purpose                                |
|-------------------------|----------------------------------------|
| Train-Test Split        | Assess model generalization            |
| Cross-Validation        | Improve result reliability             |
| Compare with Other Models | Benchmark performance                |
| Hyperparameter Tuning   | Optional smoothing enhancements        |

---

## Conclusion

This project demonstrates the use of **Gaussian Naive Bayes** for a binary classification task using a categorical dataset. While the model achieves **perfect accuracy** on training data, real-world usage would require validation on independent data to ensure robustness. Naive Bayes remains a valuable tool in any data scientist’s toolkit for its **efficiency, interpretability, and solid baseline performance**.

---

## Tools & Libraries

- Python
- Pandas
- Scikit-learn

---

## 📅 Author & Date

**Author:** _Keshav Sahani_  
**Date:** _[January 10'2024]_


