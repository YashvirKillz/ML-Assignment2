# Breast Cancer Classification - ML Assignment 2

## a. Problem Statement
In this project, I have used machine learning to find out if a breast tumor is cancerous (malignant) or not (benign). Since there are only two possible answers, this is a binary classification problem. I have trained 5 different models and compared which one works best.

## b. Dataset Description
- **Dataset:** Breast Cancer Wisconsin Dataset (from UCI / scikit-learn)
- **Number of rows (instances):** 569
- **Number of features (columns):** 30 (like radius, texture, perimeter, area, etc.)
- **Target:** 0 = malignant, 1 = benign
- **Train/Test Split:** 80% for training, 20% for testing

## c. GitHub Repository Link
https://github.com/YashvirKillz/ML-Assignment2

## d. Models Used - Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| kNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Random Forest | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |

## Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | This gave the best results with the highest accuracy, F1 and MCC. The data can be separated with a straight line easily, so this simple model worked really well. |
| Decision Tree | Its AUC was the lowest (0.944). A single tree tends to memorize the training data too much, so it is not as strong. |
| kNN | It got the same accuracy as Decision Tree, but a better AUC (0.982). Scaling the data helped this model. |
| Naive Bayes | A simple model but it worked very well, with a very high AUC (0.997). |
| Random Forest | It gave good and stable results, similar to Naive Bayes. It is better than a single Decision Tree because it uses many trees together. |
| **Overall Winner** | **Logistic Regression** - it had the highest accuracy (0.9737), F1 (0.9790) and MCC (0.9439). |

## How to Run
1. Download the project files
2. Install the libraries: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Upload the `test_data.csv` file in the app to see the results of each model.
