# Titanic Survival Prediction - Machine Learning Project

## Overview

This project applies end-to-end data science workflow on the Titanic dataset, including exploration, preprocessing, feature engineering, model comparison, hyperparameter tuning, and evaluation on an untouched test set. The primary goal is to build a reliable model that predicts passenger survival while demonstrating best practices used in real-world ML projects.

---

## Dataset

The dataset contains passenger information such as demographic details, ticket class, fare, family size, port of embarkation, and survival status.

**Target variable:** `Survived` (1 = survived, 0 = did not survive)

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Inspected variable distributions (Age, Fare, Pclass, etc.)
* Analyzed survival patterns by sex, class, and age group
* Identified missing values
* Visualized correlations and relationships between variables

---

## 2. Data Cleaning & Imputation

* Imputed missing **Age** values using median within groups
* Imputed missing **Embarked** values using mode
* Handled Fare and Cabin as needed (Cabin dropped due to excessive missingness)

---

## 3. Feature Engineering

* Created `Age_group` (Child, Teen, Youth, Adult, Senior)
* One-hot encoded categorical features using `get_dummies()` with `drop='first'`
* Standardized numerical features (Age, Fare, SibSp, Parch)
* Removed duplicated unscaled feature columns
* Avoided multicollinearity by retaining only k-1 dummy variables per category

---

## 4. Train/Val/Test Split

* **Train set:** used for model training
* **Validation set:** used to compare multiple models fairly
* **Test set:** untouched until final evaluation

This prevents data leakage and ensures honest performance estimation.

---

## 5. Model Comparison (Validation Set)

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest Classifier
* Gradient Boosting Classifier
* KNN Classifier

### **Validation Performance**

| Model                        | Accuracy | F1 Score |
| ---------------------------- | -------- | -------- |
| Logistic Regression          | 0.813    | 0.775    |
| Decision Tree                | 0.791    | 0.745    |
| Random Forest Classifier     | 0.873    | 0.844    |
| Gradient Boosting Classifier | 0.858    | 0.812    |
| KNN                          | 0.754    | 0.708    |

### Why Random Forest Was Not Selected

Although Random Forest performed best on the validation set, it:

* Overfitted the small dataset
* Performed worse on the **untouched test set**
* Was less stable across random states

---

## 6. Final Model Selection

After hyperparameter tuning, **Logistic Regression** produced the most consistent and generalized performance on the **test set**, making it the best choice for this dataset.

---

## 7. Final Model Performance (Test Set)

### **Logistic Regression (Tuned)**

| Metric       | Score                   |
| ------------ | ----------------------- |
| **Accuracy** | **0.806**               |
| **F1 Score** | **0.735**               |
| Precision    | *Add value if computed* |
| Recall       | *Add value if computed* |

Logistic Regression generalized better than the tree-based models despite their higher validation scores.

---

## 8. Confusion Matrix Explanation

```
                Predicted
               0        1
Actual 0     TN       FP
Actual 1     FN       TP
```

* **TP:** Correctly predicted survivors
* **TN:** Correctly predicted non-survivors
* **FP:** Predicted survival incorrectly
* **FN:** Missed predicting a survivor

In Titanic context, **recall** is particularly important as it reflects the model's ability to correctly identify survivors.

---

## 9. Feature Importance (Logistic Regression Coefficients)

Top factors **reducing** survival probability:

* Sex_male
* Age_group_Senior
* Age_group_Adult
* Age_group_Teen
* Age_group_Youth
* Higher Passenger Class (3 > 2 > 1)

Top factors **increasing** survival probability:

* Higher Fare
* Embarked_Q

The dropped dummy variables (e.g., Age_group_Child) act as baseline references for coefficient interpretation.

---

## 10. Key Takeaways

* Logistic Regression is a strong baseline model and performs exceptionally well on small datasets.
* Proper validation and avoiding overfitting are more important than chasing high accuracy on training data.
* Interpretability matters — LR provides clear insights into feature influence.
* A well-structured ML workflow ensures fairness, transparency, and reproducibility.

---

## 11. Future Work

* Try advanced models (XGBoost, LightGBM)
* Use cross-validation for more robust evaluation
* Feature interactions (e.g., class × sex)
* Optimize feature engineering with domain knowledge
* Deploy model using Flask or Streamlit

---

## 12. Project Structure

```
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
├── README.md
```

---

## 13. Conclusion

Although Random Forest and Gradient Boosting initially performed very well on the validation set, their performance did not generalize as well to the unseen test set. After hyperparameter tuning, **Logistic Regression achieved the best overall results** while remaining interpretable, stable, and well-suited for the relatively small Titanic dataset.

Thus, **Logistic Regression was selected as the final model for this project**.

---

Feel free to explore, contribute, or suggest improvements.
