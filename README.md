# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, remove unnecessary columns, encode categorical variables, and split the data into training and testing sets. Apply feature scaling using training data statistics.
2. Initialize weights and bias to zero. Set learning rate and number of epochs.
3. For each epoch, compute the linear combination of inputs and weights, apply the sigmoid function, calculate gradients of weights and bias, and update them to minimize the loss.
4. Predict the class labels for test data using a threshold of 0.5 and evaluate the model using accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vinodhini M.k
RegisterNumber: 212225230305 
*/

```
```
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (1).csv")
data.drop("sl_no", axis=1, inplace=True)

# Encode target
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# ==============================
# 2. Features & Target
# ==============================
X = data.drop('status', axis=1).values
y = data['status'].values

# ==============================
# 3. Train–Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# 4. Feature Scaling
# ==============================
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test  = (X_test  - X_test.mean(axis=0))  / X_test.std(axis=0)

# ==============================
# 5. Sigmoid
# ==============================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ==============================
# 6. Initialize Parameters
# ==============================
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.1      # increased
epochs = 3000            # increased

# ==============================
# 7. Gradient Descent
# ==============================
for _ in range(epochs):
    linear = np.dot(X_train, weights) + bias
    y_pred = sigmoid(linear)

    dw = (1 / len(y_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1 / len(y_train)) * np.sum(y_pred - y_train)

    weights -= learning_rate * dw
    bias -= learning_rate * db

# ==============================
# 8. Prediction
# ==============================
def predict(X):
    linear = np.dot(X, weights) + bias
    return np.where(sigmoid(linear) >= 0.5, 1, 0)

y_predicted = predict(X_test)

# ==============================
# 9. Evaluation
# ==============================
print("Accuracy:", accuracy_score(y_test, y_predicted) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predicted))

print("\nClassification Report:")
print(classification_report(y_test, y_predicted))
```

## Output:
![alt text](<Screenshot 2026-02-06 111913.png>)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

