# Logistic Regression tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 25SEP20

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# Train the Logistic Regression model on the Training set


# Predict a new result


# Predict the Test set results


# Make the Confusion Matrix


# Visualize the Training set results


# Visualize the Test set results

