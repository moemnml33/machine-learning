# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# importing the dataset
iris = sns.load_dataset("iris")
X = iris.iloc[:, :-1]   # independent variable - type DataFrame
y = iris.iloc[:, -1]    # dependent vairable - type series
# print(X)
# print(y)

# encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# creating a classifier for each feature
# iterate through each feature (independent variable)
for feature in X.columns:
    print(f"\n****************************************\n\nTraining classifier using only the feature: {feature} \n")
    
    # Select only the current feature (reshape needed for single feature)
    X_single_feature = X[[feature]]
    
    # splitting the dataset into the training set and test set for the current feature
    X_train, X_test, y_train, y_test = train_test_split(X_single_feature, y, train_size=0.7, random_state=0)
    
    # training the model(s) on the training set
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    # predict a new result
    y_pred = classifier.predict(X_test)
    
    # confusion matrix, accuracy, precision and recall score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    
    print(f"confusion matrix for {feature}:\n{cm}\n")
    print(f"accuracy score for {feature}: \n{accuracy}\n")
    print(f"precision score for {feature}:\n{precision}\n")
    print(f"recall score for {feature}:\n{recall}\n")



