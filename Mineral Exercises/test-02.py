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

# creating one classifier for all features
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)
print(f"X_train:\n{X_train}\n")
print(f"X_test:\n{X_test}\n")
print(f"y_train:\n{y_train}\n")
print(f"y_test:\n{y_test}\n")

# training the model(s) on the training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# predict a new result
y_pred = classifier.predict(X_test) # predict test set
print(f"y_pred:\n{y_pred}\n")

# confusion matrix, accuracy, precision and recall score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
print(f"confusion matrix:\n{cm}\n")
print(f"accuracy score:\n{accuracy}\n")
print(f"precision score:\n{precision}\n")
print(f"recall score:\n{recall}\n")

# plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
# Adding labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


