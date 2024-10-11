from turtle import color
from matplotlib import axis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# import the dataset
dataset = pd.read_csv(
    "/Users/moebooka/Learning/machine-learning/Mineral Exercises/support_vector_regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# before scaling
print("BEFORE SCALING: ")
print(f"X: {X}")
print(f"y: {y}")

# feature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# reshape y to expected format
y = y.reshape(len(y), 1)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
# after scaling
print("AFTER SCALING: ")
print(f"X: {X}")
print(f"y: {y}")

# training model
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predict single value
single_pred = sc_y.inverse_transform(regressor.predict(
    sc_X.transform([[6.5]])).reshape(-1, 1))
print(f"y_pred: {single_pred}")

# visualize
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(
    regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
