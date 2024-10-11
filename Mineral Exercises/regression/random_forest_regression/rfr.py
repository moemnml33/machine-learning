from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    "/Users/moebooka/Learning/machine-learning/Mineral Exercises/datasets/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# number of trees
regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print(y_pred)

# the prediction is not good and plot wont look good cause the model isn't good for the given data
# we kenw that in advance, yet we still did it just for practicing purposes
