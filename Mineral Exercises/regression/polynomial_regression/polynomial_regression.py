import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv(
    '/Users/moebooka/Learning/machine-learning/Mineral Exercises/datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

for degree in range(1, 7):
    # polynomial regression
    poly_ft = PolynomialFeatures(degree=degree)
    X_poly = poly_ft.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    # visualising the Polynomial Regression results
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg_2.predict(poly_ft.fit_transform(X)), color='blue')
    plt.title(f'Truth or Bluff (Polynomial Regression) degree {degree}')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
