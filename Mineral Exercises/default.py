import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR

iris = sns.load_dataset('iris')

print(iris.head(5))

X = iris[['petal_length']]
Y = iris[['petal_width']]
# plt.scatter(X, Y)
# plt.show()

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

model = LinearRegression()
# model = SVR(kernel='linear')
# model = RandomForestRegressor()
print(cross_val_score(model, X, Y, cv=3))

model.fit(X, Y)
Y_pred = model.predict(X)
# ypred = model.predict(X_test)
# R2 = r2_score(Y_test, ypred)
# print(model.score(X_test, Y_test))

plt.scatter(X, Y, color='red')
# plt.scatter(X_test, Y_test, color='blue')
plt.plot(X, Y_pred, color='black')
plt.show()