 
# importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR

# importing the dataset
iris = sns.load_dataset('iris')
print(iris.head(5))
X = iris[['petal_length']]
y = iris[['petal_width']]
plt.scatter(X, y)
plt.show()
 
#  Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#  Training the model on the Training set
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'black')
plt.title('length vs width (training set)')
plt.xlabel('length')
plt.ylabel('width')
plt.show()

# visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('length vs width (test set)')
plt.xlabel('length')
plt.ylabel('width')
plt.show()
