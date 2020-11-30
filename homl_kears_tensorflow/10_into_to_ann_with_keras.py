import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow import keras

iris = load_iris()
X = iris.data[:, (2, 3)]    # Petal length, petal width
y = (iris.target == 0).astype(np.int)    # Iris setosa

perceptron_clf = Perceptron()
perceptron_clf.fit(X, y)

y_pred = perceptron_clf.predict([[2, 0.5]])
print(y_pred)


# Fashion MNIST
fashion_mnsit = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnsit.load_data()
print(X_train_full.shape)