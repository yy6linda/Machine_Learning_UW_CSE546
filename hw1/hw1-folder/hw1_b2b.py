from mnist import MNIST
import numpy as np
from scipy import linalg
import random

mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

y = np.zeros([len(X_train),10])
for i in range(0, len(X_train)):
    y[i, labels_train[i]] = 1
    print(i)


def train(X_train, y, reg):
    n,d = X_train.shape
    a = X_train.T.dot(X_train) + np.diagflat(np.ones(d) * reg)
    b = X_train.T.dot(y)
    W_hat = linalg.solve(a, b, assume_a='sym') # a * w = b
    #W_hat1 = np.linalg.pinv(a).dot(b)
    return W_hat

def predict(W_hat, X_test):
    return np.argmax(W_hat.T.dot(X_test.T), axis=0) # max by column

p = 5000
n,d = X_train.shape
random.seed(1)
G = np.random.normal(0,np.sqrt(0.1),size=(p,d))
b = np.random.rand(p,1)*2*np.pi
#transformation function
def h(X_train, G, b):
    return np.cos(np.dot(G,X_train.T)+b).T

X_train = h(X_train, G, b)
X_test = h(X_test, G, b)

#CV
random.seed(1)
group_index = np.floor(np.random.uniform(0,5,len(X_train)))
# add group index into train and y
X_train = np.c_[X_train, group_index]
y = np.c_[y, group_index]
labels_train = np.c_[labels_train, group_index]
Train_Error_cv = np.zeros(5)
Validation_Error_cv = np.zeros(5)
reg = 1E-4

Test_Error_cv = np.zeros(5)
for i in range(0,5):
    print(p)
    # only select i from the last column
    # X_train[X_train[:, np.shape(X_train)[1] - 1] == 2, np.shape(X_train)[1] - 1]
    X_train_new = X_train[X_train[:, np.shape(X_train)[1] - 1] != i, 0: np.shape(X_train)[1] - 1]
    X_train_validation = X_train[X_train[:, np.shape(X_train)[1] - 1] == i, 0: np.shape(X_train)[1] - 1]
    # y
    y_new = y[y[:, np.shape(y)[1] - 1] != i, 0: np.shape(y)[1] - 1]
    y_validation = y[y[:, np.shape(y)[1] - 1] == i, 0: np.shape(y)[1] - 1]
    # labels_train
    labels_train_new = labels_train[labels_train[:, np.shape(labels_train)[1] - 1] != i, 0: np.shape(labels_train)[1] - 1]
    labels_train_validation = labels_train[labels_train[:, np.shape(labels_train)[1] - 1] == i, 0: np.shape(labels_train)[1] - 1]

    W_hat = train(X_train_new, y_new, reg)
    np.unique(predict(W_hat, X_train_new))
    Error_train = np.sum(predict(W_hat, X_train_new) != labels_train_new.T[0]) / len(X_train_new)
    print("Train Error: " + str(Error_train))
    Train_Error_cv[i] = Error_train

    y_predict = predict(W_hat, X_train_validation)
    np.unique(y_predict)
    Error_validation = np.sum(y_predict != labels_train_validation.T[0]) / len(X_train_validation)
    print("Validation Error: " + str(Error_validation))
    Validation_Error_cv[i] = Error_validation

    y_test = predict(W_hat, X_test)
    Error_test = np.sum(y_test != labels_test) / len(X_test)
    print("Test Error: " + str(Error_test))
    Test_Error_cv[i] = Error_test

Test_Error_cv.mean()
