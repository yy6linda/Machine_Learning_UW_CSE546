import torch
from mnist import MNIST
import numpy as np
from scipy import linalg
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
mndata = MNIST('./data/')
x_train, y_train = map(np.array, mndata.load_training())
x_test, y_test = map(np.array, mndata.load_testing())
x_train = torch.FloatTensor(x_train/255.0)
y_train = torch.tensor(y_train)
x_test = torch.FloatTensor(x_test/255.0)
y_test = torch.tensor(y_test)

def one_hot(labels):
    k = labels.max() + 1
    return torch.tensor([np.eye(k)[i] for i in labels])

def cal_accuracy(W,x,y):
    y_hat = torch.matmul(x, W)
    y_hat_label = torch.argmax(y_hat, axis=1)
    acc = float(torch.sum(y_hat_label == y)) / float(x.shape[0])
    return acc

# softmax
def fit_gd_cross_entropy(x_train,y_train,x_test,y_test,eta=0.1, delta = 1e-4):
    W = torch.zeros(784, 10, requires_grad=True)
    converged = False
    loss_prev = 0
    d_loss = 1
    loss = 1
    train_accuracy = []
    test_accuracy = []
    i = 0
    while d_loss > delta:
        i = i+1
        y_hat = torch.matmul(x_train, W)
        loss = torch.nn.functional.cross_entropy(y_hat, y_train.long())
        d_loss = abs(loss -loss_prev)
        loss_prev = loss
        loss.backward()
        W.data = W.data - eta * W.grad
        train_accuracy.append(cal_accuracy(W,x_train,y_train))
        test_accuracy.append(cal_accuracy(W,x_test,y_test))
        W.grad.zero_()

    return W,train_accuracy,test_accuracy,i
        #print(loss.data)
W,train_accuracy,test_accuracy,i = fit_gd_cross_entropy(x_train,y_train,x_test,y_test,eta=0.05, delta=0.001)
fig, ax = plt.subplots(figsize=(16,9))
sns.lineplot(x=range(1,i+1), y=train_accuracy, ax=ax, label="Training data")
sns.lineplot(x=range(1,i+1), y=test_accuracy, ax=ax, label="Test data")
plt.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
plt.savefig("B4c1.png")
