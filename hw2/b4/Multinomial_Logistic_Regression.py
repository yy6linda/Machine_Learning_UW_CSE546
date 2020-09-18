import torch
import numpy as np
from scipy import linalg
from mnist import MNIST
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test


def label_transormer(labels, type='one_hot'):
    if type == 'one_hot':
        k = labels.max() + 1
        return torch.tensor([np.eye(k)[i] for i in labels])
    else:
        return torch.from_numpy(labels)


def multinomial_logistic_regression(x_train, y_train, eta=0.1, epsilon=5e-4, type='mse'):
    if type not in {'softmax', 'mse'}:
        return 'Please specify right loss function type.'
    w = torch.zeros(784, 10, requires_grad=True)
    prev_loss = np.inf
    loss = 0
    while True:
        y_hat = torch.matmul(x_train.float(), w)
        # cross entropy combines softmax calculation with NLLLoss
        if type == 'softmax':
            loss = torch.nn.functional.cross_entropy(y_hat, y_train.long())
        if type == 'mse':
            loss = 0.5 * torch.nn.functional.mse_loss(y_hat, Y_train.float())
        if abs(prev_loss - loss) < epsilon:
            break
        print(loss)
        # computes derivatives of the loss with respect to W
        loss.backward()
        # gradient descent update
        w.data = w.data - eta * w.grad
        # .backward() accumulates gradients into W.grad instead
        # of overwriting, so we need to zero out the weights
        w.grad.zero_()
        prev_loss = loss
    return w.data


X_train, labels_train, X_test, labels_test = load_dataset()
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

Y_train = label_transormer(labels_train, type='raw')
Y_test = label_transormer(labels_test, type='raw')
w_softmax = multinomial_logistic_regression(X_train, Y_train, type='softmax')
train_acc_softmax = float(torch.sum(torch.argmax(torch.matmul(X_train.float(), w_softmax), axis=1) == Y_train)) / float(X_train.shape[0])
test_acc_softmax = float(torch.sum(torch.argmax(torch.matmul(X_test.float(), w_softmax), axis=1) == Y_test)) / float(X_test.shape[0])

Y_train = label_transormer(labels_train, type='one_hot')
w_mse = multinomial_logistic_regression(X_train, Y_train, epsilon=1e-5, type='mse')
Y_train = label_transormer(labels_train, type='raw')
train_acc_mse = float(torch.sum(torch.argmax(torch.matmul(X_train.float(), w_mse), axis=1) == Y_train)) / float(X_train.shape[0])
test_acc_mse = float(torch.sum(torch.argmax(torch.matmul(X_test.float(), w_mse), axis=1) == Y_test)) / float(X_test.shape[0])
