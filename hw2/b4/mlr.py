import torch
from mnist import MNIST
import numpy as np
from scipy import linalg
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class MultinomialMNIST:
    def __init__(self):
        # Load data
        mndata = MNIST('./data/')
        Xtr, ytr = map(np.array, mndata.load_training())
        # Xt, yt = map(np.array, mndata.load_testing())
        self.X_train = torch.FloatTensor(Xtr/255.0)
        self.y_train = torch.tensor(ytr/255.0)

        # self.X_test = torch.FloatTensor(Xt)
        # self.y_test = torch.tensor(yt)

    def fit_gd_cross_entropy(self, eta=0.1, delta=0.1, tmax=200):
        W = torch.zeros(784, 10, requires_grad=True)
        converged = False
        prev_loss = 0
        t = 0
        while not converged:
            y_hat = torch.matmul(self.X_train, W)
            # cross entropy combines softmax calculation with NLLLoss
            loss = torch.nn.functional.cross_entropy(y_hat, self.y_train.long())
            print(f"[t={t}] loss={loss.data}")
            # computes derivatives of the loss with respect to W
            loss.backward()
            # gradient descent update
            W.data = W.data - eta * W.grad
            # .backward() accumulates gradients into W.grad instead
            # of overwriting, so we need to zero out the weights
            W.grad.zero_()
            t += 1
            # converged = torch.abs(loss - prev_loss) < delta or t > tmax
            converged = t > tmax
            # train_error = self.error_rate(self.X_train, self.y_train, W)
            # print(f"[t={t}] train_error={train_error:.5f}, loss={loss:.5f}")
            # converged = True
            # prev_loss = loss


if __name__ == "__main__":
    mm = MultinomialMNIST()
    mm.fit_gd_cross_entropy()
