from mnist import MNIST
import numpy as np
from scipy import linalg
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
mndata = MNIST('./data/')
x_train, y_train = map(np.array, mndata.load_training())
x_test, y_test = map(np.array, mndata.load_testing())
x_train = torch.FloatTensor(x_train/255.0)
y_train = torch.tensor(y_train)
x_test = torch.FloatTensor(x_test/255.0)
y_test = torch.tensor(y_test)

def init_W():
    d = 784
    h = 64
    k = 10
    alpha1 = (1/d)**0.5
    alpha2 = (1/h)**0.5
    W_0 = np.random.uniform(-alpha1, alpha1,(h,d))
    W_1 = np.random.uniform(-alpha2, alpha2,(k,h))
    b_0 = np.zeros((h,1))
    b_1 = np.zeros((k,1))
    W_0 = torch.FloatTensor(W_0).requires_grad_(True)
    W_1 = torch.FloatTensor(W_1).requires_grad_(True)
    b_0 = torch.FloatTensor(b_0).requires_grad_(True)
    b_1 = torch.FloatTensor(b_1).requires_grad_(True)
    return W_0,W_1,b_0,b_1

def fit_gd_cross_entropy(x_train, y_train,x_test,y_test):
    W_0,W_1,b_0,b_1 = init_W()
    loss_prev = 0
    d_loss = 1
    loss = 1
    accuracy = 0
    ep = 0
    batchsize = 10000
    nbatch = int(x_train.shape[0]/batchsize)
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    traccuracy = 0
    taccuracy = 0
    while traccuracy <0.99:
        ep = ep+1
        accuracy_list = []
        for i in range(0, nbatch):
            x = x_train[i*batchsize:(i+1)*batchsize]
            y = y_train[i*batchsize:(i+1)*batchsize]
            optim = torch.optim.Adam([W_0,W_1,b_0,b_1], lr=1e-3)
            x_1 = torch.matmul(W_0, x.T) + b_0
            x_1 = torch.nn.functional.relu(x_1)
            y_hat = torch.matmul(W_1, x_1) + b_1
            #y_hat = torch.nn.functional.relu(y_hat)
            y_hat = y_hat.T
            loss = torch.nn.functional.cross_entropy(y_hat, y.long())
            optim.zero_grad()
            loss.backward()
            accuracy = float(torch.sum(torch.argmax(y_hat, axis=1) == y)) / float(y.shape[0])
            #accuracy_list.append(accuracy)
            optim.step()
        xtr_1 = torch.matmul(W_0, x_train.T) + b_0
        xtr_1 = torch.nn.functional.relu(xtr_1)
        ytr_hat = torch.matmul(W_1, xtr_1) + b_1
        #yt_hat = torch.nn.functional.relu(yt_hat)
        ytr_hat = ytr_hat.T
        train_loss = torch.nn.functional.cross_entropy(ytr_hat, y_train.long())
        traccuracy = float(torch.sum(torch.argmax(ytr_hat, axis=1) == y_train)) / float(y_train.shape[0])
        xt_1 = torch.matmul(W_0, x_test.T) + b_0
        xt_1 = torch.nn.functional.relu(xt_1)
        yt_hat = torch.matmul(W_1, xt_1) + b_1
        #yt_hat = torch.nn.functional.relu(yt_hat)
        yt_hat = yt_hat.T
        test_loss = torch.nn.functional.cross_entropy(yt_hat, y_test.long())
        taccuracy = float(torch.sum(torch.argmax(yt_hat, axis=1) == y_test)) / float(y_test.shape[0])
        train_accuracy_list.append(traccuracy)
        test_accuracy_list.append(taccuracy)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        #print("epoch {}, accurracy on training dataset{}, accuracy on test dataset{} ".format(ep,traccuracy,taccuracy))
    print("Finally epoch {}, accurracy on training dataset {}, accuracy on test dataset {}; loss on training {} loss on test{} ".format(ep,traccuracy,taccuracy,train_loss,test_loss),flush = True)
    return(train_accuracy_list, test_accuracy_list,ep,train_loss_list,test_loss_list)

def plot(train_loss_list,ep,plot_name):
    plt.clf()
    sns.set()
    plt.plot(range(1,ep+1),train_loss_list,'-',label ='train loss')
    #plt.plot(range(1,ep+1),test_loss_list,'-',label ='test loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylim(0,1)
    plt.savefig('./'+plot_name)

def init_W2():
    d = 784
    h0 = 32
    h1 = 32
    k = 10
    alpha1 = (1/d)**0.5
    alpha2 = (1/h0)**0.5
    alpha3 = (1/h1)**0.5
    W_0 = np.random.uniform(-alpha1, alpha1,(h0,d))
    W_1 = np.random.uniform(-alpha2, alpha2,(h1,h0))
    W_2 = np.random.uniform(-alpha3, alpha3,(k,h1))
    b_0 = np.zeros((h0,1))
    b_1 = np.zeros((h1,1))
    b_2 = np.zeros((k,1))
    W_0 = torch.FloatTensor(W_0).requires_grad_(True)
    W_1 = torch.FloatTensor(W_1).requires_grad_(True)
    W_2 = torch.FloatTensor(W_2).requires_grad_(True)
    b_2 = torch.FloatTensor(b_2).requires_grad_(True)
    b_0 = torch.FloatTensor(b_0).requires_grad_(True)
    b_1 = torch.FloatTensor(b_1).requires_grad_(True)

    return W_0,W_1,W_2,b_0,b_1,b_2

def fit_gd_cross_entropy2(x_train, y_train,x_test,y_test):
    W_0,W_1,W_2,b_0,b_1,b_2 = init_W2()
    loss_prev = 0
    d_loss = 1
    loss = 1
    accuracy = 0
    ep = 0
    batchsize = 10000
    nbatch = int(x_train.shape[0]/batchsize)
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    traccuracy = 0
    taccuracy = 0
    while traccuracy <0.99:
        ep = ep+1
        accuracy_list = []
        for i in range(0, nbatch):
            x = x_train[i*batchsize:(i+1)*batchsize]
            y = y_train[i*batchsize:(i+1)*batchsize]
            optim = torch.optim.Adam([W_0,W_1,b_0,b_1], lr=1e-3)
            x_1 = torch.matmul(W_0, x.T) + b_0
            x_1 = torch.nn.functional.relu(x_1)
            x_2 = torch.matmul(W_1, x_1) + b_1
            x_2 = torch.nn.functional.relu(x_2)
            y_hat = torch.matmul(W_2, x_2) + b_2
            #y_hat = torch.nn.functional.relu(y_hat)
            y_hat = y_hat.T
            loss = torch.nn.functional.cross_entropy(y_hat, y.long())
            optim.zero_grad()
            loss.backward()
            accuracy = float(torch.sum(torch.argmax(y_hat, axis=1) == y)) / float(y.shape[0])
            #accuracy_list.append(accuracy)
            optim.step()
        xtr_1 = torch.matmul(W_0, x_train.T) + b_0
        xtr_1 = torch.nn.functional.relu(xtr_1)
        xtr_2 = torch.matmul(W_1, xtr_1) + b_1
        xtr_2 = torch.nn.functional.relu(xtr_2)
        ytr_hat = torch.matmul(W_2, xtr_2) + b_2
        #yt_hat = torch.nn.functional.relu(yt_hat)
        ytr_hat = ytr_hat.T
        train_loss = torch.nn.functional.cross_entropy(ytr_hat, y_train.long())
        traccuracy = float(torch.sum(torch.argmax(ytr_hat, axis=1) == y_train)) / float(y_train.shape[0])
        xt_1 = torch.matmul(W_0, x_test.T) + b_0
        xt_1 = torch.nn.functional.relu(xt_1)
        xt_2 = torch.matmul(W_1, xt_1) + b_1
        xt_2 = torch.nn.functional.relu(xt_2)
        yt_hat = torch.matmul(W_2, xt_2) + b_2
        #yt_hat = torch.nn.functional.relu(yt_hat)
        yt_hat = yt_hat.T
        test_loss = torch.nn.functional.cross_entropy(yt_hat, y_test.long())
        taccuracy = float(torch.sum(torch.argmax(yt_hat, axis=1) == y_test)) / float(y_test.shape[0])
        train_accuracy_list.append(traccuracy)
        test_accuracy_list.append(taccuracy)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print("epoch {}, accurracy on training dataset{}, accuracy on test dataset{},loss on training {} loss on test{} ".format(ep,traccuracy,taccuracy,train_loss,test_loss),flush = True)
    print("Finally epoch {}, accurracy on training dataset{}, accuracy on test dataset{},loss on training {} loss on test{} ".format(ep,traccuracy,taccuracy,train_loss,test_loss),flush = True)
    return(train_accuracy_list, test_accuracy_list,ep,train_loss_list,test_loss_list)

train_accuracy_list, test_accuracy_list,ep,train_loss_list,test_loss_list = fit_gd_cross_entropy(x_train, y_train,x_test,y_test)
plot(train_loss_list,ep,'a51.png')
train_accuracy_list, test_accuracy_list,ep,train_loss_list,test_loss_list = fit_gd_cross_entropy2(x_train, y_train,x_test,y_test)
plot(train_loss_list,ep,'a52.png')
