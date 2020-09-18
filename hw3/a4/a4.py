from mnist import MNIST
import numpy as np
from scipy import linalg
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

mndata = MNIST('./data/')
x_train, y_train = map(np.array, mndata.load_training())
x_test, y_test = map(np.array, mndata.load_testing())
x_train = x_train/255.0
x_test = x_test/255.0
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

# clustered_dict = {0:[(np.array(784,),np.array(label)),...],1:(np.array(784,),np.array(label)),}
def init_center(k,x):
    init_c = random.choices(x_train,k=k)
    center_dict = dict(zip(list(range(0,k)),init_c))
    return  center_dict

def form_cluster(labeled_data,center_dict):
    cluster_dict = dict()
    for i in labeled_data:
        x = i[0]
        min_distance = 1000
        label = 100
        for j in center_dict.keys():
            c = center_dict[j]
            distance = np.linalg.norm(x - c)
            if (distance < min_distance):
                min_distance = distance
                label = j
        if label in  cluster_dict.keys():
            cluster_dict[label].append(i)
        else:
            cluster_dict[label] = [i]
    return cluster_dict

#cluster_list [(np.array(784,),np.array(label)),(,),(,)]
def sumCluster(cluster_list):
    sumc = cluster_list[0][0].copy()
    for i in range(1,len(cluster_list)):
        sumc = sumc + cluster_list[i][0]
    return sumc

def mean(cluster_list):
    return  sumCluster(cluster_list)/len(cluster_list)

# center_dict{1:np.array(mean_x)}
def new_center(cluster_dict):
    center_dict = dict()
    for i in cluster_dict.keys():
        center_dict[i] = mean(cluster_dict[i])
    return(center_dict)

def distance_change(center_old_dict,center_dict):
    distance_total = 0
    for i in center_old_dict.keys():
        distance = np.linalg.norm(center_old_dict[i] - center_dict[i])
        distance_total = distance_total + distance
    return distance_total

def cal_objective(converged_cluster_dict,center_dict):
    err = 0
    total = 0
    for i in center_dict.keys():
        #total = total + len(converged_cluster_dict[i])
        c = center_dict[i]
        for j in converged_cluster_dict[i]:
            x = j[0]
            err = err + np.linalg.norm(x - c)**2
    #err = err/total
    return err

def cal_err(converged_cluster_dict,center_dict):
    err = 0
    total = 0
    for i in center_dict.keys():
        total = total + len(converged_cluster_dict[i])
        c = center_dict[i]
        for j in converged_cluster_dict[i]:
            x = j[0]
            err = err + np.linalg.norm(x - c)**2
    err = err/total
    return err

def training(labeled_data,k):
    center_dict = init_center(k,labeled_data)
    cluster_dict = form_cluster(labeled_data,center_dict)
    distance = 10
    iter = 0
    obj_list = []
    while distance != 0:
        print(iter)
        old_center = center_dict
        center_dict = new_center(cluster_dict)
        cluster_dict = form_cluster(labeled_data,center_dict)
        obj = cal_objective(cluster_dict,center_dict)
        distance = distance_change(old_center,center_dict)
        iter = iter+1
        print(obj)
        obj_list.append(obj)
    return(center_dict,cluster_dict,obj_list,iter,k)

def a4b(err_list,iter):
    plt.clf()
    plt.plot(range(1,iter+1),err_list,'-')
    plt.xlabel('Iteration')
    plt.ylabel('Objective function')
    plt.savefig('./a4b.png',bbox_inches='tight')

def plot_mnist(center_dict,k):
    fig = plt.figure()
    for i in range(0,k):
        cur_ax = fig.add_subplot(2, 5, i+1)
        sns.heatmap(center_dict[i].reshape((28,28)), cbar=False, ax=cur_ax)
        cur_ax.get_xaxis().set_visible(False)
        cur_ax.get_yaxis().set_visible(False)
        plt.savefig("./a4b2.png")

def a4c(labeled_data_train,labeled_data_test,k_list):
    test_err_list = []
    train_err_list = []
    for k in k_list:
        center_dict,cluster_dict,err_list,iter,k = training(labeled_data_train,k)
        test_cluster_dict = form_cluster(labeled_data_test,center_dict)
        test_err = cal_err(test_cluster_dict,center_dict)
        train_err = cal_err(cluster_dict,center_dict)
        train_err_list.append(train_err)
        test_err_list.append(test_err)
    plt.clf()
    sns.set()
    plt.plot(k_list,train_err_list,'-',label = 'training error')
    plt.plot(k_list,test_err_list,'-',label = 'test error')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.savefig('./a4c.png',bbox_inches='tight')

# labeled_data is a list [(np.array(784,),np.array(label)),(),()]
labeled_data_train = []
labeled_data_test = []
for i in range(0,x_train.shape[0]):
    labeled_data_train.append((x_train[i],y_train[i]))
for i in range(0,x_test.shape[0]):
    labeled_data_test.append((x_test[i],y_test[i]))

center_dict,cluster_dict,obj_list,iter,k = training(labeled_data_train,10)
a4b(obj_list,iter)
k_list = [2,4,8,16,32,64]
a4c(labeled_data_train,labeled_data_test,k_list)
plot_mnist(center_dict,10)
