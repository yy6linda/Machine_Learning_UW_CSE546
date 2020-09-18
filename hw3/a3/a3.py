import numpy as np
from matplotlib import pyplot as plt
import scipy
import seaborn as sns

def generate_dataset(n):
    np.random.seed(23)
    sigma = 1
    x = np.random.uniform (0, 1, n)
    err = np.random.normal(0, sigma, n)
    f = 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)
    y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2) + err
    x_order = x.argsort()
    x = x[x_order[::1]]
    x = x.reshape(x.shape[0],1)
    f = f[x_order[::1]]
    f = f.reshape(f.shape[0],1)
    y = y[x_order[::1]]
    y = y.reshape(y.shape[0],1)
    return x,f,y

def train(K,y,L):
    # a = argmin(Ka-y)^2+La^TKa
    # using closed-form solution:
    # a = (K+LI)^(-1)y
    m = (K + L*np.identity(K.shape[0]))
    a = scipy.linalg.solve(m,y)
    return a

def poly(x,z,d):
    k = (1+x.T.dot(z))**d
    return k

def rbf(x,z,r):
    k = np.exp(-r*(x-z)**2)
    return k

def form_kernel(x,kernel,param,x2=None):
    if x2 is None:
        x2 = x
    n = x.shape[0]
    m = x2.shape[0]
    K = np.zeros((n,m))
    for i in range(0,n):
        for j in range(0,m):
            K[i,j] = kernel(x[i],x2[j],param)
    return K

def pred(K,a):
    y = K.dot(a)
    return y

def kfold(K,y,k=5):
    K_trains = []
    K_vals = []
    y_trains = []
    y_vals = []
    #idx = np.random.permutation(K.shape[0])
    idx = np.random.RandomState(seed=23).permutation(K.shape[0])
    part_size = int(K.shape[0]/k)
    for i in range(0,k):
        start = i * part_size
        end = (i + 1) * part_size
        val_idx = idx[start:end]
        train_idx = np.concatenate((idx[0:start],idx[end:]))
        K_trains.append(  K[train_idx, :][:,train_idx]  )
        K_vals.append(K[val_idx, :][:, train_idx])
        y_trains.append(y[train_idx, :])
        y_vals.append(y[val_idx, :])
    return K_trains, K_vals, y_trains, y_vals

def MSE(y,y_pred):
    mse = ((y_pred-y)**2).mean()
    return mse

def loo(x,y,param_list, L_list,kernel):
    results = []
    for L in L_list:
        for param in param_list:
            mse_list = []
            for i in range(x.shape[0]):
                #print(i)
                cx = np.concatenate((x[:i],x[(i+1):]))
                cy = np.concatenate((y[:i],y[(i+1):]))
                x_i = x[i]
                x_i = x_i.reshape(x_i.shape[0],1)
                y_i = y[i]
                y_i = y_i.reshape(y_i.shape[0],1)
                K = form_kernel(cx,kernel,param,x2=None)
                #print(L)
                a = train(K,cy,L)
                Kc = form_kernel(x_i,kernel,param,x2=cx)
                y_pred = pred(Kc,a)
                mse = MSE(y_i,y_pred)
                #print(y_i,y_pred)
                mse_list.append(mse)
            results.append( ( np.mean(mse_list), param, L ) )
    min_mse = 1000
    best_param = 0
    best_L = 0
    for i in results:
        mse_avg,param,L = i
        if mse_avg < min_mse:
            best_param = param
            best_L = L
            min_mse = mse_avg
    return best_param,best_L, min_mse

def cv(x,y,param_list, L_list,kernel,k):
    results = []
    for param in param_list:
        K = form_kernel(x,kernel,param,x2=None)
        K_trains, K_vals, y_trains, y_vals = kfold(K,y,k=k)
        for L in L_list:
            mse_list = []
            for i in range(k):
                a = train(K_trains[i],y_trains[i],L)
                y_pred = pred(K_vals[i],a)

                mse = MSE(y_vals[i],y_pred)
                mse_list.append(mse)
                #print("****np.mean(mse_list){}".format(mse))
            results.append( ( np.mean(mse_list), param, L ) )
            #print("mse{}".format(np.mean(mse_list)))
    min_mse = 10000
    best_param = 0
    best_L = 0
    #print (results)
    for i in results:
        mse_avg,param,L = i
        if mse_avg < min_mse:
            best_param = param
            best_L = L
            min_mse = mse_avg
    return best_param,best_L, min_mse

'''Q3 bootstrapping '''
def bootstrap(x, y, kernel, param, L, B=300):
    x_plot = np.arange(0,1,0.001)
    x_plot = x_plot.reshape(x_plot.shape[0], 1)
    n = x.shape[0]
    f_array = np.zeros((B, x_plot.shape[0]))
    for i in range(B):
        idxs = np.random.choice(n, n)
        x_sub = x[idxs,:]
        y_sub = y[idxs,:]
        K = form_kernel(x_sub,kernel,param,x2=None)
        a = train(K, y_sub, L=L)
        kx = form_kernel(x_plot,kernel,param,x2= x_sub)
        f = pred(kx, a)
        f_array[i] = f[:,0]
        #print(f_array[i,:])
    #f_array[np.isnan(f_array)] = 0
    p5 = np.percentile(f_array, 5, axis=0)
    p95 = np.percentile(f_array, 95, axis=0)
    return(x, p5, p95)

def plotb( x,f,y,kernel,best_param,best_L,fig_name):
    plt.clf()
    K_poly = form_kernel(x,kernel,best_param,x2=None)
    a = train(K_poly,y,best_L)
    x_plot = np.arange(0,1,0.001)
    x_plot =  x_plot.reshape(x_plot.shape[0],1)
    K_plot = form_kernel(x_plot,kernel,best_param,x2=x)
    y_pred = pred(K_plot,a)
    f_plot = 4*np.sin(np.pi*x_plot)*np.cos(6*np.pi*x_plot**2)
    #x, p5, p95 = bootstrap(x, y, kernel,best_param, best_L, B=300)
    sns.set()
    plt.plot(x,y,'.',label ='y')
    plt.plot(x_plot,f_plot,'-',label = 'f')
    plt.plot(x_plot,y_pred,label = 'f_{}'.format(str(kernel)))
    #plt.plot(x_plot,y_pred_rbf_best,label = 'frbf')
    #plt.plot(x_plot,p5,'-',label = 'p5')
    #plt.plot(x_plot,p95,'-',label = 'p95')
    plt.title(fig_name+"_param_{}_lambda_{}".format(best_param,best_L))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-10, 10)
    plt.savefig(fig_name+'b.png')

def plot( x,f,y,kernel,best_param,best_L,fig_name):
    ker = str(kernel)
    plt.clf()
    K_poly = form_kernel(x,kernel,best_param,x2=None)
    a = train(K_poly,y,best_L)
    x_plot = np.arange(0,1,0.001)
    x_plot =  x_plot.reshape(x_plot.shape[0],1)
    K_plot = form_kernel(x_plot,kernel,best_param,x2=x)
    y_pred = pred(K_plot,a)
    f_plot = 4*np.sin(np.pi*x_plot)*np.cos(6*np.pi*x_plot**2)
    x, p5, p95 = bootstrap(x, y, kernel,best_param, best_L, B=300)
    sns.set()
    plt.plot(x,y,'.',label ='y')
    plt.plot(x_plot,f_plot,'-',label = 'f')
    plt.plot(x_plot,y_pred,label = 'f_{}'.format(ker))
    #plt.plot(x_plot,y_pred_rbf_best,label = 'frbf')
    plt.plot(x_plot,p5,'-.',label = 'p5')
    plt.plot(x_plot,p95,'-.',label = 'p95')
    plt.title(fig_name+"_param_{}_lambda_{}".format(best_param,best_L))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-10, 10)
    plt.savefig(fig_name+'.png')


'''Q1 hyperparameter tuning'''
x,f,y = generate_dataset(30)
param_list_poly= np.arange(1,100,5)
param_list_rbf= np.arange(8, 11,0.5)
L_list = np.float_power( 10, np.arange(-8, 0) )
best_param_poly_loo,best_L_poly_loo, min_mse_poly_loo  = loo(x,y,param_list_poly,L_list,poly)
best_param_rbf_loo,best_L_rbf_loo, min_mse_rbf_loo = loo(x,y,param_list_rbf,L_list,rbf)
x,f,y = generate_dataset(300)
best_param_poly_cv,best_L_poly_cv, min_mse_poly_cv  = cv(x,y,param_list_poly,L_list,poly,10)
best_param_rbf_cv,best_L_rbf_cv, min_mse_rbf_cv  = cv(x,y,param_list_rbf,L_list,rbf,10)
print('best_L_poly_cv:{}, best_param_poly_cv:{}, min_mse_poly_cv:{}'.format(best_L_poly_cv, best_param_poly_cv,min_mse_poly_cv))
print('best_L_poly_loo:{}, best_param_poly_loo:{}, min_mse_poly_loo:{}'.format(best_L_poly_loo, best_param_poly_loo,min_mse_poly_loo))
print('best_L_rbf_cv:{}, best_param_rbf_cv:{}, min_mse_rbf_cv:{}'.format(best_L_rbf_cv, best_param_rbf_cv,min_mse_rbf_cv))
print('best_L_rbf_loo:{}, best_param_rbf_loo:{}, min_mse_rbf_loo:{}'.format(best_L_rbf_loo, best_param_rbf_loo,min_mse_rbf_loo))

x,f,y = generate_dataset(300)
plotb(x,f,y,rbf,best_param_rbf_cv,best_L_rbf_cv,'300_rbf')
plot(x,f,y,rbf,best_param_rbf_cv,best_L_rbf_cv,'300_rbf')

x,f,y = generate_dataset(300)
plotb(x,f,y,poly,best_param_poly_cv,best_L_poly_cv,'300_poly')
plot(x,f,y,poly,best_param_poly_cv,best_L_poly_cv,'300_poly')

x,f,y = generate_dataset(30)
plotb(x,f,y,rbf,best_param_rbf_loo,best_L_rbf_loo,'30_rbf')
plot(x,f,y,rbf,best_param_rbf_loo,best_L_rbf_loo,'30_rbf')

x,f,y = generate_dataset(30)
plotb(x,f,y,poly,best_param_poly_loo,best_L_poly_loo,'30_poly')
plot(x,f,y,poly,best_param_poly_loo,best_L_poly_loo,'30_poly')

m = 1000
B = 300
x_train,f_train,y_train = generate_dataset(300)
x,f,y = generate_dataset(m)
K_poly = form_kernel(x_train,poly,best_param_poly_cv,x2=None)
a_poly = train(K_poly, y_train, L=best_L_poly_cv)
K_rbf = form_kernel(x_train,rbf,best_param_rbf_cv,x2=None)
a_rbf = train(K_rbf, y_train, L=best_L_rbf_cv)
f_array = np.zeros((1, B))
for i in range(B):
    #print(i)
    idxs = np.random.choice(m, m)
    x_sub = x[idxs,:]
    y_sub = y[idxs,:]
    kx_poly = form_kernel(x_sub,poly,best_param_poly_cv,x2= x_train)
    f_poly = pred(kx_poly, a_poly)
    kx_rbf = form_kernel(x_sub,rbf,best_param_rbf_cv,x2= x_train)
    f_rbf = pred(kx_rbf, a_rbf)
    f = np.mean(np.power(y_sub-f_poly,2)-np.power(y_sub-f_rbf,2))
    #print(f)
    #print(len(set(idx)))
    f_array[0,i] = f
p5 = np.percentile(f_array, 5, axis = 1)
p95 = np.percentile(f_array, 95, axis = 1)

print("The confidence interval is ({},{})".format(p5,p95))
