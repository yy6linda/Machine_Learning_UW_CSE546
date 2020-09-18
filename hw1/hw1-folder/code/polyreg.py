'''
    Template for polynomial regression
    AUTHOR Yao Yan
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.degree = degree
        self.reg_lambda = reg_lambda
        #TODO

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        output = np.empty([len(X), degree])
        for i in range(0, len(X)):
            x = X[i]
            for j in range(0, degree):
                output[i, j] = x ** (j + 1)
        return output

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        X = self.polyfeatures(X, self.degree)
        n = X.shape[0]
        self.get_mean_std(X)
        X = self.standardization(X)
        X = np.c_[np.ones([n, 1]), X]
        d = X.shape[1]
        '''this is from linreg_closedform.py'''
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        #print(y.shape)
        #print(X.T.shape)
        #print(reg_matrix.shape)
        self.theta = np.linalg.pinv(X.T.dot(X) + reg_matrix).dot(X.T).dot(y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        X = self.polyfeatures(X, self.degree)
        X = self.standardization(X)
        n = X.shape[0]
        X = np.c_[np.ones([n, 1]), X]
        y_pred = X.dot(self.theta)
        print("self.theta")
        print(self.theta)
        return y_pred

    def get_mean_std(self,X):
        """
        get the mean and standard deviation from training data
        """
        n = X.shape[0]
        d = X.shape[1]
        self.mean= np.zeros([d, 1])
        self.std = np.zeros([d, 1])
        for i in range(0, d):
            feature = np.empty(n)
            for j in range(0, n):
                feature[j] = X[j, i]
            self.mean[i] = np.mean(feature)
            self.std[i] = np.std(feature)

    def standardization(self,X):
        n = X.shape[0]
        d = X.shape[1]
        output = np.zeros([n, d])
        for i in range(0, n):
            for j in range(0, d):
                if self.std[j] == 0:
                    output[i, j] = 1
                else:
                    output[i, j] = (X[i, j] - self.mean[j])/self.std[j]

        return output

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    for i in range(0, n):
        model = PolynomialRegression(degree = degree, reg_lambda = reg_lambda)
        model.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])

        errorTrain[i] = stepError(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
        errorTest[i] = stepError(model.predict(Xtest), Ytest)
        print("errorTest[i]")
        print(errorTest[i])
    return errorTrain, errorTest

def stepError(pred, real):
    n = len(pred)
    total = 0
    for i in range(0, n):
        total += (pred[i] - real[i]) ** 2
    return total/n
