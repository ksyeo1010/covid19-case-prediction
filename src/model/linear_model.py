import numpy as np
from numpy.linalg import solve
from utils import findMin
from scipy.optimize import approx_fprime
from utils import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        self.w = solve(X.T@z@X, X.T@z@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w.flatten(), lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        if (w.ndim is 1):
            w = w[:, np.newaxis]

        r = (X@w - y)
        
        # Calculate the function value
        # f = 0.5*np.sum((X@w - y)**2)
        f = np.sum(np.log(np.exp(r)+np.exp(-r)))

        # Calculate the gradient value
        # g = X.T@(X@w-y)
        g = X.T@((np.exp(r)-np.exp(-r))/(np.exp(r)+np.exp(-r)))
        

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        N, D = X.shape
        Z = np.ones((N, D))
        # Z = w_0*x_i**0 + w_i*x_i**1 
        Z = np.append(Z, X, axis=1)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        N, D = X.shape
        Z = np.ones((N, D))
        Z = np.append(Z, X, axis=1)
        return Z@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.leastSquares.fit(Z, y)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return self.leastSquares.predict(Z)

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        N,D = X.shape
        Z = np.ones((N, self.p + 1))
        for j in range(self.p + 1):
            # Z = [x_i**0 x_i**1 ... x_i**p]
            Z[:,j] = X[:,0]**j
        return Z

