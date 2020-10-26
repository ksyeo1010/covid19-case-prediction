import numpy as np
from model.linear_model import LeastSquares

class AutoRegressor:
    def __init__(self, k):
        self.k = k
        self.x_model = LeastSquares()
        self.y_model = LeastSquares()

    def fit(self, X, y):
        N,D = X.shape
        k=self.k

        Z = self.__transform(X)
        X_t = X[k+1:N, :]
        y_t = y[k+1:N]

        self.X_t = X_t
        self.y_t = y_t
        # now we have model for X
        self.x_model.fit(Z, X_t)
        # models for y
        self.y_model.fit(Z, y_t)

    def __transform(self, X):
        """
        Transform data X to Z for time series
                       1 ... T-k                  1 ... k
        Columns of Z = 1 ... T-k+1   = rows of X  1 ... k  
                       ...                        ...
        """
        N,D = X.shape
        k = self.k

        size = N-k-1
        Z = np.ones(shape=(size, D*k+1))

        for i in range(k):
            start = i*D+1
            end = start+D
            Z[:, start:end] = X[i+1:size+i+1, :]

        return Z


    def predict(self, X, num_times=11):
        def append_time_series(Z, x_pred):
            """
            Get new Z with calculated x_pred for time series
            """
            N,D = Z.shape
            n = x_pred.size

            Z = Z[1:]
            new_t = np.zeros(shape=(D))

            new_t[:-n] = Z[-1,n:]
            new_t[-n:] = x_pred
            Z = np.append(Z, new_t.reshape(1,-1), axis=0)

            return Z

        Z = self.__transform(X)
        x_pred = self.x_model.predict(Z)[-1] # first pred to get next y
        
        y_preds = np.zeros(num_times)

        for i in range(num_times):
            Z = append_time_series(Z, x_pred)
            y_preds[i] = self.y_model.predict(Z)[-1]
            x_pred = self.x_model.predict(Z)[-1]

        return y_preds



