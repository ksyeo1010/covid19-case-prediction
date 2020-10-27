import numpy as np
from model.linear_model import LeastSquares

class AutoRegressor:
    def __init__(self, k, p=1, 
                log_base=False,
                use_prediction=True,
                model=LeastSquares):
        """
        k                   = the number of data sets to look behind to make next prediction
        p                   = polynomial basis
        log_base            = change data to log space
        use_prediction       = use prediction from X_k to test y_k+1, if false uses
                              time series the the deaths column itself
        model               = the linear model to use. We only use Least Squares
        In this class, everything with variable X means it is not weighted
        everything with variable Z means it has been transformed/weighted.
        """
        self.k = k
        self.p = p
        self.log_base = log_base
        self.use_prediction = use_prediction
        self.model = model
        self.y_model = model()

    def fit(self, X, y):
        N,D = X.shape
        k = self.k

        # if k is too large don't do this.
        self.size = N - k
        if (self.size <= 0): return

        # transform X to Z using transform method.
        Z = self.__transform(X)
        # get values of y for time series
        y_k = y[self.size:N]

        # set our features to predict next features.
        self.__feature_time_series(X, Z)
        # our general prediction for y_k+1
        self.y_model.fit(Z, y_k)

    def __feature_time_series(self, X, Z):
        """
        Grabs data from k to N-1 for each feature. If p > 0, also adds polynomial basis
        columns.
        Generates models to predict x_i_k+1 so that we can comput y_k+1.
        The models here are by independent, meaning they will have their own
        bias when fitting and predicting the model.
        """
        N, D = X.shape
        p = self.p

        feature_models = []
        for i in range(D):
            pos = i*p+1
            # grab the data from Z and add a bias 1 or e.
            x_i = self.__add_bias(Z[:,pos:pos+p])
            x_k = X[self.size:N,i]
            # create model and fit.
            model = self.model()
            model.fit(x_i, x_k)
            # save model to predict x_i_k+1
            feature_models.append(model)

        self.feature_models = feature_models

    def __transform(self, X):
        """
        Transforms X into Z. Adds the bias 1 if non logbase, e if logbase.
        Adds polynomial basis given p.
        """
        N,D = X.shape
        k = self.k
        p = self.p

        # new data set
        Z = np.ones(shape=(k, D*p+1))
        k_data = X[self.size-1:N-1, :]

        for i in range(D):
            pos = i*p+1
            # add polynomial basis
            Z[:,pos:pos+p] = self.__polyBasis(k_data[:,i])

        # handle log case
        if self.log_base == True:
            Z[:,0] = np.exp(1)

        return Z

    def __polyBasis(self, X):
        """
        General function to change to poly basis.
        NOTE: this does not add 1 as a bias.
        """
        N = X.size
        p = self.p

        Z = np.ones((N, p))
        for j in range(1, p + 1):
            Z[:,j-1] = X**j

        return Z

    def __add_bias(self, X):
        """
        Adds a bias of e if log_base, 1 otherwise.
        """
        N,D = X.shape
        Z = np.ones(shape=(N,D+1))
        Z[:,1:] = X

        if self.log_base == True:
            Z[:,0] = np.exp(1)

        return Z

    def predict(self, X, num_times):
        """
        Predict the next sequence of num_times.
        """
        def append_time_series(Z, y_k):
            """
            Predict x_i_k+1, removes x_i_1 from Z, and appends the new value.
            If use_prediction is on, it will use the predicted value y_k+1 and its 
            polynomial biases for next predict. Else it will use the values from
            its features series x_i_k+1.
            """
            N,D = Z.shape
            p = self.p
            feature_models = self.feature_models

            i = 0
            time_preds = np.ones(shape=(1,D))
            for model in feature_models:
                # predict the new features
                pos = i*p+1
                x_i = self.__add_bias(Z[:,pos:pos+p])
                pred = model.predict(x_i)[-1]
                i += 1

                time_preds[:,pos:pos+p] = self.__polyBasis(pred)

            # determine log base
            if self.log_base == True:
                time_preds[:,0] = np.exp(1)

            # remove first row, append to last row.
            Z = Z[1:]
            Z = np.append(Z, time_preds, axis=0)

            # use prediction will change the deaths column to predicted value and its
            # polynomial basis.
            if self.use_prediction == True:
                Z[:,pos:pos+p] = self.__polyBasis(y_k)

            return Z

        Z = self.__transform(X)
        y_k = self.y_model.predict(Z)[-1]
        
        y_pred = np.zeros(num_times)

        for i in range(num_times):
            # predict, and save results.
            Z = append_time_series(Z, y_k)
            y_k = self.y_model.predict(Z)[-1]
            y_pred[i] = y_k

        return y_pred



