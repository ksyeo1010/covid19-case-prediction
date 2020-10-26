# standard Python imports
import os
import argparse
import time
import pickle
import pandas as pd

# cs-340 helper code
from utils import utils

# 3rd party libraries
import numpy as np

# my modules
from model.linear_model import LeastSquares, LeastSquaresPoly
from model.auto_regressor import AutoRegressor
from model.knn import KNN


def __polyBasis(X, p):
    N = X.size
    Z = np.ones((N, p))
    for j in range(1, p + 1):
        # Z = [x_i**0 x_i**1 ... x_i**p]
        Z[:,j-1] = X**j
    return Z

def __sinBasis(X):
    N = X.size
    Z = np.ones((N, 2))
    Z[:,0] = X
    Z[:,1] = np.sin(X)
    return Z

def __find_min_poly_basis(X_nearest):
    y_test = [9504, 9530, 9541, 9557, 9585, 9585, 9585, 9627, 9654, 9664, 9699]

    min_error = np.inf

    ca_cases = X_ca["cases"].values
    ca_100k = X_ca["cases_100k"].values
    ca_14_100k = X_ca["cases_14_100k"].values

    for p1 in range(1,4):
        for p2 in range(1,4):
            for p3 in range(1,6):
                for k in range(30, 211):
                    ca_cases_trans = __polyBasis(ca_cases, p1)
                    ca_100k_trans = __polyBasis(ca_100k, p2)
                    ca_14_100k_trans = __polyBasis(ca_14_100k, p3)

                    X = np.column_stack((ca_cases_trans, ca_100k_trans, ca_14_100k_trans))
                    X = np.append(X, X_nearest, axis=1)

                    # fit to autogressor model
                    model = AutoRegressor(k=k)
                    model.fit(X, y_ca)
                    y_preds = model.predict(X)

                    error = np.sum(np.square(y_preds-y_test))
                    if (error < min_error):
                        p1_low = p1
                        p2_low = p2
                        p3_low = p3
                        k_low = k
                        min_error = error
    
    print("lowest p1: %d, p2: %d, p3: %d, k:%d, error: %.3f" % (p1_low, p2_low, p3_low, k_low, min_error)) 

def __find_min_sin_basis(X_nearest):
    y_test = [9504, 9530, 9541, 9557, 9585, 9585, 9585, 9627, 9654, 9664, 9699]

    min_error = np.inf

    ca_cases = X_ca["cases"].values
    ca_100k = X_ca["cases_100k"].values
    ca_14_100k = X_ca["cases_14_100k"].values

    for p in range(1,6):
        for k in range(30, 211):
            ca_cases_trans = __sinBasis(ca_cases)
            ca_100k_trans = __sinBasis(ca_100k)
            ca_14_100k_trans = __polyBasis(ca_14_100k, p)

            X = np.column_stack((ca_cases_trans, ca_100k_trans, ca_14_100k_trans))
            X = np.append(X, X_nearest, axis=1)

            # fit to autogressor model
            model = AutoRegressor(k=k)
            model.fit(X, y_ca)
            y_preds = model.predict(X)

            error = np.sum(np.square(y_preds-y_test))
            if (error < min_error):
                p_low = p
                k_low = k
                min_error = error
    
    print("lowest p: %d, k:%d, error: %.3f" % (p_low, k_low, min_error)) 


# main
if __name__ == "__main__":
    phase = "phase1"
    num_data = 11

    df = pd.read_csv(os.path.join("..", "data", phase + "_training_data.csv"))
    
    X_codes = df.country_id.unique()
    X_codes = X_codes[X_codes != "CA"]

    X_ca = df.loc[df["country_id"] == "CA"]

    ca_cases = X_ca["cases"].values
    ca_100k = X_ca["cases_100k"].values
    ca_14_100k = X_ca["cases_14_100k"].values
    y_ca = X_ca["deaths"].values

    ca_initial = np.column_stack((ca_cases, ca_100k, ca_14_100k))

    # get similar data sets
    # knn = KNN(k=1)
    # knn.fit(ca_initial, y_ca)
    # X_similar = np.ones(X_codes.size)
    # self_pred = knn.predict(ca_initial)
    # self_val = np.max(np.bincount(self_pred))
    # i = 0
    # for code in X_codes:
    #     data = df.loc[df["country_id"] == code]
    #     X_test = np.array(data[["cases", "cases_100k", "cases_14_100k"]].values)
    #     pred = knn.predict(X_test)
    #     if pred.size == 0: continue
    #     X_similar[i] = np.max(np.bincount(pred))
    #     i = i + 1

    # # use nearest = abs(X_similar - self_val) < 2
    # nearest = X_codes[np.abs(X_similar - self_val) < 2]

    # generate feature with basis
    ca_cases_trans = __polyBasis(ca_cases, 3)
    ca_100k_trans = __polyBasis(ca_100k, 1)
    ca_14_100k_trans = __polyBasis(ca_14_100k, 1)

    # combine column
    X = np.column_stack((ca_cases_trans, ca_100k_trans, ca_14_100k_trans))

    # append nearest to training data
    # X_nearest = np.zeros(shape=(X.shape[0], nearest.size*3))
    # i = 0
    # for code in nearest:
    #     data = df.loc[df["country_id"] == code]
    #     vals = data[["cases", "cases_100k", "cases_14_100k"]].values
    #     X_nearest[:, i:i+3] = vals
    #     i+=3

    # X = np.append(X, X_nearest.mean(axis=1).reshape(-1,1), axis=1)
    # __find_min_sin_basis(X_nearest)

    # fit to autogressor model
    model = AutoRegressor(k=42)
    model.fit(X, y_ca)
    y_preds = model.predict(X, num_data)
    print(y_preds)
    
    # to csv
    output = pd.DataFrame({'deaths': y_preds,
                           'id': range(num_data)})
    out_path = os.path.join("..", "data", phase + "_out.csv")
    output.to_csv(path_or_buf=out_path,index=False)