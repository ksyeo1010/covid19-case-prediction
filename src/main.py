# standard Python imports
import os
import argparse
import time
import pickle
import pandas as pd
import datetime as dt

# cs-340 helper code
from utils import utils

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# my modules
from model.linear_model import LeastSquares, LeastSquaresPoly
from model.auto_regressor import AutoRegressor

# try many for diff values of k and p
def __find_min_poly_basis(X, y, num_data, log_base=False, use_prediction=True):
    # first is for phase 1, second is for phase 2
    # y_test = [9504,9530,9541,9557,9585,9585,9585,9627,9654,9664,9699]
    y_test = [9504,9530,9541,9557,9585,9585,9585,9627,9654,9664,9699,9722,9746,9760,9778,9794,9829,9862,9888,9922]

    min_error = np.inf

    for p in range(1,5):
        for k in range(1, 280):
            model = AutoRegressor(k=k, p=p, log_base=log_base, use_prediction=use_prediction)
            try:
                model.fit(X, y)
            except:
                continue
            y_preds = model.predict(X, num_data)
            if log_base == True:
                y_preds = np.exp(y_preds)
            error = np.sum(np.square(y_preds-y_test))
            # print("p: %d, k: %d, error: %.3f" % (p, k, error))
            if (error < min_error):
                p_low = p
                k_low = k
                min_error = error
    
    print("lowest p: %d, k: %d, error: %.3f" % (p_low, k_low, min_error)) 

def __plot(x, y, fname, label, xlabel, ylabel):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()
    plt.title(label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join("..", "figs", fname))
    plt.clf()

# main
if __name__ == "__main__":
    # modify variables here
    phase = "phase2"
    num_data = 5
    log_base = True
    use_prediction = True

    # load file + set data
    df = pd.read_csv(os.path.join("..", "data", phase + "_training_data.csv"))
    
    X_codes = df.country_id.unique()
    X_codes = X_codes[X_codes != "CA"]

    X_ca = df.loc[df["country_id"] == "CA"]

    ca_cases = X_ca["cases"].values
    ca_100k = X_ca["cases_100k"].values
    ca_14_100k = X_ca["cases_14_100k"].values
    ca_deaths = X_ca["deaths"].values
    dates = X_ca["date"].values

    # combine column
    X = np.column_stack((ca_cases, ca_100k, ca_14_100k, ca_deaths))

    # if we are using log space
    if log_base == True:
        X = np.log(X)
        X[X == -np.inf] = 0

        ca_deaths = np.log(ca_deaths)
        ca_deaths[ca_deaths == -np.inf] = 0

    # run test only in phase 1 mode
    # __find_min_poly_basis(X, ca_deaths, num_data, log_base, use_prediction)
    
    # code for plots
    # dates_obj = [dt.datetime.strptime(d, "%m/%d/%Y").date() for d in dates]
    # labels = ["cases vs date", "cases_100k vs date", "cases_14_100k vs date", "deaths vs date"]
    # ylabels= ["cases", "cases_100k", "cases_14_100k", "deaths"]
    # fnames = ["cases", "cases_100k", "cases_14_100k", "deaths"]

    # for i in range (4):
    #     if log_base == True:
    #         __plot(dates_obj, X[:,i], fnames[i]+"_log", labels[i] + " (log)", "date", ylabels[i] + " (log)")
    #     else:
    #         __plot(dates_obj, X[:,i], fnames[i], labels[i], "date", ylabels[i])
        

    # fit to autogressor model
    model = AutoRegressor(k=79, p=1, log_base=log_base, use_prediction=use_prediction)
    model.fit(X, ca_deaths)
    y_preds = model.predict(X, num_data)

    if log_base == True:
        y_preds = np.exp(y_preds)

    y_preds = y_preds.astype('int64')
    print(y_preds)
    
    # to csv
    output = pd.DataFrame({'deaths': y_preds,
                           'id': range(num_data)})
    out_path = os.path.join("..", "data", phase + "_out.csv")
    output.to_csv(path_or_buf=out_path,index=False)