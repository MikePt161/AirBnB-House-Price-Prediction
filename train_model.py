import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression


def read_data():
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').to_numpy().ravel()
    y_test = pd.read_csv('data/y_test.csv').to_numpy().ravel()

    X_train = X_train.drop('id', axis=1)
    X_test = X_test.drop('id', axis=1)

    return X_train, X_test, y_train, y_test


def run_experiment(X_train, X_test, y_train, y_test, model=LinearRegression()):
    """
    Runs experiment by using the ready_data function
    #### Inputs:
    #### Outputs:
    """

    # Fit Model

    model.fit(X_train, y_train)

    # Predict

    y_pred = model.predict(X_test)

    # Diagnostics

    print(str(model), "Diagnostics: ")
    # if (str(model) == "LinearRegression(n_jobs=-1)"):
    #     print(str(model), " Coefficients: \n ", model.coef_)
    #     print(f'R2: {r2_score(y_test, y_pred)}')

    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)} \n')

    return None


def main():
    X_train, X_test, y_train, y_test = read_data()

    model = [LinearRegression(n_jobs=-1),
             KNeighborsRegressor(),
             RandomForestRegressor(n_estimators=200, n_jobs=-1),
             GradientBoostingRegressor(n_estimators=5),
             DummyRegressor(strategy='quantile', quantile=0.5)
             ]

    for element in model:
        run_experiment(X_train, X_test, y_train, y_test, model=element)

    # Save feature names to use in deployment
    model[2].feature_names = X_train.columns
    # print(model[0].feature)

    # Save model and load to test it
    from joblib import dump, load
    dump(model[0], 'data/rf.joblib')


if __name__ == "__main__":
    main()
