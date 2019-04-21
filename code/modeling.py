__author__ = 'bobo'

import pandas as pd
import numpy as np
from pandas import read_csv
from pandas.tools.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import DataFrame
# from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation as cv
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class Modeling:
    def __init__(self):
        self.n_folds = 3
        self.f_parsed_requests_per_month = 'data/ParsedRequestsOverTime.csv'

    # TODO: regression model to predict monthly/daily requests
    # data scaling, feature selection, PCA, etc (with full dataset)
    def regression_analysis(self):
        # year info is not needed,
        # convert month & day to categorical
        df = pd.read_csv('data/RegressionData.csv', delimiter=',')
        df['Month'] = df.Month.astype('category')
        df_borough = pd.get_dummies(df.Borough.astype('category'))
        df = pd.concat([df, df_borough], axis=1)

        # todo: double check for the other two borough ... to create a dictionary ..
        idx_X = ['Month', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND', 'f1', 'f2', 'f3']
        idx_y = ['requests']

        df['f2'] = df['f2'].fillna(0)

        regressor = LinearRegression()
        eva = cross_val_score(regressor, df[idx_X], df[idx_y], cv=5)  # R^2
        print(eva)

        # todo: DT need some tuning ..
        regressor = DecisionTreeRegressor(random_state=0)
        eva = cross_val_score(regressor, df[idx_X], df[idx_y], cv=5)  # R^2
        print(eva)


        # print(df[idx_X].head(5))
        # print(df[idx_y].head(5))
        # # cross validation
        # for idx_train, idx_test in cv.KFold(df.shape[0], n_folds=self.n_folds):
        #     X_train, X_test = df[idx_X].iloc[idx_train], df[idx_X].iloc[idx_test]
        #     y_train, y_test = df[idx_y].iloc[idx_train], df[idx_y].iloc[idx_test]
        #
        #     print(X_train.head(5))

    def time_series_analysis(self):
        series = read_csv(self.f_parsed_requests_per_month, header=0, parse_dates=[0], index_col=0, squeeze=True)
        # # # Overall trend
        print(series.head())
        # series.plot()
        # pyplot.show()
        #
        # # # autocorrelation plot of the time series
        # autocorrelation_plot(series)
        # pyplot.show()

        # # Residual analysis

        model = ARIMA(series, order=(10, 1, 0))
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = DataFrame(model_fit.resid)
        residuals.plot()
        pyplot.show()
        residuals.plot(kind='kde')
        pyplot.show()
        print(residuals.describe())

        # # Create Prediction model & Rolling predictions
        X = series.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(10, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            output = model_fit.forecast(steps=7)

            model_fit.plot_predict(1, 80)

            yhat = output[0]
            time = test[0]
            t = series[size+t]
            t = series[size + t]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('t={}, predicted={}, expected={}'.format(t, yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: {}'.format(error))
        # plot
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()


def main():
    self = Modeling()
    self.regression_analysis()


if __name__ == '__main__':
    main()