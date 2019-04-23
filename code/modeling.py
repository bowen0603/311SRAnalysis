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
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class Modeling:
    def __init__(self):
        self.n_folds = 5
        self.f_parsed_requests_per_month = 'data/ParsedRequestsOverTime.csv'

    # TODO: data scaling, feature selection, PCA, etc
    def regression_analysis(self):
        # convert month & day to categorical
        df = pd.read_csv('data/RegressionDailyData.csv', delimiter=',')
        df = df.drop(columns=['MinTemp', 'MaxTemp', 'MaxSustainedWind'])
        df['Month'] = df.Month.astype('category')
        df['Day'] = df.Day.astype('category')

        regressor = LinearRegression()
        eva = cross_val_score(regressor, df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests'],
                              cv=5)  # R^2
        print(eva)
        mse = make_scorer(mean_squared_error)
        eva = cross_val_score(regressor, df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests'],
                              cv=5, scoring=mse)
        print(eva)
        # R^2 [0.67927329 0.68152598 0.70665882 0.69121535 0.68096461]
        # MSE: [89406.04691581 89661.77389583 80106.25388333 87323.91620207 86910.77663342]

        # todo: DT need some tuning ..
        regressor = DecisionTreeRegressor(random_state=0)
        eva = cross_val_score(regressor, df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests'],
                              cv=5)  # R^2
        print(eva)
        eva = cross_val_score(regressor, df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests'],
                              cv=5, scoring=mse)
        print(eva)
        # R^2 [0.29121242 0.39687124 0.38564635 0.44500327 0.39355215]
        # MSE: [197582.22209066 169802.21786198 167769.04533153 156952.3849797 165206.91813261]

    # TODO: borough level time series on day
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