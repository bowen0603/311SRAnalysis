__author__ = 'bobo'

from pandas import read_csv
from pandas.tools.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class Modeling:
    def __init__(self):
        self.f_parsed_requests_per_month = 'data/ParsedRequestsOverTime.csv'

    def regression_analysis(self):
        pass

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
    self.time_series_analysis()


if __name__ == '__main__':
    main()