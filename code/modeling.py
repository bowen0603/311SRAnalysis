__author__ = 'bobo'

import pandas as pd
import numpy as np
from pandas import read_csv
from pandas.tools.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
import operator
from statsmodels.tsa.arima_model import ARIMA
from statistics import mean
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation as cv
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from constants import Constants


class Modeling:
    def __init__(self):
        self.n_folds = 5
        self.digit = 3
        self.const = Constants()
        self.f_parsed_requests_per_month = 'data/ParsedRequestsOverTime.csv'

        self.mse_scorer = make_scorer(mean_squared_error)
        self.r2_scorer = make_scorer(r2_score)
        self.mae_scorer = make_scorer(mean_absolute_error)

    @staticmethod
    def select_features():
        df = pd.read_csv('data/RegressionDailyData5BoroughsData.csv', delimiter=',')
        # df = df.drop(columns=['MinTemp', 'MaxTemp', 'MaxSustainedWind', 'DewPoint', 'Gust',
        #                       'USAF', 'WBAN', 'StationName', 'State', 'Latitude', 'Longitude'])
        # print(df.columns)
        # print(df.shape)  # 12295, 14

        df = df.drop(columns=['USAF', 'WBAN', 'StationName', 'State', 'Latitude', 'Longitude'])

        # for regression prediction
        df['Month'] = df.Month.astype('category')
        df['Day'] = df.Day.astype('category')
        return df

    def regression_analysis(self):
        df = self.select_features()

        df = df.drop(columns=['Day', 'Month'])
        # X = preprocessing.scale(df.loc[:, df.columns != 'requests'])
        X = df.loc[:, df.columns != 'requests']
        y = df.loc[:, df.columns == 'requests']

        # TODO: interaction between variables
        # TODO: std data for explanation
        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

    # TODO: data scaling, feature selection, PCA, etc
    def regression_prediction(self):

        # shuffle the data set
        df = self.select_features().sample(frac=1)
        df['Month'] = df.Month.astype('category')
        df['Day'] = df.Day.astype('category')
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']

        regressor = LinearRegression()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Linear Regression: {},{},{}'.format(round(r2, 3), round(mse, 3), round(mae, 3)))

        # R^2 [0.67927329 0.68152598 0.70665882 0.69121535 0.68096461]
        # MSE: [89406.04691581 89661.77389583 80106.25388333 87323.91620207 86910.77663342]

        regressor = LogisticRegression()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Logistic Regression: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

        # R^2 [0.29121242 0.39687124 0.38564635 0.44500327 0.39355215]
        # MSE: [197582.22209066 169802.21786198 167769.04533153 156952.3849797 165206.91813261]

        regressor = MLPRegressor()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Neural Networks: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

        regressor = GradientBoostingRegressor()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Gradient Boosting: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

        # ('46\t18\tlog2\t0.671\t115284.532\t246.164', '0.671')
        regressor = RandomForestRegressor()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print(
            'Gradient Boosting: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

    def grid_search_logistic_regression(self):
        df = self.select_features().sample(frac=1)
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']
        fout = open(self.const.tuning_lg, 'w')

        print('Logistic Regression')
        print('Regularizer\tC\tR2\tMSE\tMAE')
        for l in ['l1', 'l2']:
            for c in [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]:
                regressor = LogisticRegression(penalty=l, C=c)
                r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
                mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
                mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
                print('{}\t{}\t{}\t{}\t{}'.format(l, c, round(r2, self.digit),
                                                  round(mse, self.digit), round(mae, self.digit)), file=fout)
        print('done ... ')

    def grid_search_neural_network(self):
        df = self.select_features().sample(frac=1)
        # df = df.dropna()
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']
        fout = open(self.const.tuning_nnt, 'w')

        # TODO: fill nan ...
        print('Neural Networks')
        for layer in range(1, 50, 5):
            for loss in ['identity', 'logistic', 'tanh', 'relu']:
                for sol in ['lbfgs', 'sgd', 'adam']:
                    for l2 in [0.001, 0.01, 0.1, 1, 10, 100]:
                        for lrate in ['constant', 'invscaling', 'adaptive']:
                            try:
                                regressor = MLPRegressor(hidden_layer_sizes=layer, activation=loss, solver=sol,
                                                         alpha=l2, learning_rate=lrate)
                                r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
                                mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
                                mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
                                print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(layer, loss, sol, l2, lrate,
                                                                              round(r2, self.digit), round(mse, self.digit),
                                                                              round(mae, self.digit)), file=fout)
                            except:
                                continue
        print('done ... ')

    def grid_search_random_forests(self):
        df = self.select_features().sample(frac=1)
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']
        fout = open(self.const.tuning_rf, 'w')

        print('Random Forests')
        for trees in range(1, 50, 5):
            for depth in range(2, 20):
                for features in ['auto', 'sqrt', 'log2']:
                    regressor = RandomForestRegressor(n_estimators=trees, max_depth=depth, max_features=features)
                    r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
                    mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
                    mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
                    print('{}\t{}\t{}\t{}\t{}\t{}'.format(trees, depth, features,
                                                          round(r2, self.digit), round(mse, self.digit),
                                                          round(mae, self.digit)), file=fout)
        print('done ... ')

    def grid_search_gradient_boosting(self):
        df = self.select_features().sample(frac=1)
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']
        fout = open(self.const.tuning_gb, 'w')

        print('Gradient Boosting')
        for loss in ['ls', 'lad', 'huber', 'quantile']:
            for lrate in [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]:
                for estimators in range(20, 200, 10):
                    for depth in range(2, 20, 1):
                        for features in ['auto', 'sqrt', 'log2']:
                            regressor = GradientBoostingRegressor(loss=loss, learning_rate=lrate,
                                                                  n_estimators=estimators, max_depth=depth,
                                                                  max_features=features)
                            r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
                            mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
                            mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
                            print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(loss, lrate, estimators, features,
                                                                      round(r2, self.digit), round(mse, self.digit),
                                                                      round(mae, self.digit)), file=fout)
        print('done ... ')

    def grid_search_analysis(self):
        # d = {}
        # for line in open(self.const.tuning_gb, 'r'):
        #     data = line.strip().split('\t')
        #     d[line.strip()] = data[4]
        # lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        # print(lst[: 10])

        d = {}
        for line in open(self.const.tuning_rf, 'r'):
            data = line.strip().split('\t')
            d[line.strip()] = data[3]
        lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        print(lst[: 10])

        # d = {}
        # for line in open(self.const.tuning_nnt, 'r'):
        #     data = line.strip().split('\t')
        #     d[line.strip()] = data[5]
        # lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        # print(lst[: 10])
        #
        # d = {}
        # for line in open(self.const.tuning_lg, 'r'):
        #     data = line.strip().split('\t')
        #     d[line.strip()] = data[2]
        # lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        # print(lst[: 10])

    # TODO: borough level time series on day
    def time_series_analysis(self):
        series = read_csv(self.const.f_parsed_time_series_daily, header=0, parse_dates=[0], index_col=0, squeeze=True)
        # series = read_csv(self.const.f_parsed_time_series_monthly, header=0, parse_dates=[0], index_col=0, squeeze=True)
        # series = read_csv(self.f_parsed_requests_per_month, header=0, parse_dates=[0], index_col=0, squeeze=True)
        # # # Overall trend
        # print(series.head())
        # series.plot()
        # pyplot.show()
        # #
        # # # # autocorrelation plot of the time series
        # autocorrelation_plot(series)
        # pyplot.show()
        #
        # # # Residual analysis
        # model = ARIMA(series, order=(10, 1, 0))
        # model_fit = model.fit(disp=0)
        # print(model_fit.summary())
        # # plot residual errors
        # residuals = DataFrame(model_fit.resid)
        # residuals.plot()
        # pyplot.show()
        # residuals.plot(kind='kde')
        # pyplot.show()
        # print(residuals.describe())

        # # Create Prediction model & Rolling predictions
        # X = series.values
        # size = int(len(X) * 0.66)
        # train, test = X[0:size], X[size:len(X)]
        # history = [x for x in train]
        # predictions = list()
        # for t in range(len(test)):
        #     model = ARIMA(history, order=(10, 1, 0))
        #     model_fit = model.fit(disp=0)
        #     output = model_fit.forecast()
        #     output = model_fit.forecast(steps=7)
        #
        #     model_fit.plot_predict(1, 80)
        #
        #     yhat = output[0]
        #     time = test[0]
        #     t = series[size+t]
        #     # t = series[size + t] ###
        #     predictions.append(yhat)
        #     obs = test[t]
        #     history.append(obs)
        #     print('t={}, predicted={}, expected={}'.format(t, yhat, obs))
        # error = mean_squared_error(test, predictions)
        # print('Test MSE: {}'.format(error))
        # # plot
        # pyplot.plot(test)
        # pyplot.plot(predictions, color='red')
        # pyplot.show()

        X = series.values
        size = int(len(X) * 0.95)
        print(len(X), size)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        model_fit = None
        for t in range(len(test)):
            model = ARIMA(history, order=(10, 1, 0))
            model_fit = model.fit(disp=0)
            # from start to future .
            # model_fit.plot_predict(1, 80)
            # plt.show()

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))

        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)
        r2 = r2_score(test, predictions)

        print('Test MSE: {}, MAE: {}, R2: {}'.format(mse, mae, r2))
        # plot
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()

        model_fit.plot_predict(1, 3300)
        pyplot.show()
        print(model_fit.forecast(7))
        # pyplot.show()


def main():
    self = Modeling()
    self.time_series_analysis()
    # self.regression_analysis()
    # self.regression_prediction()

    # self.grid_search_random_forests()
    # self.grid_search_gradient_boosting()
    # self.grid_search_neural_network()
    # self.grid_search_logistic_regression()
    # self.grid_search_analysis()


if __name__ == '__main__':
    main()