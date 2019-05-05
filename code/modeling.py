__author__ = 'bobo'

import pandas as pd
import numpy as np
from pandas import read_csv
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from matplotlib import pyplot
import operator
from statsmodels.tsa.arima_model import ARIMA
from statistics import mean
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from constants import Constants


class Modeling:
    def __init__(self):
        self.n_folds = 5
        self.digit = 3
        self.const = Constants()

        self.mse_scorer = make_scorer(mean_squared_error)
        self.r2_scorer = make_scorer(r2_score)
        self.mae_scorer = make_scorer(mean_absolute_error)

    def select_features(self):
        df = pd.read_csv(self.const.f_regression_data, delimiter=',')
        df = df.drop(columns=['USAF', 'WBAN', 'StationName', 'State', 'Latitude', 'Longitude'])
        df = df.drop(columns=['MinTemp', 'MaxTemp', 'MaxSustainedWind'])
        df = df.drop(columns=['Day', 'Month'])

        # for regression prediction
        # df['Month'] = df.Month.astype('category')
        # df['Day'] = df.Day.astype('category')
        return df

    def regression_analysis(self):
        df = self.select_features()
        df = df.drop(columns=['DewPoint', 'Gust', 'SnowDepth'])
        df = df.drop(columns=['Day', 'Month'])
        X = df.loc[:, df.columns != 'requests']
        y = df.loc[:, df.columns == 'requests']

        est = sm.OLS(y, sm.add_constant(X))
        est2 = est.fit()
        print(est2.summary())

    def regression_prediction(self):
        # shuffle the data set
        df = self.select_features().sample(frac=1)
        # df['Month'] = df.Month.astype('category')
        # df['Day'] = df.Day.astype('category')
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']

        regressor = LinearRegression()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Linear Regression: {},{},{}'.format(round(r2, 3), round(mse, 3), round(mae, 3)))

        regressor = MLPRegressor()
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Neural Networks: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

        regressor = GradientBoostingRegressor(loss='ls', learning_rate=0.1,
                                                                  n_estimators=100, max_depth=3,
                                                                  max_features='sqrt')
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Gradient Boosting: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

        regressor = RandomForestRegressor(n_estimators=48, max_depth=18, max_features='log2')
        r2 = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.r2_scorer))
        mse = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mse_scorer))
        mae = mean(cross_val_score(regressor, X, y, cv=self.n_folds, scoring=self.mae_scorer))
        print('Random Forests: {},{},{}'.format(round(r2, self.digit), round(mse, self.digit), round(mae, self.digit)))

    def grid_search_neural_network(self):
        df = self.select_features().sample(frac=1)
        X, y = df.loc[:, df.columns != 'requests'], df.loc[:, df.columns == 'requests']
        fout = open(self.const.tuning_nnt, 'w')

        print('Neural Networks')
        for layer in range(1, 50, 5):
            for loss in ['identity', 'logistic', 'tanh', 'relu']:
                for sol in ['lbfgs', 'sgd', 'adam']:
                    for l2 in [0.01, 0.1, 1, 10, 100]:
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
        for trees in range(1, 100, 5):
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
            for lrate in [0.01, 0.1, 1, 5, 10, 100]:
                for estimators in range(50, 200, 10):
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
        d = {}
        for line in open(self.const.tuning_gb, 'r'):
            data = line.strip().split('\t')
            d[line.strip()] = data[4]
        lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        print(lst[: 10])

        d = {}
        for line in open(self.const.tuning_rf, 'r'):
            data = line.strip().split('\t')
            d[line.strip()] = data[3]
        lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        print(lst[: 10])

        d = {}
        for line in open(self.const.tuning_nnt, 'r'):
            data = line.strip().split('\t')
            d[line.strip()] = data[5]
        lst = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        print(lst[: 10])

    def time_series_analysis(self):
        series = read_csv(self.const.f_parsed_time_series_daily, header=0, parse_dates=[0], index_col=0, squeeze=True)
        # # Overall trend
        print(series.head())
        series.plot()
        pyplot.show()
        #
        # # # autocorrelation plot of the time series
        autocorrelation_plot(series)
        pyplot.show()

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
        size = int(len(X) * 0.99)
        print(len(X), size)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        model_fit = None
        for t in range(len(test)):
            model = ARIMA(history, order=(100, 1, 0))
            model_fit = model.fit(disp=0)

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('t={}, predicted={}, expected={}'.format(t, yhat, obs))

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

    def borough_analysis(self):
        df = self.select_features().sample(frac=1)
        df = df.drop(columns=['DewPoint', 'Gust', 'SnowDepth'])

        print(df.shape)
        df = df.drop(df[df['Percipitation'] >= 90].index)
        # df = df.drop(columns=['Day', 'Month'])
        # X = df.loc[:, df.columns != 'requests']
        # y = df.loc[:, df.columns == 'requests']

        # split train and test
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=22)
        print(df.shape, train_data.shape, test_data.shape)

        # train model and only run for test
        regressor = RandomForestRegressor(n_estimators=48, max_depth=18, max_features='log2')
        regressor.fit(train_data.loc[:, df.columns != 'requests'], train_data.loc[:, df.columns == 'requests'])



        X_test = test_data.loc[:, test_data.columns != 'requests']
        y_test = test_data.loc[:, test_data.columns == 'requests']
        # print(regressor.score(X_test, y_test))  # R^2

        y_pred = regressor.predict(X_test)
        print(r2_score(y_test, y_pred))

        print(df.columns)

        for borough in ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']:
            X_i = X_test[X_test[borough] == 1]
            y_i = y_test[X_test[borough] == 1]
            print(borough, X_i.shape, y_i.shape)
            # print(X_i.head(5))

            y_pred = regressor.predict(X_i)
            # print(regressor.score(X_i, y_i)) # R^2
            # print(mean_squared_error(y_i, y_pred))
            # print(mean_absolute_error(y_i, y_pred))
            print(r2_score(y_i, y_pred))

            # MSE
            # BRONX(449, 10)(449, 1)
            # 109940.16763186494
            # BROOKLYN(527, 10)(527, 1)
            # 291986.416238013
            # MANHATTAN(508, 10)(508, 1)
            # 106255.39869019548
            # QUEENS(515, 10)(515, 1)
            # 164849.31945597604
            # STATEN
            # ISLAND(460, 10)(460, 1)
            # 9163.177264208301

            # BRONX(477, 10)(477, 1)
            # 260.95069150268415
            # BROOKLYN(482, 10)(482, 1)
            # 405.0421529545333
            # MANHATTAN(507, 10)(507, 1)
            # 247.00113431769677
            # QUEENS(529, 10)(529, 1)
            # 311.3085174773187
            # STATEN
            # ISLAND(464, 10)(464, 1)
            # 76.81972654059618

    def learning_curve(self):
        df = self.select_features().sample(frac=1)
        df = df.drop(columns=['DewPoint', 'Gust', 'SnowDepth'])
        # df = df.drop(columns=['Day', 'Month'])
        # X = df.loc[:, df.columns != 'requests']
        # y = df.loc[:, df.columns == 'requests']

        plot_x = []
        plot_train = []
        plot_test = []
        for size in range(100, df.shape[0], 100):
            df_sub = df[:size]
            print(size, df_sub.shape)

            train_data, test_data = train_test_split(df_sub, test_size=0.3, random_state=22)
            # print(df.shape, train_data.shape, test_data.shape)

            regressor = RandomForestRegressor(n_estimators=48, max_depth=18, max_features='log2')
            X_train, y_train = train_data.loc[:, df.columns != 'requests'], train_data.loc[:, df.columns == 'requests']
            X_test, y_test = test_data.loc[:, test_data.columns != 'requests'], test_data.loc[:,
                                                                                test_data.columns == 'requests']
            regressor.fit(X_train, y_train)

            y_train_pred = regressor.predict(X_train)
            y_test_pred = regressor.predict(X_test)

            # mse_train = mean_squared_error(y_train, y_train_pred)
            # mse_test = mean_squared_error(y_test, y_test_pred)

            import math
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            mse_train = math.sqrt(mse_train)
            mse_test = math.sqrt(mse_test)

            # mse_train = mean_absolute_error(y_train, y_train_pred)
            # mse_test = mean_absolute_error(y_test, y_test_pred)

            plot_x.append(size)
            plot_train.append(mse_train)
            plot_test.append(mse_test)

        plt.plot(plot_x, plot_train, label='Train')
        plt.plot(plot_x, plot_test, label='Test')
        plt.legend(loc='best')
        plt.xlabel('# of data points')
        plt.ylabel('RMSE')
        plt.title('RMSE Train V.S. Test Data')
        plt.show()

    def individual_checkout(self):
        df = self.select_features().sample(frac=1)
        df = df.drop(columns=['DewPoint', 'Gust', 'SnowDepth'])



        train_data, test_data = train_test_split(df, test_size=0.3, random_state=22)
        print(df.shape, train_data.shape, test_data.shape)

        # train model and only run for test
        regressor = RandomForestRegressor(n_estimators=48, max_depth=18, max_features='log2')
        regressor.fit(train_data.loc[:, df.columns != 'requests'], train_data.loc[:, df.columns == 'requests'])



        # [0.15397431 0.05981746 0.15788365 0.01561587 0.00354469 0.03394104 0.18243093 0.03303471 0.0318347  0.32792266]
        # MeanTemp  Percipitation  WindSpeed  Rain  SnowIce  BRONX  BROOKLYN  MANHATTAN  QUEENS  STATEN ISLAND
        print(regressor.feature_importances_)


        d_mse_X = {}
        X_test, y_test = test_data.loc[:, df.columns != 'requests'], test_data.loc[:, df.columns == 'requests']
        for idx in range(test_data.shape[0]):
            X = X_test.iloc[[idx]]
            y = y_test.iloc[[idx]]
            y_pred = regressor.predict(X)
            mse = mean_squared_error(y, y_pred)
            d_mse_X[mse] = X

        import collections
        d_mse_X = collections.OrderedDict(sorted(d_mse_X.items(), reverse=True))
        # d_mse_X = collections.OrderedDict(sorted(d_mse_X, reverse=True))
        cnt = 0
        for k, v in d_mse_X.items():
            print('***{},{},{}'.format(cnt, k, v))
            cnt += 1
            if cnt == 10:
                break



def main():
    self = Modeling()
    # self.time_series_analysis()
    # self.regression_analysis()
    # self.regression_prediction()
    # self.borough_analysis()
    self.learning_curve()
    # self.individual_checkout()

    # self.grid_search_random_forests()
    # self.grid_search_gradient_boosting()
    # self.grid_search_neural_network()
    # self.grid_search_analysis()


if __name__ == '__main__':
    main()