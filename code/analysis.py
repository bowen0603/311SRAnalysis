import numpy as np
import pandas as pd

from pandas import read_csv
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class DataExploration:
    def __init__(self):
        self.file_sr = 'data/311_Service_Requests_from_2010_to_Present.csv'
        self.file_weather = 'data/weather_NY_2010_2018Nov.csv'
        self.f_raw_requests_per_day = 'data/RequestsPerDay.csv'
        self.f_parsed_requests_per_day = 'data/ParsedRequestsPerDay.csv'
        self.f_raw_requests_per_month = 'data/RequestsPerMonth.csv'
        self.f_parsed_requests_per_month = 'data/ParsedRequestsPerMonth.csv'
        self.f_parsed_requests_per_month = 'data/ParsedRequestsOverTime.csv'
        self.df_sr = None
        self.df_weather = None

    def read_data(self):
        self.df_sr = pd.read_csv(self.file_sr, delimiter=',')
        # self.df_weather = pd.read_csv(self.file_weather, delimiter = ',')
        print("finished reading ... ")
        print(self.df_sr.groupby(['Location Type']).count())

    def plot_service_data(self):
        print("One dimensional analysis")
        # # Request over time
        # df = pd.read_csv(self.f_parsed_requests_per_month, delimiter=',')
        # df.plot.line(x='date', y='requests', title='Service Requests over Time', marker='o')
        # plt.show()

        # # Complaint type bar chart => Probably better to just use list and categorization
        # df = pd.read_csv('data/complainTypeSort.csv', delimiter=',')
        # topN = 20
        # df = df.head(topN)
        # df.plot.bar(x='ComplaintType', y='requests', title='Top {} Complaint Type Distribution'.format(topN), rot=0)
        # plt.xticks(rotation=45)
        # plt.show()

        # location/Borough bar chart
        # df = pd.read_csv('data/RequestsPerBorough.csv', delimiter=',')
        # df.set_index('Borough')
        # df.plot.pie(y='requests', autopct='%.2f', labels=df['Borough'])
        # plt.show()

        # over year bar chart
        df = pd.read_csv(self.f_parsed_requests_per_month, delimiter=',')
        df_year = df.groupby('year').sum()
        df_year.reset_index().plot.line(x=df_year.index, y='requests', title='Requests Over Years', marker='o')
        plt.xlabel('year')
        plt.show()

        # over month bar chart
        df_month = df.groupby('month').sum()
        df_month.reset_index().plot.line(x=df_month.index, y='requests', title='Requests Over Months', marker='o')
        plt.xlabel('month')
        plt.show()

        # TODO:2 D
        # over complaint type over year
        # over complaint type over months

        # location/boar over year
        # location in months

        # TODO: 3D
        # complaint tyoe in location over time (gif)

    def plot_weather_data(self):
        # TODO
        # changes over time
        # correlation among each attributes
        pass

    def create_time_series_data(self):
        fin_raw_time_series_data = self.f_raw_requests_per_month
        fout_parsed_time_series_data = self.f_parsed_requests_per_month

        d_date = {}
        min_year, max_year = 2020, 2000
        for line in open(fin_raw_time_series_data, 'r'):
            data = line.strip().split(',')
            year, month, requests = int(data[0]), int(data[1]), int(data[2])
            # data = load(line.strip())
            if year not in d_date:
                d_date[year] = {month: requests}
            else:
                d_date[year][month] = requests
            min_year, max_year = min(min_year, year), max(max_year, year)

        fout = open(fout_parsed_time_series_data, 'w')
        for year in range(min_year, max_year, 1):
            for month in range(1, 13, 1):
                if year in d_date and month in d_date[year]:
                    print("{},{},{}".format(year, month, d_date[year][month]))
                    print("{},{},{}".format(year, month, d_date[year][month]), file=fout)
                    # print("{}-{},{}".format(year, month, d_date[year][month]))
                    # print("{}-{},{}".format(year, month, d_date[year][month]), file=fout)

        # TODO: convert the daily file ...

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
    self = DataExploration()
    # self.read_data()
    # self.create_time_series_data()
    # self.time_series_analysis()
    self.plot_service_data()


if __name__ == '__main__':
    main()