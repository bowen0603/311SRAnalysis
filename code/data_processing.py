__author__ = 'bobo'
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from constants import Constants
import json

class Processor:
    def __init__(self):
        self.const = Constants()
        self.f_raw_requests_per_month = 'data/RequestsPerMonth.csv'
        self.f_parsed_requests_per_month = 'data/ParsedRequestsPerMonth.csv'

    def create_time_series_data(self):
        fin_raw_time_series_data = self.f_raw_requests_per_month
        fout_parsed_time_series_data = self.f_parsed_requests_per_month

        # Requests per month
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

        fin_raw_time_series_data = 'data/RequestsPerBoroughOverTime.csv'
        fout_parsed_time_series_data = 'data/RequestsPerBoroughOverTimeParsed.csv'

        # Requests per month per borough
        d_date = {}
        min_year, max_year = 2020, 2000
        for line in open(fin_raw_time_series_data, 'r'):
            data = line.strip().split(',')
            year, month, borough, requests = int(data[0]), int(data[1]), data[2], int(data[3])
            # data = load(line.strip())
            if borough not in d_date:
                d_date[borough] = {year: {month: requests}}
            else:
                # d_date[borough][year][month] = requests
                if year not in d_date[borough]:
                    d_date[borough][year] = {month: requests}
                else:
                    d_date[borough][year][month] = requests
            min_year, max_year = min(min_year, year), max(max_year, year)

        fout = open(fout_parsed_time_series_data, 'w')
        for borough in ['BRONX', 'QUEENS', 'MANHATTAN', 'BROOKLYN', 'STATEN ISLAND', 'Unspecified']:
            for year in range(min_year, max_year, 1):
                for month in range(1, 13, 1):
                    if year in d_date[borough] and month in d_date[borough][year]:
                        print("{}-{},{},{}".format(year, month, borough, d_date[borough][year][month]))
                        print("{}-{},{},{}".format(year, month, borough, d_date[borough][year][month]), file=fout)
                        # print("{}-{},{}".format(year, month, d_date[year][month]))
                        # print("{}-{},{}".format(year, month, d_date[year][month]), file=fout)

    def create_time_series_data_daily(self):
        # d Da,Cr,at,1

        d_date = {}
        min_year, max_year = 2020, 2000
        for line in open(self.const.f_raw_time_series_daily, 'r'):
            data = line.strip().split(',')
            year, month, day, requests = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            # data = load(line.strip())
            if year not in d_date:
                d_date[year] = {month: {day: requests}}
            else:
                if month not in d_date[year]:
                    d_date[year][month] = {day: requests}
                else:
                    d_date[year][month][day] = requests
            min_year, max_year = min(min_year, year), max(max_year, year)

        fout = open(self.const.f_parsed_time_series_daily, 'w')
        for year in range(min_year, max_year, 1):
            for month in range(1, 13, 1):
                for day in range(1, 32, 1):
                    if year in d_date and month in d_date[year] and day in d_date[year][month]:
                        print("{}-{}-{},{}".format(year, month, day, d_date[year][month][day]))
                        print("{}-{}-{},{}".format(year, month, day, d_date[year][month][day]), file=fout)

    def create_time_series_data_monthly(self):
        d_date = {}
        min_year, max_year = 2020, 2000
        for line in open(self.const.f_raw_time_series_monthly, 'r'):
            data = line.strip().split(',')
            year, month, requests = int(data[0]), int(data[1]), int(data[2])
            # data = load(line.strip())
            if year not in d_date:
                d_date[year] = {month: requests}
            else:
                d_date[year][month] = requests
            min_year, max_year = min(min_year, year), max(max_year, year)

        fout = open(self.const.f_parsed_time_series_monthly, 'w')
        for year in range(min_year, max_year, 1):
            for month in range(1, 13, 1):
                if year in d_date and month in d_date[year]:
                    print("{}-{},{}".format(year, month, d_date[year][month]))
                    print("{}-{},{}".format(year, month, d_date[year][month]), file=fout)

    def geo_retriever(self):
        geolocator = Nominatim()

        d_coor_addr = {}
        for line in open(self.const.f_data_weather, 'r'):
            data = line.strip().split(',')
            lat, long = data[4], data[5]
            coor = "{}, {}".format(lat, long)
            if coor not in d_coor_addr:
                addr = str(geolocator.reverse(coor))
                d_coor_addr[coor] = addr
                print('{}, {}'.format(coor, addr))

        fout = open('data/LatitudeLongitudeToAddress.json', 'w')
        for coor in d_coor_addr.keys():
            d_out = {'lat_long': coor, 'address': d_coor_addr[coor]}
            print(json.dumps(d_out), file=fout)


    @staticmethod
    def geo_convert():
        file = 'data/weatherPerMonthLatLon.csv'
        # Latitude,Longitude,Year,Month,MeanTemp,WindSpeed,rain
        geolocator = Nominatim()

        fout = open('data/ParsedWeatherPerMonthLatLon.csv', 'w')
        d_coor_borough = {}
        for line in open(file, 'r'):
            # Latitude,Longitude,Year,Month,MeanTemp,WindSpeed,rain
            data = line.strip().split(',')
            coor = "{}, {}".format(data[0], data[1])
            if coor in d_coor_borough:
                borough = d_coor_borough[coor]
            else:
                loc = str(geolocator.reverse(coor))
                borough = None
                if 'Queens County' in loc:
                    borough = 'QUEENS'
                elif 'Bronx County' in loc:
                    borough = 'BRONX'
                elif 'Kings County' in loc:
                    borough = 'BROOKLYN'
                elif 'Richmond County' in loc:
                    borough = 'STATEN ISLAND'
                elif 'New York County' in loc:
                    borough = 'MANHATTAN'
                d_coor_borough[coor] = borough

            if borough is not None:
                print("{},{}".format(borough, line.strip()))
                print("{},{}".format(borough, line.strip()), file=fout)

    def join_weather_requests(self):
        # Join monthly data ...
        df_requests = pd.read_csv('data/RequestsPerBoroughOverTime.csv', delimiter=',')
        df_requests['Month'] = df_requests['Month'].astype(int)
        df_weather = pd.read_csv('data/ParsedWeatherPerMonthLatLon.csv', delimiter=',')

        df = pd.merge(df_requests, df_weather, on=['Borough', 'Year', 'Month'], how='inner')
        print(df.head(10))

        idx_X = ['Month', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND', 'f1', 'f2', 'f3']
        misses = np.where(pd.isnull(df[idx_X]))
        df['f2'] = df['f2'].fillna(0)

        print(df_requests.shape, df_weather.shape, df.shape)
        df.to_csv('data/RegressionData.csv', index=False)

        # Join daily data
        # df_weather = pd.read_csv('data/WeatherBoroughsFull.csv', delimiter=',')
        df_weather = pd.read_csv('data/Weather5BoroughsData.csv', delimiter=',')
        df_requests = pd.read_csv('data/RequestsPerBoroughPerDay.csv', delimiter=',')
        df_requests['Month'] = df_requests['Month'].astype(int)
        df = pd.merge(df_requests, df_weather, on=['Year', 'Month', 'Day', 'Borough'], how='inner')

        idx_features = ['MeanTemp','MinTemp','MaxTemp','DewPoint',
                        'Percipitation','WindSpeed','MaxSustainedWind','Gust','Rain',
                        'SnowDepth','SnowIce','Year','Month','Day', 'requests', 'Borough']

        for feature in idx_features:
            idx = [feature]
            misses = np.where(pd.isnull(df[idx]))
            print(feature, 1.0*len(misses[0])/df.shape[0])
            if len(misses[0]) > 0:
                df[idx] = df[idx].fillna(0)

        # # fill missing values
        # df['Percipitation'] = df['Percipitation'].fillna(0)
        # df['WindSpeed'] = df['WindSpeed'].fillna(0)
        # df['SnowDepth'] = df['SnowDepth'].fillna(0)

        # df['MaxSustainedWind'] = df['MaxSustainedWind'].fillna(0)
        # encode categorical feature
        df_borough = pd.get_dummies(df.Borough.astype('category'))
        df = df.drop(columns=['Borough', 'Year'])
        df = pd.concat([df, df_borough], axis=1)

        df.to_csv(self.const.f_regression_data, index=False)

    def insert_col(self):
        df = pd.read_csv('data/manhattan.csv', delimiter=',')
        df['Borough'] = 'MANHATTAN'
        df.to_csv('data/manhattan_full.csv', index=False)

        df = pd.read_csv('data/staten island.csv', delimiter=',')
        df['Borough'] = 'STATEN ISLAND'
        df.to_csv('data/staten_island_full.csv', index=False)

        df = pd.read_csv('data/queens.csv', delimiter=',')
        df['Borough'] = 'QUEENS'
        df.to_csv('data/queens_full.csv', index=False)

        df = pd.read_csv('data/brooklyn.csv', delimiter=',')
        df['Borough'] = 'BROOKLYN'
        df.to_csv('data/brooklyn_full.csv', index=False)

        df = pd.read_csv('data/bronx.csv', delimiter=',')
        df['Borough'] = 'BRONX'
        df.to_csv('data/bronx_full.csv', index=False)


def main():
    self = Processor()
    self.geo_convert()
    self.join_weather_requests()
    self.geo_retriever()
    self.insert_col()
    self.create_time_series_data_monthly()
    self.create_time_series_data_daily()


if __name__ == '__main__':
    main()