__author__ = 'bobo'
import pandas as pd
import numpy as np

class Processor:
    def __init__(self):
        self.f_raw_requests_per_month = 'data/RequestsPerMonth.csv'
        self.f_parsed_requests_per_month = 'data/ParsedRequestsPerMonth.csv'

    def create_time_series_data(self):
        fin_raw_time_series_data = self.f_raw_requests_per_month
        fout_parsed_time_series_data = self.f_parsed_requests_per_month

        # d_date = {}
        # min_year, max_year = 2020, 2000
        # for line in open(fin_raw_time_series_data, 'r'):
        #     data = line.strip().split(',')
        #     year, month, requests = int(data[0]), int(data[1]), int(data[2])
        #     # data = load(line.strip())
        #     if year not in d_date:
        #         d_date[year] = {month: requests}
        #     else:
        #         d_date[year][month] = requests
        #     min_year, max_year = min(min_year, year), max(max_year, year)
        #
        # fout = open(fout_parsed_time_series_data, 'w')
        # for year in range(min_year, max_year, 1):
        #     for month in range(1, 13, 1):
        #         if year in d_date and month in d_date[year]:
        #             print("{},{},{}".format(year, month, d_date[year][month]))
        #             print("{},{},{}".format(year, month, d_date[year][month]), file=fout)
        #             # print("{}-{},{}".format(year, month, d_date[year][month]))
        #             # print("{}-{},{}".format(year, month, d_date[year][month]), file=fout)

        fin_raw_time_series_data = 'data/RequestsPerBoroughOverTime.csv'
        fout_parsed_time_series_data = 'data/RequestsPerBoroughOverTimeParsed.csv'

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

        # TODO: convert the daily file ...

    @staticmethod
    def geo_convert():
        file = 'data/weatherPerMonthLatLon.csv'
        # Latitude,Longitude,Year,Month,MeanTemp,WindSpeed,rain
        from geopy.geocoders import Nominatim
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

        # todo: missing values for each column ..
        idx_X = ['Month', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND', 'f1', 'f2', 'f3']
        misses = np.where(pd.isnull(df[idx_X]))
        # filling missing values for a particular column ..
        df['f2'] = df['f2'].fillna(0)

        print(df_requests.shape, df_weather.shape, df.shape)
        df.to_csv('data/RegressionData.csv', index=False)

        # TODO: Join daily data


def main():
    self = Processor()
    # self.geo_convert()
    self.join_weather_requests()


if __name__ == '__main__':
    main()