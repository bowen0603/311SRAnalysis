import numpy as np
import pandas as pd

from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame

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

        ##########################################################################################
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
        # plt.legend(loc='best')
        # plt.show()

        # # over year bar chart
        # df = pd.read_csv(self.f_parsed_requests_per_month, delimiter=',')
        # df_year = df.groupby('year').sum()
        # df_year.reset_index().plot.line(x=df_year.index, y='requests', title='Requests Over Years', marker='o')
        # plt.xlabel('year')
        # plt.show()
        #
        # # over month bar chart
        # df_month = df.groupby('month').sum()
        # df_month.reset_index().plot.line(x=df_month.index, y='requests', title='Requests Over Months', marker='o')
        # plt.xlabel('month')
        # plt.show()

        ##########################################################################################
        print("Two dimensional analysis")
        # over complaint type over year
        # over complaint type over months

        # # Requests per Complaint Type Over Years
        # df = pd.read_csv('data/RequestsOverTimeOverAreaOverType.csv', delimiter=',')
        # df_year = df.groupby(['Year', 'ComplaintType']).sum().reset_index()
        # df_year.pivot(index='Year', columns='ComplaintType', values='requests').plot()
        # plt.legend(loc='best')
        # # plt.title(title='Requests Over Years', loc='center')
        # plt.xlabel('Year')
        # df_year.plot()
        # plt.show()
        #
        # # Requests per Borough Over Months
        # df_month = df.groupby(['Month', 'ComplaintType']).sum().reset_index()
        # df_month.pivot(index='Month', columns='ComplaintType', values='requests').plot()
        # plt.legend(loc='best')
        # # plt.title(title='Requests Over Months', loc='center')
        # plt.xlabel('Month')
        # plt.show()

        # # Requests per Borough per Complaint Type
        # df = pd.read_csv('data/RequestsOverTimeOverAreaOverType.csv', delimiter=',')
        # df_borough = df.groupby(['Borough', 'ComplaintType']).sum().reset_index()
        # df_borough.pivot("Borough", "ComplaintType", "requests").plot(kind='bar')
        # plt.legend(loc='best')
        # plt.xticks(rotation=45)
        # plt.show()

        # # # Requests per Borough Over Time
        # df_time = pd.read_csv('data/RequestsPerBoroughOverTimeParsed.csv', delimiter=',')
        # df_time.pivot(index='Time', columns='Borough', values='requests').plot()
        # plt.legend(loc='best')
        # plt.xlabel('Time')
        # plt.show()

        # # Requests per Borough Over Years
        # df = pd.read_csv('data/RequestsPerBoroughOverTime.csv', delimiter=',')
        # df_year = df.groupby(['Year', 'Borough']).sum().reset_index()
        # df_year.reset_index().plot.line(x=df_year.index, y='requests', title='Requests Over Years')
        # df_year.pivot(index='Year', columns='Borough', values='requests').plot()
        # plt.legend(loc='best')
        # plt.xlabel('Year')
        # df_year.plot()
        # plt.show()

        # # Requests per Borough Over Months
        # df_month = df.groupby(['Month', 'Borough']).sum().reset_index()
        # df_month.pivot(index='Month', columns='Borough', values='requests').plot()
        # plt.legend(loc='best')
        # plt.xlabel('Month')
        # plt.show()

        # TODO: 3D
        ##########################################################################################
        print("Two dimensional analysis")
        # complaint tyoe in location over time (gif)

    def gioheatmap(self):
        from ipyleaflet import Map, Heatmap
        from random import uniform
        m = Map(center=(0, 0), zoom=2)

        heatmap = Heatmap(
            locations=[[uniform(-80, 80), uniform(-180, 180), uniform(0, 1000)] for i in range(1000)],
            radius=20
        )

        m.add_layer(heatmap)
        print(type(m))
        m

    def geo_convert(self):
        file = 'data/weatherPerMonthLatLon.csv'
        # Latitude,Longitude,Year,Month,MeanTemp,WindSpeed,rain
        from geopy.geocoders import Nominatim
        geolocator = Nominatim()

        fout = open('data/parsedWeatherPerMonthLatLon.csv', 'w')
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





        # borough = []
        # loc = ['40.764141, -73.954430', '40.78993085, -73.9496098723']
        # for l in loc:
        #     sub = str(geolocator.reverse(l))
        #     borough.append(sub.split(', ')[2])
        # borough

    def plot_weather_data(self):
        # TODO
        # changes over time
        # correlation among each attributes
        pass

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

    def stats(self):
        df = pd.read_csv('data/ResolutionDescriptionCounts.csv', delimiter=',')
        print(df.describe())


def main():
    self = DataExploration()
    # self.read_data()
    # self.create_time_series_data()
    # self.time_series_analysis()
    # self.plot_service_data()
    # self.gioheatmap()
    self.geo_convert()
    # self.stats()


if __name__ == '__main__':
    main()