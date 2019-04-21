__author__ = 'bobo'

import numpy as np
import pandas as pd

from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

class Analysis:
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

    def plot_weather_data(self):
        # TODO
        # changes over time
        # correlation among each attributes
        pass

    def stats(self):
        df = pd.read_csv('data/ResolutionDescriptionCounts.csv', delimiter=',')
        print(df.describe())

    def corr_mtx(self, df, dropDuplicates=True):

        # Compute the correlation matrix
        # corr = df.corr()

        # # Generate a mask for the upper triangle
        # mask = np.zeros_like(corr, dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True
        #
        # # Set up the matplotlib figure
        # f, ax = plt.subplots(figsize=(11, 9))
        #
        # # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        #
        # # Draw the heatmap with the mask and correct aspect ratio
        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
        #             square=True, linewidths=.5, cbar_kws={"shrink": .5})

        # Your dataset is already a correlation matrix.
        # If you have a dateset where you need to include the calculation
        # of a correlation matrix, just uncomment the line below:
        df = df.corr()

        # Exclude duplicate correlations by masking uper right values
        if dropDuplicates:
            mask = np.zeros_like(df, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

        # Set background color / chart style
        sns.set_style(style = 'white')

        # Set up  matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Add diverging colormap from red to blue
        cmap = sns.diverging_palette(250, 10, as_cmap=True)

        # Draw correlation plot with or without duplicates
        if dropDuplicates:
            sns.heatmap(df, mask=mask, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        else:
            sns.heatmap(df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.show()

    def correlation_analysis(self):
        df_raw = pd.read_csv('data/RegressionData.csv', delimiter=',')
        print(df_raw.head(5))
        idx_cols = ['requests', 'f1', 'f2', 'f3', 'f4']
        df = df_raw[idx_cols]
        print(df.head(5))
        self.corr_mtx(df, False)



def main():
    self = Analysis()
    # self.read_data()
    # self.create_time_series_data()
    # self.time_series_analysis()
    # self.plot_service_data()
    # self.gioheatmap()
    # self.stats()
    self.correlation_analysis()


if __name__ == '__main__':
    main()