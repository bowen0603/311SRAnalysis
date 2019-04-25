__author__ = 'bobo'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from folium.plugins import HeatMap
import geopandas as gpd
from constants import Constants


class Analysis:
    def __init__(self):

        self.const = Constants()

        self.f_raw_requests_per_day = 'data/RequestsPerDay.csv'
        self.f_parsed_requests_per_day = 'data/ParsedRequestsPerDay.csv'
        self.f_raw_requests_per_month = 'data/RequestsPerMonth.csv'
        self.f_parsed_requests_per_month = 'data/ParsedRequestsPerMonth.csv'
        self.f_parsed_requests_per_month = 'data/ParsedRequestsOverTime.csv'

    def service_requests_analysis(self):
        ##########################################################################################
        print("One dimensional analysis")
        # Request over time
        df = pd.read_csv(self.f_parsed_requests_per_month, delimiter=',')
        df.plot.line(x='date', y='requests', title='Service Requests over Time', marker='o')
        plt.show()

        # Complaint type bar chart => Probably better to just use list and categorization
        df = pd.read_csv('data/complainTypeSort.csv', delimiter=',')
        topN = 20
        df = df.head(topN)
        df.plot.bar(x='ComplaintType', y='requests', title='Top {} Complaint Type Distribution'.format(topN), rot=0)
        plt.xticks(rotation=45)
        plt.show()

        # location/Borough bar chart
        df = pd.read_csv('data/RequestsPerBorough.csv', delimiter=',')
        df.set_index('Borough')
        df.plot.pie(y='requests', autopct='%.2f', labels=df['Borough'])
        plt.legend(loc='best')
        plt.show()

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

        ##########################################################################################
        print("Two dimensional analysis")

        # Requests per Complaint Type Over Years
        df = pd.read_csv('data/RequestsOverTimeOverAreaOverType.csv', delimiter=',')
        df_year = df.groupby(['Year', 'ComplaintType']).sum().reset_index()
        df_year.pivot(index='Year', columns='ComplaintType', values='requests').plot()
        plt.legend(loc='best')
        plt.xlabel('Year')
        df_year.plot()
        plt.show()

        # Requests per Borough Over Months
        df_month = df.groupby(['Month', 'ComplaintType']).sum().reset_index()
        df_month.pivot(index='Month', columns='ComplaintType', values='requests').plot()
        plt.legend(loc='best')
        plt.xlabel('Month')
        plt.show()

        # Requests per Borough per Complaint Type
        df = pd.read_csv('data/RequestsOverTimeOverAreaOverType.csv', delimiter=',')
        df_borough = df.groupby(['Borough', 'ComplaintType']).sum().reset_index()
        df_borough.pivot("Borough", "ComplaintType", "requests").plot(kind='bar')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        plt.show()

        # # Requests per Borough Over Time
        df_time = pd.read_csv('data/RequestsPerBoroughOverTimeParsed.csv', delimiter=',')
        df_time.pivot(index='Time', columns='Borough', values='requests').plot()
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.show()

        # Requests per Borough Over Years
        df = pd.read_csv('data/RequestsPerBoroughOverTime.csv', delimiter=',')
        df_year = df.groupby(['Year', 'Borough']).sum().reset_index()
        df_year.reset_index().plot.line(x=df_year.index, y='requests', title='Requests Over Years')
        df_year.pivot(index='Year', columns='Borough', values='requests').plot()
        plt.legend(loc='best')
        plt.xlabel('Year')
        df_year.plot()
        plt.show()

        # Requests per Borough Over Months
        df_month = df.groupby(['Month', 'Borough']).sum().reset_index()
        df_month.pivot(index='Month', columns='Borough', values='requests').plot()
        plt.legend(loc='best')
        plt.xlabel('Month')
        plt.show()

    def gioheatmap(self):
        congr_districts = gpd.read_file('zip://' + 'cb_2015_us_cd114_20m.zip')
        congr_districts.crs = {'datum': 'NAD83', 'ellps': 'GRS80', 'proj': 'longlat', 'no_defs': True}

        district23 = congr_districts[congr_districts.GEOID == '3623']  # 36 = NY, 23 = District
        # convert it to the projection of our folium openstreetmap
        district23 = district23.to_crs({'init': 'epsg:3857'})

        for_map = pd.read_csv('data/blockDriveWay.csv', delimiter=',').dropna()
        max_amount = float(for_map['requests'].max())

        hmap = folium.Map(location=[40.7308268, -73.9995207], zoom_start=7)

        hm_wide = HeatMap(list(zip(for_map.Latitude.values, for_map.Longitude.values, for_map.requests.values)),
                          min_opacity=0.2, max_val=max_amount, radius=17, blur=15,max_zoom=1)

        folium.GeoJson(district23).add_to(hmap)
        hmap.add_child(hm_wide)
        hmap.save('heatmap.html')

    def stats(self):
        df = pd.read_csv('data/ResolutionDescriptionCounts.csv', delimiter=',')
        print(df.describe())

    def corr_mtx(self, df, dropDuplicates=True):

        # Compute the correlation matrix
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
            sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        else:
            sns.heatmap(df, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xticks(rotation=45)
        plt.show()

    def missing_values(self):
        df = pd.read_csv(self.const.f_data_weather, delimiter=',')
        for feature in df.columns:
            idx = [feature]
            misses = np.where(pd.isnull(df[idx]))
            print(feature, 1.0 * len(misses[0]) / df.shape[0])

    def plot_histogram(self):
        df = pd.read_csv(self.const.f_data_weather, delimiter=',')
        feature = ['Day', 'Month', 'Year', 'WindSpeed', 'SnowDepth', 'SnowIce', 'MaxSustainedWind',
                   'Rain', 'MeanTemp', 'MinTemp', 'MaxTemp', 'Percipitation', 'Gust', 'DewPoint']
        df_hist = df[feature]
        df_hist.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
        plt.show()

    @staticmethod
    def visual_outliers():
        df = pd.read_csv('data/RegressionDailyData.csv', delimiter=',')

        features = ['Day', 'Month', 'WindSpeed', 'SnowIce', 'Rain', 'MeanTemp', 'Percipitation', 'requests']
        df = df[features]
        for i in range(0, len(df.columns), 4):
            print(i, i+4)
            sns.pairplot(data=df,
                         x_vars=df.columns[i:i + 4],
                         y_vars=['requests'])
        plt.show()

    def correlation_analysis(self):
        df = pd.read_csv(self.const.f_data_regression, delimiter=',')
        print(df.head(5))

        feature = ['Day', 'Month', 'WindSpeed', 'SnowIce', 'Rain', 'MeanTemp',
                    'Percipitation', 'BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND', 'requests']
        self.corr_mtx(df[feature], True)


def main():
    self = Analysis()
    self.service_requests_analysis()
    self.gioheatmap()
    self.stats()
    self.correlation_analysis()
    self.visual_outliers()
    self.plot_histogram()


if __name__ == '__main__':
    main()