__author__ = 'bobo'


class Constants:
    def __init__(self):

        self.f_data_weather = 'data/weather_NY_2010_2018Nov.csv'
        self.f_regression_data = 'data/RegressionDailyData5BoroughsData.csv'

        self.f_raw_time_series_daily = 'data/RequestsPerDay.csv'
        self.f_parsed_time_series_daily = 'data/ParsedRequestsPerDay.csv'

        self.f_raw_time_series_monthly = 'data/RequestsPerMonth.csv'
        self.f_parsed_time_series_monthly = 'data/ParsedRequestsPerMonth.csv'

        self.tuning_nnt = 'data/tuning_nnt.csv'
        self.tuning_rf = 'data/tuning_rf.csv'
        self.tuning_gb = 'data/tuning_gb.csv'
