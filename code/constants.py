__author__ = 'bobo'

class Constants:
    def __init__(self):
        self.f_raw_time_series_daily = 'data/RequestsPerDay.csv'
        self.f_parsed_time_series_daily = 'data/ParsedRequestsPerDay.csv'

        self.f_raw_time_series_monthly = 'data/RequestsPerMonth.csv'
        self.f_parsed_time_series_monthly = 'data/ParsedRequestsPerMonth.csv'

        self.tuning_lg = 'data/tuning_lg.csv'
        self.tuning_nnt = 'data/tuning_nnt.csv'
        self.tuning_rf = 'data/tuning_rf.csv'
        self.tuning_gb = 'data/tuning_gb.csv'
