from __future__ import absolute_import

from energyworx.rules.base_rule import AbstractRule
from energyworx.domain import RuleResult, Detector, KeyValue
import logging
import pytz
import datetime as dt

logger = logging.getLogger()


class IntervalTimeShiftForecast(AbstractRule):
    """ Time shift forecast with optional weather adjustments. """
    AIRTEMPERATURE = 'AIRTEMPERATURE'
    HOURLY = 'hourly'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    DETECTOR_NAME = 'is_weather_sensitive'

    def apply(self,
              forecast_till=None,
              months_forward=None,
              reset_month_day=None,
              from_start_of_year=False,
              output_granularity="own_granularity",
              **kwargs):
        """ Produces a forecast based on time shifting the last full year
        of data multiple times. It takes into account the day of the week
        when time shifting such that mondays in the time shift remain mondays.
        There's also the option to apply weather normalization.

        Args:
            forecast_till (str/datetime):
            months_forward (int):
            reset_month_day (int):
            from_start_of_year (boolean):
            output_granularity (string): selected from a certain list of values

        Returns:
            RuleResult(result=pd.DataFrame, metadata=list)
        """
        import pandas as pd
        source_df = self.dataframe.loc[self.data_filter, [self.source_column, self.source_heartbeat_column]].dropna(how='any')
        one_interval = pd.Timedelta(seconds=source_df[self.source_heartbeat_column].iloc[-1])
        timezone = pytz.timezone(self.datasource.timezone)
        df_end = source_df.index[-1]

        forecast_till = self.forecast_length_checker(forecast_till, months_forward, reset_month_day, self.datasource.timezone)
        forecast_start = df_end + one_interval
        if from_start_of_year:
            forecast_start = timezone.localize(dt.datetime(df_end.year, 1, 1)).astimezone(pytz.utc) + one_interval
        # Make sure the data is enough long to time shift
        source_df = self._prepend_to_df(source_df, self.source_heartbeat_column, forecast_from=forecast_start)
        # The output column should be the same than source column
        output_column = 'interval_time_shift_forecast'
        source_df.loc[:, output_column] = source_df.loc[:, self.source_column]
        # time shift
        source_df = self.append(source_df, forecast_till, source_df.loc[source_df.index[-1], self.source_heartbeat_column])
        source_df = source_df.loc[forecast_start: forecast_till, :]

        dict_of_output_granularities = dict(hourly=('H', pd.Timedelta(hours=1), pd.DateOffset(hours=1)),
                                            daily=('D', pd.Timedelta(days=1), pd.DateOffset(days=1)),
                                            monthly=('M', pd.Timedelta(days=31), pd.DateOffset(days=1)),
                                            annual=('A', pd.Timedelta(days=365), pd.DateOffset(days=1)))
        output_granularity_timedelta, date_offset_correction = None, None
        if output_granularity == "own_granularity":
            pass
        elif output_granularity not in dict_of_output_granularities.keys():
            raise ValueError("Parameter output_granularity should be one of " + str(dict_of_output_granularities.keys()) + " or should be 'own_granularity' indicating no time grouping should be done.")
        else:
            output_granularity, output_granularity_timedelta, date_offset_correction = dict_of_output_granularities.get(output_granularity)

        if output_granularity and date_offset_correction and output_granularity_timedelta and one_interval < output_granularity_timedelta:
            resample_df = source_df[[output_column]]
            # temporary shift index so that time grouper will capture correct groups
            resample_df.index -= one_interval
            resample_df = resample_df.tz_convert(self.datasource.timezone)
            resample_df = resample_df.groupby(pd.Grouper(freq=output_granularity)).sum()
            # revert timestamp changes
            resample_df = resample_df.tz_convert(pytz.UTC)
            resample_df.index += date_offset_correction
            # remove old (non adjusted) forecast column
            del source_df[output_column]
            source_df = source_df.loc[:source_df.last_valid_index()]
            # add to new granularity adjusted forecast to df
            source_df = source_df.combine_first(resample_df)

        extra_sequence_data = [{"name": "total", "value": source_df[output_column].sum()}, {"name": "start_date", "value": forecast_start}]
        # drop the old destination column (which was a copy of the source_column) and add the new (forecast) destination column
        source_df = pd.concat([self.dataframe, source_df[[output_column]]], axis=1)
        return RuleResult(result=source_df, metadata=extra_sequence_data)

    @staticmethod
    def forecast_length_checker(forecast_till, months_forward, reset_month_day, time_zone):
        """

        Args:
            forecast_till: (str)
            months_forward: (int)
            reset_month_day: (int)
            time_zone: (str)

        Returns: pandas.tslib.Timestamp

        """
        from datetime import datetime as datetime
        import pandas as pd
        account_timezone = pytz.timezone(time_zone)
        if forecast_till and months_forward:
            logger.error("Cannot have both forecast_till and months_forward as input. Returning no forecast.")
            return None
        elif forecast_till:
            forecast_till = pd.Timestamp(forecast_till)
        elif months_forward:
            today = datetime.now()
            forecast_ref = datetime(today.year, today.month, 1)
            if reset_month_day and today.day >= reset_month_day:
                months_forward += 1
            forecast_till = forecast_ref + pd.DateOffset(months=months_forward)
        else:
            logger.error("No forecast length configured. Either configure forecsat_till or months_forward.")
            return None
        forecast_till = account_timezone.localize(forecast_till)
        return forecast_till

    @staticmethod
    def prepend(df, prepend_till, hb):
        """

        Args:
            df: (pd.DataFrame)
            prepend_till: ((pd.Timestamp)
            hb: (int)

        Returns: (pd.DataFrame)

        """
        import pandas as pd
        one_interval = pd.Timedelta(seconds=hb)
        # prepending timeshifted data to existing data
        year = 1
        while df.index[0] > prepend_till:
            go_forw_time = pd.Timedelta(days=364) # 364 is divisible by 7, this will shift the peak only 1 day per year
            if (year % 6 == 0 and year > 0):  # Every 6 year, a 'shift' of 7 days will occur
                go_forw_time = pd.Timedelta(days=371)
            date_first_value = df.index[0]
            cast = df.loc[date_first_value: date_first_value + go_forw_time - one_interval].copy()
            cast.index = cast.index - go_forw_time
            year += 1
            df = pd.concat([cast, df], join='inner', axis=0)
        return df

    @staticmethod
    def append(df, append_till, hb):
        """

        Args:
            df: (pd.dataframe)
            append_till: ((pd.Timestamp)
            hb: (int)

        Returns: (pd.DataFrame)

        """
        import pandas as pd
        one_interval = pd.Timedelta(seconds=hb)
        # appending timeshifted data to existing  data
        year = 1
        while df.index[-1] < append_till:
            go_back_time = pd.Timedelta(days=364) # 364 is divisible by 7, this will shift the peak only 1 day per year
            if (year % 6 == 0 and year > 0):  # Every 6 year, a 'shift' of 7 days will occur
                go_back_time = pd.Timedelta(days=371)
            date_last_value = df.index[-1]
            cast = df.loc[date_last_value - go_back_time + one_interval: date_last_value].copy()
            cast.index = cast.index + go_back_time
            year += 1
            df = pd.concat([df, cast], join='inner', axis=0)
        return df

    def _prepend_to_df(self, df, heartbeat_column_name, minimum_days=373, forecast_from=None):
        """ Prepends months or weeks of data to the dataframe to make sure
        there is enough data to create a time shift forecast.

        Args:
            df (pd.DataFrame):
            heartbeat_column_name (str):
            minimum_days (int):
            forecast_from (pd.Timestamp):

        Returns:
            pd.DataFrame
        """
        import numpy as np
        import pandas as pd
        one_interval = pd.Timedelta(seconds=df[heartbeat_column_name].iloc[0])
        one_week = pd.Timedelta(days=7)
        min_days = pd.Timedelta(days=minimum_days)
        week_multiplier = 4
        if df.index[-1] - df.index[0] < (week_multiplier + 1) * one_week:
            logger.info("Very short data, falling back to prepending one week of data at a time.")
            week_multiplier = 1
        prepend_period = week_multiplier * one_week
        nr_days_to_add_option1 = min_days - (df.index[-1] - df.index[0])
        nr_days_to_add_option2 = df.index[0] - forecast_from
        nr_days_to_add = max(nr_days_to_add_option1, nr_days_to_add_option2)
        # calculate the number of casts needed to prepend to the dataframe
        nr_casts_needed = int(np.ceil(nr_days_to_add / prepend_period))
        cast = df.loc[: df.index[0] + prepend_period - one_interval, [self.source_column]]
        # create the casts and adjust the index accordingly
        casts = [cast.copy() for i in range(nr_casts_needed)]
        for i, cast in enumerate(casts, start=1):
            cast.index = cast.index - i * prepend_period
        casts.append(df)
        # TODO [VG]: when pandas version is upgraded to 0.23.0, change sort_index to parameter in concat function
        df = pd.concat(casts, axis=0).sort_index()
        return df
