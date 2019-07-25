from __future__ import absolute_import

import pytz
from energyworx.rules.rule_util import RuleUtil
from energyworx_public.rule import AbstractRule, RuleResult, PREPARE_DATASOURCE_IDS_KEY
import logging

logger = logging.getLogger()


class ScalarTimeShiftForecast(AbstractRule):
    """
    This rule work with weather normed interval data or profiled interval data, calculate the factor
    (weather normed usage/sum_profile_usage or delivery_scalar_usage/sum_profile_usage), and timeshift the factor by year
    until the forecast_end date. Forecast interval usage would be profile-usage * factors.

    """

    def prepare_context(self, datasource_id=None, datasource_format='{metadata.discocode}{_}{metadata.profileclass}',
                        channel_classifier="PROFILE", **kwargs):

        if not datasource_id:
            datasource_id = self.get_profile_datasource_id(datasource_format)
        profile_datasource = self.load_datasource(datasource_id)
        usage_channel = profile_datasource.get_channel_by_classifier(channel_classifier)
        if usage_channel is None:
            usage_channel = profile_datasource.get_channel_by_classifier('Usage')

        delivery_scalar_channel_id = self.datasource.get_channel_by_classifier('DELIVERY_SCALAR').id
        delivery_scalar_heartbeat_channel_id = self.datasource.get_channel_by_classifier(
            'DELIVERY_SCALAR_HEARTBEAT').id

        return {'prepare_datasource_ids': [self.datasource.id, datasource_id],
                'delivery_scalar_channel_id': delivery_scalar_channel_id,
                "delivery_scalar_heartbeat_channel_id": delivery_scalar_heartbeat_channel_id,
                'profile_datasource_id': datasource_id,
                'usage_channel_id': usage_channel.id}

    def apply(self, forecast_till=None, months_forward=None, reset_month_day=None, from_start_of_year=None, **kwargs):
        """
        for profiled data: (not influenced by profiled_scalar channel, start from scratch)
        1) calculate the factors original_scalar/aggregated_profile_usage
        2) backfill the factors to interval heartbeat for each of the scalar periods
        3) time_shift the factors until the forecast_until date
        4) time_shift the profile data until the forecast_until date
        5) forecast = factors * profile_usage

        for weather norm:
        1) aggregate the interval data back to scalar data according to the original scalar read dates,
           weather_normed_scalar (for at least the last 1 year, or 373 days)
        2) calculate the factors weather_normed_scalar/aggregated_profile_usage
        3) backfill the factors to interval heartbeat for each of the scalar periods
        4) time_shift the factors until the forecast_until date
        5) time_shift the profile data until the forecast_until date
        6) forecast = factors * profile_usage

        Args:
            forecast_till:
            months_forward:
            reset_month_day:
            from_start_of_year:
            **kwargs:

        Returns:

        """
        import pandas as pd
        import datetime as dt

        df_interval = self.dataframe[[self.source_column]].copy()
        df_end = df_interval.index[-1]
        one_hour = pd.Timedelta(hours=1)

        # Find the scalar data for at least the last 12 months
        end = self.dataframe.index[-1]
        start = min(end - pd.DateOffset(months=13), self.dataframe.index[0])
        delivery_scalar_channel_id = self.context['delivery_scalar_channel_id']
        delivery_scalar_heartbeat_channel_id = self.context['delivery_scalar_heartbeat_channel_id']
        scalar_df = self.load_side_input(datasource_id=self.datasource.id,
                                         channel_id=delivery_scalar_channel_id,
                                         start=start, end=end)
        if len(scalar_df.columns) > 1:  # Used when loading older ingested datasources
            logger.warning("Found more than 1 channel for DELIVERY_SCALAR")
            # todo: check with Erik the logic of loading more than one columns
            delivery_scalar_classifier = str(self.datasource.get_channel_by_classifier('DELIVERY_SCALAR').classifier_id)
            delivery_scalar_heartbeat_classifier = str(
                self.datasource.get_channel_by_classifier('DELIVERY_SCALAR_HEARTBEAT').classifier_id)
            delivery_scalar_column = 'INGEST:' + delivery_scalar_classifier if 'INGEST:' + delivery_scalar_classifier in scalar_df.columns else 'INGEST:DELIVERY_SCALAR'
            delivery_scalar_heartbeat_column = 'INGEST:' + delivery_scalar_heartbeat_classifier if 'INGEST:' + delivery_scalar_heartbeat_classifier in scalar_df.columns else 'INGEST:DELIVERY_SCALAR_HEARTBEAT'

            scalar_df = scalar_df[[delivery_scalar_column, delivery_scalar_heartbeat_column]]
            scalar_df.columns = ['DELIVERY_SCALAR', 'DELIVERY_SCALAR_HEARTBEAT']
            scalar_df['DELIVERY_SCALAR_HEARTBEAT'] = self.load_side_input(datasource_id=self.datasource.id,
                                                                          channel_id=delivery_scalar_heartbeat_channel_id,
                                                                          start=start, end=end)
            scalar_df = scalar_df.dropna()

            from energyworx.cleansing import Clean
            scalar_df = Clean.calculate_heartbeat(scalar_df, 'DELIVERY_SCALAR_HEARTBEAT')
        else:
            scalar_df.columns = ['DELIVERY_SCALAR']
            scalar_df['DELIVERY_SCALAR_HEARTBEAT'] = self.load_side_input(datasource_id=self.datasource.id,
                                                                          channel_id=delivery_scalar_heartbeat_channel_id,
                                                                          start=start,
                                                                          end=end)

        usage_channel_id = self.context['usage_channel_id']
        profile_datasource_id = self.context['profile_datasource_id']
        timezone = pytz.timezone(self.datasource.timezone)
        forecast_till = self.forecast_length_checker(forecast_till, months_forward, reset_month_day,
                                                     self.datasource.timezone)
        forecast_end = timezone.localize(dt.datetime(forecast_till.year, forecast_till.month, 1)).astimezone(pytz.utc)

        profile_df = self.load_side_input(datasource_id=profile_datasource_id, channel_id=usage_channel_id, start=start,
                                          end=forecast_end)
        if (profile_df is not None) and profile_df.empty or start < profile_df.index[0] or end > profile_df.index[-1]:
            logger.warning(
                "No or not enough profile_df/proxy data found for profile_df/proxy %s. "
                "Attempting to retrieve more than just the requested time period.",
                profile_datasource_id)
            profile_df = self.load_side_input(datasource_id=profile_datasource_id, channel_id=usage_channel_id,
                                              start=dt.datetime(1970, 1, 1, 1, tzinfo=pytz.UTC),
                                              end=dt.datetime(3000, 1, 1, tzinfo=pytz.UTC))
        if len(profile_df.columns) > 1:
            profile_df = profile_df[[profile_df.columns[0]]]
        profile_df.columns = ['Usage']
        local_tz = self.datasource.timezone
        profile_df = RuleUtil.localize_profile(profile_df, local_tz, self.datasource)

        logger.info('length scalar_df = %s', len(scalar_df.index))
        logger.info('length profile_df = %s', len(profile_df.index))

        # Aggregate the interval weather normed value back to the same scalar_df read dates
        scalar_df = self.calculate_start_column(scalar_df)
        # Check if weather norm is performed
        if any(['WEATHER_NORM' in x for x in self.dataframe.columns]):
            for index, row in scalar_df.iterrows():
                start = row['START'] + pd.Timedelta('1h')
                end = index
                scalar_df.loc[index, 'normed_scalar_usage'] = df_interval.loc[start: end, self.source_column].sum()
        else:  # no weather norm
            scalar_df.loc[:, 'normed_scalar_usage'] = scalar_df['DELIVERY_SCALAR']

        # Prepare the profile data
        forecast_start = df_end + one_hour
        if from_start_of_year:
            forecast_start = timezone.localize(dt.datetime(df_end.year, 1, 1)).astimezone(pytz.utc) + one_hour

        # APPEND PROFILE
        if profile_df.index[-1] < forecast_end:
            logger.info('Profile data reaches till {} while forecast till date is {}. Timeshifting the profile.'.format(
                profile_df.index[-1], forecast_end))
            profile_df = self.append_to_profile(profile_df, forecast_end)
        # PREPEND PROFILE
        if profile_df.index[0] > min(forecast_start, scalar_df.loc[scalar_df.index[0], 'START']):
            logger.info(
                "Profile data does not have data early enough a year ago or the first date of forecast, prepend it")
            profile_df = self.prepend_to_profile(profile_df,
                                                 min(forecast_start, scalar_df.loc[scalar_df.index[0], 'START']))
        profile_df = profile_df.loc[: forecast_end, ['Usage']]

        # Prepare the factors_df for the last one year
        factors_df = self.prepare_factors_df(profile_df, scalar_df, 'normed_scalar_usage')
        # Start time-shifting
        factors_df = self.time_shift_factors_df(factors_df, forecast_start, forecast_end)

        rule_name = 'scalar_time_shift_forecast'
        profile_df.loc[:, rule_name] = profile_df.loc[:, 'Usage'] * factors_df.loc[:, 'factors']
        # forecast_df, use the profiled data for the part where channel PROFILED_SCALAR or WEATHER_NORM is available
        forecast_df = self.dataframe.loc[forecast_start: forecast_end, self.source_column]
        forecast_df.columns = [rule_name]
        forecast_df = forecast_df.combine_first(profile_df.loc[forecast_start: forecast_end, rule_name])
        forecast_df = forecast_df.to_frame(rule_name)
        nr_missing_values = len(forecast_df.loc[pd.isnull(forecast_df.loc[:, rule_name])])
        if nr_missing_values > 0:
            logger.warning("Forecast contains {} missing values, use back fill.".format(nr_missing_values))
            forecast_df = forecast_df.fillna(method='bfill')
            logger.info("After bfill, forecast contains %s missing values.",
                        len(profile_df.loc[pd.isnull(profile_df[rule_name])]))
        else:
            logger.info("Forecast contains no missing values")
        # correct negative values
        forecast_df.loc[forecast_df[rule_name] <= 0, rule_name] = 0.01
        if len(forecast_df[forecast_df.index.duplicated()]) > 0:
            forecast_df = forecast_df[~forecast_df.index.duplicated()]

        return RuleResult(result=forecast_df)

    def get_profile_datasource_id(self, datasource_format='{metadata.discocode}{_}{metadata.profileclass}'):
        """ Parses the datasource_format by splitting it based on {} characters.
        E.g. {general.utility}{__}{profile.name} is parsed such that you get
        [general.utility, __, profile.name]. This is then used to retrieve the
        necessary information from the datasource's tags. Note that property keys
        on tags have to be denoted as <tag_name>.<property_key> (thus separated
        by a dot).

        Args:
            datasource_format (str): format of the datasource to retrieve.

        Returns:
            str
        """
        if datasource_format is None:
            datasource_format = '{metadata.discocode}{_}{metadata.profileclass}'
        datasource_name_pieces = datasource_format.replace('}', '').replace('{', '', 1).split('{')
        datasource_id = []
        for format_piece in datasource_name_pieces:
            name_piece_split = format_piece.split('.')
            if len(name_piece_split) == 1:
                # this is no tag_name property_key pair but a separator
                datasource_id.append(format_piece)
                continue
            # name_piece_split should be of length two here
            tag_name, property_key = name_piece_split
            if property_key.endswith('_'):
                property_key = property_key.replace('_', '')
            current_tag, _, _ = self.datasource.get_tag(tag_name, latest_version=True)
            if not current_tag:
                logger.warning("No '%s' tag available on datasource %s. No profile datasouce id will be returned." % (
                    tag_name, self.datasource.id))
                continue
            raveled_tag_properties = {kv.key: kv.value for kv in current_tag.properties}
            name_piece = raveled_tag_properties.get(property_key)
            # Overwrite discocode with "ALL_ERCOT" for market ERCOT.
            if property_key == "discocode" and raveled_tag_properties.get('market') == "ERCOT":
                name_piece = 'ALL_ERCOT'
            if name_piece:
                datasource_id.append(name_piece)
            else:
                logger.warning("Tag '%s' on datasource %s does not have property key %s" % (
                    tag_name, self.datasource.id, property_key))
        return "".join(datasource_id)

    @staticmethod
    def forecast_length_checker(forecast_till, months_forward, reset_month_day, time_zone):
        import datetime as dt
        import pandas as pd

        if forecast_till and months_forward:
            logger.error("Cannot have both forecast_till and months_forward as input. Returning no forecast.")
            return None
        elif forecast_till:
            forecast_till = pd.Timestamp(forecast_till)
        elif months_forward:
            today = dt.datetime.now()
            forecast_ref = dt.datetime(today.year, today.month, 1)
            if reset_month_day and today.day >= reset_month_day:
                months_forward += 1
            forecast_till = forecast_ref + pd.DateOffset(months=months_forward)
            account_timezone = pytz.timezone(time_zone)
            forecast_till = account_timezone.localize(forecast_till)
        else:
            logger.error("No forecast length configured. Either configure forecsat_till or months_forward.")
            return None
        return forecast_till

    @staticmethod
    def append_to_profile(profile_df, append_till):
        import pandas as pd

        one_interval = pd.Timedelta(hours=1)
        # appending timeshifted profile data to existing profile data
        year = 1
        while profile_df.index[-1] < append_till:
            go_back_time = pd.Timedelta(days=364)  # 364 is divisible by 7, this will shift the peak only 1 day per year
            if (year % 6 == 0 and year > 0):  # Every 6 year, a 'shift' of 7 days will occur
                go_back_time = pd.Timedelta(days=371)
            date_last_value = profile_df.index[-1]
            cast = profile_df.loc[date_last_value - go_back_time + one_interval: date_last_value].copy()
            cast.index = cast.index + go_back_time
            year += 1
            profile_df = pd.concat([profile_df, cast], join='inner', axis=0)
        return profile_df

    @staticmethod
    def prepare_factors_df(profile_df, scalar_df, data_column):
        """This function will create factors_df for the last one year (compared with the last scalar reads)
        Args:
            profile_df: a dataframe containing profile data
            scalar_df: a dataframe of scalar df_interval
            data_column: (str) the column of scalar_df that has scalar read

        Returns:
            profile_df, df
            factors_df, df
            correction factor: float

        """
        import pandas as pd
        one_hour = pd.Timedelta(hours=1)
        one_year = pd.DateOffset(years=1)
        start_dates = scalar_df['START']
        stop_dates = scalar_df.index
        # TODO[BP] found out when start_dates[0] has a timezone and when it does not, this is a fallback scenario for when it does not.
        if not start_dates[0].tz:
            start_dates[0] = start_dates[0].tz_localize(pytz.UTC)
        # Construct the factors dataframe, closed='right' make sure it starts on 01:00 hour
        dates = pd.date_range(start_dates[0], stop_dates[-1], freq='1h', closed='right', tz=pytz.UTC)

        factors_df = pd.DataFrame(index=dates, columns=['factors'])
        for index, row in scalar_df.iterrows():
            # The 00:00 hour of a scalar period belongs to the last period, plus 1 hour to exclude that hour.
            prof_start = row['START'] + pd.Timedelta('1h')
            prof_end = index
            scalar_usage = scalar_df.loc[index, data_column]
            profile_usage = profile_df.loc[prof_start: prof_end, 'Usage'].sum()
            if profile_usage == 0:
                logger.warning('There are no profile usage between start %s and end %s', prof_start, prof_end)
                continue
            factors_df.loc[prof_start: prof_end, 'factors'] = scalar_usage / profile_usage

        # check if the factors_df is less than a year, then create the index at least make sure it's more than a year
        if factors_df.index[0] > stop_dates[-1] - pd.DateOffset(years=1):
            logger.warning("Scalar data is shorter than a year")
            dates = pd.date_range(stop_dates[-1] - pd.DateOffset(years=1), factors_df.index[-1], freq='1h',
                                  closed='right', tz=pytz.UTC)
            factors_df = factors_df.reindex(index=dates)
        logger.info('factors_df has %s missing values in the past one year, fill with correction factors '
                    'for timeshifting', len(factors_df[pd.isnull(factors_df['factors'])]))
        # note that value_counts creates a series with counts (and an index with the factors!)
        factors_with_weights = factors_df['factors'].value_counts(sort=False,
                                                                  dropna=True).reset_index()
        correction_factor = (factors_with_weights['factors'] * factors_with_weights['index']).sum() / \
                            factors_with_weights['factors'].sum()
        logger.info("Correction factor: {} used to fill the missing values".format(correction_factor))

        # fill missing values with correction factor
        # factors_df.loc[:, 'factors'] = factors_df.loc[:, 'factors'].fillna(method='bfill')
        factors_df.loc[pd.isnull(factors_df['factors']), 'factors'] = correction_factor
        logger.info('factors_df has %s missing values', len(factors_df[pd.isnull(factors_df['factors'])]))
        if any(pd.isnull(factors_df.loc[:, 'factors'])):
            factors_df.loc[:, 'factors'] = factors_df.loc[:, 'factors'].fillna(method='bfill')
            logger.info('factors_df still has %s missing values', len(factors_df[pd.isnull(factors_df['factors'])]))
        logger.info('Before to_numeric')
        factors_df['factors'] = pd.to_numeric(factors_df['factors'], errors='coerce')
        logger.info('After to_numeric')
        factors_df.bfill(inplace=True)
        return factors_df.loc[factors_df.index[-1] - one_year + one_hour:]

    @staticmethod
    def time_shift_factors_df(factors_df, forecast_start, forecast_end):
        """

        Args:
            factors_df:
            forecast_start:
            forecast_end:

        Returns:

        """
        import pandas as pd
        one_year = pd.DateOffset(years=1)
        factors_cast_df = factors_df.copy()
        while factors_df.index[-1] < forecast_end:
            factors_cast_df.index += one_year
            # remove duplicate timestamps (this occurs if we shift from a leap year to a normal year)
            factors_cast_df = factors_cast_df[~factors_cast_df.index.duplicated(keep='first')]
            factors_cast_df.sort_index(
                inplace=True)  # This is to make sure the index is monotonic increasing or decreasing
            # add timestamps (this occurs if we shift from a normal year to a leap year)
            all_dates = pd.date_range(factors_cast_df.index[0], factors_cast_df.index[-1], freq='1h', tz='UTC')
            factors_cast_df = factors_cast_df.reindex(all_dates)
            # Keep consistant with the fill method with profile_scalar_to_interval
            factors_cast_df.loc[:, 'factors'] = factors_cast_df.loc[:, 'factors'].fillna(method='bfill')
            # factors_cast_df.loc[pd.isnull(factors_cast_df['factors']), 'factors'] = correction_factor
            factors_df = pd.concat([factors_df, factors_cast_df], axis=0)
        # remove duplicte
        factors_df = factors_df[~factors_df.index.duplicated(keep='first')]
        return factors_df.loc[forecast_start: forecast_end]

    @staticmethod
    def prepend_to_profile(profile_df, prepend_till):
        """

        Args:
            profile_df(pd.DataFrame):
            prepend_till(pd.Timestamp):

        Returns:

        """
        import pandas as pd
        one_interval = pd.Timedelta(hours=1)
        # appending timeshifted profile data to existing profile data
        year = 1
        while profile_df.index[0] > prepend_till:
            go_forw_time = pd.Timedelta(days=364)  # 364 is divisible by 7, this will shift the peak only 1 day per year
            if (year % 6 == 0 and year > 0):  # Every 6 year, a 'shift' of 7 days will occur
                go_forw_time = pd.Timedelta(days=371)
            date_first_value = profile_df.index[0]
            cast = profile_df.loc[date_first_value: date_first_value + go_forw_time - one_interval].copy()
            cast.index = cast.index - go_forw_time
            year += 1
            profile_df = pd.concat([cast, profile_df], join='inner', axis=0)
        return profile_df

    def calculate_start_column(self, df):
        """ Calculates and adds a START column to the dataframe.
        The START column is calculated based on the index and the
        heartbeat column.

        Args:
            df (pd.DataFrame):

        Returns:
            pd.DataFrame
        """

        import pandas as pd
        import numpy as np
        df = df[np.isfinite(df['DELIVERY_SCALAR_HEARTBEAT'])]
        df.loc[:, 'START'] = [x - pd.Timedelta(seconds=y) for (x, y) in zip(df.index, df['DELIVERY_SCALAR_HEARTBEAT'])]
        logger.info("scalar_time_shift_forecast: calculated START column with heartbeat %s", df)
        return df

    def _calculate_start_column(self, df):
        """ Calculates and adds a START column of the dataframe, by using the difference between index

        Args:
            df (pd.DataFrame):

        Returns:
            pd.DataFrame
        """
        import pandas as pd
        df.loc[:, 'index'] = df.index
        df['DELIVERY_SCALAR_HEARTBEAT'] = df.DELIVERY_SCALAR_HEARTBEAT.fillna(0)
        if len(df) >= 2:
            df.loc[df.index[1]:, 'START'] = df.loc[df.index[1]:, 'index'] - df.loc[:, 'index'].diff()[1:]
            df.loc[df.index[0], 'START'] = df.index[0] - pd.Timedelta(
                seconds=df.loc[df.index[0], 'DELIVERY_SCALAR_HEARTBEAT'])
        else:
            df.loc[df.index[0], 'START'] = df.index[0] - pd.Timedelta(
                seconds=df.loc[df.index[0], 'DELIVERY_SCALAR_HEARTBEAT'])
        del df['index']
        return df
