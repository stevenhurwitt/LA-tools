from __future__ import absolute_import
import logging
import datetime as dt
from energyworx.rules.rule_util import RuleUtil
from energyworx.domain import DictWrapper
from energyworx_public.domain import FlowCancelException
from energyworx_public.rule import PREPARE_DATASOURCE_IDS_KEY, RuleResult, AbstractRule
import pytz
LOGGER = logging.getLogger()


class ProfileScalarToInterval(AbstractRule):
    """
    This rule 'disaggregate' the scalar data to interval data by profile.
    The partial month on both end will be complete first.

    """
    AIRTEMPERATURE = 'AIRTEMPERATURE'

    def prepare_context(self, datasource_id=None, channel_classifier=None, **kwargs):
        """Retrieving profile and weather data.

        Args:
            datasource_id (str): used to override the default profile/proxy to forecast with
            channel_classifier (str): channel classifier that belongs to the profile/proxy
            **kwargs:

        Returns:

        """
        profile_datasource_id = datasource_id or self.get_profile_datasource_id_with_tag()
        # check for profile/poxy data
        if profile_datasource_id is None:
            raise FlowCancelException("No profile/proxy datasource found on datasource {}".format(self.datasource.id))

        profile_datasource = self.load_datasource(profile_datasource_id)
        if profile_datasource:
            usage_channel = profile_datasource.get_channel_by_classifier(channel_classifier)
            if usage_channel is None:
                usage_channel = profile_datasource.get_channel_by_classifier('Usage')  # Use older channel classifier
        else:
            raise FlowCancelException("No profile/proxy datasource found for {}".format(datasource_id))

        return {'prepare_datasource_ids': [profile_datasource_id],
                'profile_datasource_id': profile_datasource_id,
                'profile_channel_id': usage_channel.id}

    def apply(self, datasource_id=None, channel_classifier=None, **kwargs):
        """

        Args:
            datasource_id (str):
            channel_classifier (str):

        Returns:

        """
        import pandas as pd

        local_tz = self.datasource.timezone
        data_frame = self.calculate_start_column(self.dataframe)
        df_end_date, df_start_date, new_end, new_start = self.get_new_start_end_date(data_frame)
        if df_end_date < df_start_date:
            raise FlowCancelException("End date ({}) of scalar data is before start({}) for {}".format(df_end_date, df_start_date, self.datasource.id))

        profile_datasource_id = self.context['profile_datasource_id']
        profile_channel_id = self.context['profile_channel_id']

        # correct for profile that should still be localized to the datasource timezone
        prof_start = new_start.tz_convert(local_tz).tz_localize(None).tz_localize(pytz.UTC)
        prof_end = new_end.tz_convert(local_tz).tz_localize(None).tz_localize(pytz.UTC)
        # At least retrieve a year of profile to be able to extend it.
        if (new_end - new_start).days < 366:
            prof_end = prof_start + pd.Timedelta('366d')
        profile_df = self.load_side_input(profile_datasource_id, profile_channel_id, prof_start, prof_end)

        if profile_df is None or profile_df.empty:
            raise FlowCancelException("No profile/proxy timeseries found for profile/proxy datasource id {}".format(profile_datasource_id))
        if prof_start < profile_df.index[0] or prof_end > profile_df.index[-1] or profile_df.index[-1] - profile_df.index[0] < dt.timedelta(days=371):
            LOGGER.warn("Not enough profile/proxy data found for profile/proxy %s. Attempting to search beyond the requested time period.", profile_datasource_id)
            prof_end = prof_end + pd.DateOffset(years=10)
            prof_start = prof_start - pd.DateOffset(years=10)
            profile_df = self.load_side_input(profile_datasource_id, profile_channel_id, prof_start, prof_end)

        if len(profile_df.columns) > 1:
            profile_df = profile_df[[profile_df.columns[0]]]
        profile_df.columns = [channel_classifier]
        profile_df = RuleUtil.localize_profile(profile_df, local_tz, self.datasource)

        LOGGER.info("Using profile %s to de-aggregate scalar data to interval data", datasource_id)
        # Append profile if not long enough
        if profile_df.index[-1] < new_end:
            LOGGER.warn('Profile data reaches till %s while last date of normalisation %s.'
                        'Timeshifting the last part of the profile.', profile_df.index[-1], new_end)
            profile_df = self.append_to_profile(profile_df, new_end)
        # Prepend profile if not long enough
        if new_start < profile_df.index[0]:
            LOGGER.warn('Profile data starts from %s while first date of normalisation %s.'
                        'Timeshifting the first part of the profile.', profile_df.index[0], new_start)
            profile_df = self.prepend_to_profile(profile_df, new_start)
        profile_df = profile_df.loc[new_start:new_end]

        # To profile full range of dates of the historical scalar reads, the partial months on both edges need to be completed.
        data_frame = self.complete_partial_month(data_frame, self.source_column, profile_df, channel_classifier, self.datasource.timezone)
        data_frame, df_interval = self.profile_scalar_to_interval(data_frame, self.source_column, profile_df, channel_classifier, new_start, new_end, self.destination_column)

        # This is used to annotate the historical dates before complete the partial months
        df_origin_index = df_interval.index
        origin_start_dates = df_interval['START']

        df_interval[self.source_heartbeat_column] = 3600
        # Add annotations of the historical reads
        flag_col = 'FLAG:historical_scalar_reads_date'

        df_interval.loc[df_origin_index, flag_col] = [
            DictWrapper(historical_scalar_end_date=date, historical_scalar_start_date=start)
            for (date, start) in zip(df_origin_index, origin_start_dates)]

        return RuleResult(result=df_interval)

    def calculate_start_column(self, data_frame):
        """ Calculates and adds a START column to the dataframe.
        The START column is calculated based on the index and the
        heartbeat column.

        Args:
            data_frame (pd.DataFrame):

        Returns:
            pd.DataFrame
        """

        import pandas as pd
        data_frame.loc[:, 'START'] = [x - pd.Timedelta(seconds=y) for (x, y) in zip(data_frame.index, data_frame[self.source_heartbeat_column])]
        return data_frame

    def get_new_start_end_date(self, data_frame):
        """Returns the original start and end of the scalar data_frame, and the start and end date
        after completing the partial months, i.e. 01:00 hour of month start of the first month and
        month end of the last month (the 00:00 hour of the next month start) .

        Args:
            data_frame (dataframe):

        Returns: timestamp, timestamp, timestamp, timestamp

        """
        import pandas as pd
        df_end_date = data_frame.index[-1]
        df_start_date = data_frame.loc[data_frame.index[0], 'START']
        LOGGER.info('Original scalar read covers dates from %s to %s', df_start_date, df_end_date)
        local_tz = self.datasource.timezone
        # new_start_utc = df_start_date.tz_convert('UTC') - pd.offsets.MonthBegin(1)
        # new_end_utc = df_end_date.tz_convert('UTC') + pd.offsets.MonthBegin(1)
        # if (new_end_utc - pd.offsets.MonthBegin(1)) == df_end_date:
        #     new_end_utc = df_end_date
        #     LOGGER.info("the end date %s is already an month end, we don't need to build out", new_end_utc)
        # if (new_start_utc + pd.offsets.MonthBegin(1)) == df_start_date:
        #     new_start_utc = df_start_date
        #     LOGGER.info("the start date %s is already an month start, we don't need to build out", new_start_utc)
        # LOGGER.info('For weather norm, scalar read is built out on both edges, new range from %s to %s', new_start_utc, new_end_utc)
        # return df_end_date, df_start_date, new_end_utc, new_start_utc
        new_end = (df_end_date.tz_convert(local_tz) + pd.offsets.MonthBegin(1)).tz_convert('UTC')
        # If the df_end_date or df_start_date is already MonthBegin, don't need to extend the data.
        if (new_end.tz_convert(local_tz) - pd.offsets.MonthBegin(1)).tz_convert('UTC') == df_end_date:
            new_end = df_end_date
            LOGGER.info("the end date %s is already an month end, we don't need to build out", new_end)
        new_start = (df_start_date.tz_convert(local_tz) - pd.offsets.MonthBegin(1)).tz_convert('UTC')
        if (new_start.tz_convert(local_tz) + pd.offsets.MonthBegin(1)).tz_convert('UTC') == df_start_date:
            new_start = df_start_date
            LOGGER.info("the start date %s is already an month start, we don't need to build out", new_start)
        LOGGER.info('For weather norm, scalar read is built out on both edges, new range from %s to %s', new_start, new_end)
        return df_end_date, df_start_date, new_end, new_start

    def get_profile_datasource_id_with_tag(self, tag_name='metadata'):
        """ Create a profile_datasource_id by tag

        Returns:
            str
        """
        tag, _, _ = self.datasource.get_tag(tag_name, latest_version=True)
        if not tag:
            LOGGER.warn("No \'%s\' tag available on datasource %s. No profile datasouce id will be returned.", tag_name, self.datasource.id)
            return None
        tag_properties = {kv.key: kv.value for kv in tag.properties}
        for key in ['market', 'discocode', 'profileclass']:
            if not tag_properties.get(key, None):  # If any of the keys not available, can't create a datasource_id for profile
                LOGGER.warn("Tag '%s' on datasource %s does not have property key %s", tag_name, self.datasource.id, key)
                return None
        # Two situations of profile datasource_id
        if 'ERCOT' in tag_properties['market']:
            prof_datasource_id = 'ALL_' + tag_properties['market'] + '_' + tag_properties['profileclass']
        else:
            prof_datasource_id = tag_properties['discocode'] + '_' + tag_properties['profileclass']
        return prof_datasource_id

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
            go_forw_time = pd.Timedelta(days=364) # 364 is divisible by 7, this will shift the peak only 1 day per year
            if (year % 6 == 0 and year > 0):  # Every 6 year, a 'shift' of 7 days will occur
                go_forw_time = pd.Timedelta(days=371)
            date_first_value = profile_df.index[0]
            cast = profile_df.loc[date_first_value: date_first_value + go_forw_time - one_interval].copy()
            cast.index = cast.index - go_forw_time
            year += 1
            profile_df = pd.concat([cast, profile_df], join='inner', axis=0)
        return profile_df

    @staticmethod
    def append_to_profile(profile_df, append_till):
        """

        Args:
            profile_df(pd.DataFrame):
            append_till(pd.Timestamp):

        Returns:

        """
        import pandas as pd
        one_interval = pd.Timedelta(hours=1)
        # appending timeshifted profile data to existing profile data
        year = 1
        while profile_df.index[-1] < append_till:
            go_back_time = pd.Timedelta(days=364) # 364 is divisible by 7, this will shift the peak only 1 day per year
            if (year % 6 == 0 and year > 0):  # Every 6 year, a 'shift' of 7 days will occur
                go_back_time = pd.Timedelta(days=371)
            date_last_value = profile_df.index[-1]
            cast = profile_df.loc[date_last_value - go_back_time + one_interval: date_last_value].copy()
            cast.index = cast.index + go_back_time
            year += 1
            profile_df = pd.concat([profile_df, cast], join='inner', axis=0)
        return profile_df

    def complete_partial_month(self, data_frame, data_col, profile_df, prof_data_col, local_tz):
        """complete the partial months on both ends of a scalar data_frame

        Args:
            data_frame(dataframe): a scalar data_frame
            data_col(str): column name of the column containing scalar read
            profile_df (dataframe): profile data that covers the whole period of scalar_df
            prof_data_col (str): the classifier (column name) of the profile data
            local_tz(str):

        Returns:
            data_frame(dataframe): data_frame extended on both end

        """
        import pandas as pd

        data_frame = data_frame[~pd.isnull(data_frame[data_frame.columns[0]])]

        old_end, old_start, new_end, new_start = self.get_new_start_end_date(data_frame)
        one_hour = pd.Timedelta('1h') # new_start starts 01:00, old_start comes directly from scalar, starts from 00:00
        if new_end != old_end:  # If the last scalar read is not the end of the month.
            start_last_month = (old_end.tz_convert(local_tz) - pd.offsets.MonthBegin(1) + one_hour).tz_convert('UTC')

            prof_use_to_esti = profile_df.loc[old_end + one_hour: new_end, prof_data_col].sum()
            prof_use_first_part = profile_df.loc[start_last_month: old_end, prof_data_col].sum()  # The second part of the month is to be estimated.
            prof_use_whole_scalar = profile_df.loc[data_frame.loc[old_end, 'START'] + one_hour: old_end, prof_data_col].sum()
            if prof_use_whole_scalar == 0:
                LOGGER.error('The sum of profile usage between %s and %s is 0, not able to complete the last month', data_frame.loc[old_end, 'START'], old_end)
                raise ValueError('The sum of profile usage between {} and {} is 0, not able to complete the last month'.format(data_frame.loc[old_end, 'START'], old_end))
            else:
                portion_first_part = prof_use_first_part / prof_use_whole_scalar
                usage_first_part = portion_first_part * data_frame.loc[old_end, data_col]
                ratio = prof_use_to_esti / prof_use_first_part  # The ratio of usage between same dates stays the same for profile and scalar usage.
                data_frame.loc[new_end, ['START', data_col]] = old_end, usage_first_part * ratio

        if new_start != old_start:  # If the first scalar read is not the start of the month.
            end_first_month = (old_start.tz_convert(local_tz) + pd.offsets.MonthBegin(1)).tz_convert('UTC')

            prof_use_to_esti = profile_df.loc[new_start: old_start - one_hour, prof_data_col].sum()
            prof_use_second_part = profile_df.loc[old_start: end_first_month, prof_data_col].sum() # the part to be estimated is the first part.
            prof_use_whole_scalar = profile_df.loc[old_start: data_frame.index[0], prof_data_col].sum()
            if prof_use_whole_scalar == 0:
                LOGGER.error('The sum of profile usage between %s and %s is 0, not able to complete the last month', old_start, data_frame.index[0])
                raise ValueError('The sum of profile usage between {} and {} is 0, not able to complete the last month'.format(old_start, data_frame.index[0]))
            else:
                portion_second_part = prof_use_second_part / prof_use_whole_scalar
                usage_second_part = portion_second_part * data_frame.loc[data_frame.index[0], data_col]
                ratio = prof_use_to_esti / prof_use_second_part  # The ratio of usage between same dates stays the same for profile and scalar usage.
                data_frame.loc[old_start, ['START', data_col]] = new_start - one_hour, usage_second_part * ratio
        data_frame.sort_index(inplace=True)
        return data_frame

    @staticmethod
    def profile_scalar_to_interval(data_frame, data_col, profile_df, prof_data_col, start, end, destination_col):
        """profile the scalar_df to hourly frequency

        Args:
            data_frame (dataframe): scalar_df that needs to be profiled
            data_col (str):  column name of the column containing scalar read
            profile_df (dataframe): profile data that covers the whole period of scalar_df
            prof_data_col (str): the classifier (column name) of the profile data
            start(pandas.Timestamp): start timestamp of data_frame
            end(pandas.Timestamp): end timestamp of data_frame
            destination_col(str): destination column name

        Returns:
            data_frame: dataframe with more columns: 'factor' and 'prof_usage'
            df_interval: dataframe, hourly freq

        """

        import pandas as pd
        for index, row in data_frame.iterrows():
            # The 00:00 hour of a scalar period belongs to the last period, plus 1 hour to exclude that hour.
            prof_start = row['START'] + pd.Timedelta('1h')
            prof_end = index
            data_frame.loc[index, 'prof_usage_sum'] = profile_df.loc[prof_start: prof_end, prof_data_col].sum()
            # Then calculate factor(scalar/profile_usage) by every scalar reading period.
        data_frame.loc[:, 'factor'] = data_frame.loc[:, data_col]/data_frame.loc[:, 'prof_usage_sum']
        # Map the factors to every hour of the profile by scalar reading period.
        df_interval = data_frame.reindex(profile_df[start + pd.Timedelta('1h'): end].index, method='bfill')
        # Multiply by the factor to scale to the same level than scalar usage
        df_interval.loc[:, destination_col] = profile_df.loc[df_interval.index, prof_data_col] * df_interval.loc[:, 'factor']
        return data_frame, df_interval