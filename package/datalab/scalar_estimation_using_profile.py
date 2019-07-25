from __future__ import absolute_import

from energyworx.rules.rule_util import RuleUtil
from energyworx_public.domain import FlowCancelException
from energyworx_public.rule import RuleResult, AbstractRule, PREPARE_DATASOURCE_IDS_KEY
import logging
import pytz
import datetime as dt

logger = logging.getLogger()


class ScalarEstimationUsingProfile(AbstractRule):
    """ Estimates new values for scalar data based on the profile data. """
    START = 'START'
    FACTORS = 'FACTORS'

    def prepare_context(self, datasource_id=None, datasource_format=None, classifier=None, **kwargs):
        """ Retrieves profile/proxy datasource dataframe.

        Args:
            datasource_id (str): datasource identifier of profile/proxy
            datasource_format (str): format of the datasource to be retrieved
            classifier (str): channel classifier to be retrieved

        Returns:
            dict(datasource_id=pd.DataFrame)
        """
        profile_datasource_id = datasource_id
        # TODO[BP] DISABLED LOGGING FOR NOW!! logger is None so logger.info will throw a NoneTypeError!!!
#         logger.info("Loading prepare context for datasource id %s and classifier %s", profile_datasource_id, classifier)
        if not profile_datasource_id:
            profile_datasource_id = self.get_profile_datasource_id(datasource_format)
        if profile_datasource_id is None:
            raise FlowCancelException("No profile/proxy datasource found with name {} for datasource {}".format(datasource_format, self.datasource.id))
#         logger.info("Continuing with datasource id %s for loading profile", profile_datasource_id)
        profile_datasource = self.load_datasource(profile_datasource_id)
        if profile_datasource is None:
            raise FlowCancelException("No profile/proxy datasource found with id {} for datasource {}".format(profile_datasource_id, self.datasource.id))

        usage_channel_classifier_name = classifier
        usage_channel = profile_datasource.get_channel_by_classifier(classifier)
        if usage_channel is None:
            usage_channel = profile_datasource.get_channel_by_classifier("Usage")  # use older profile classifier
            usage_channel_classifier_name = 'Usage'
        if usage_channel is None:
            raise FlowCancelException("No correct channel found on profile/proxy datasource datasource {}".format(profile_datasource_id))
#         logging.info("Context prepared with %s: %s, usage_channel_id: %s, usage_channel_classifier_name: %s", PREPARE_DATASOURCE_IDS_KEY, profile_datasource_id, usage_channel.id, usage_channel_classifier_name)
        return {'prepare_datasource_ids': [profile_datasource_id],
                'profile_datasource_id': profile_datasource_id,
                'usage_channel_id': usage_channel.id,
                'usage_channel_classifier_name': usage_channel_classifier_name}

    def apply(self, **kwargs):
        """ Estimation algorithm for scalar data based on profile
        or proxy usage.

        Returns:
            RuleResult(result=pd.Series)
        """
        import numpy as np
        df = self.dataframe
        profile_datasource_id = self.context['profile_datasource_id']
        usage_channel_id = self.context['usage_channel_id']
        usage_channel_classifier_name = self.context['usage_channel_classifier_name']
        output_column = 'PRED:scalar_estimation_using_profile'
        if not profile_datasource_id or not usage_channel_id:
            logger.warning('no profile datasource id or usage channel id provided in rule context')
            return RuleResult()

        df[output_column] = np.nan
        if not self.data_filter.any():
            # no need for estimations (this also means no profile was retrieved in prepare_context)
            return RuleResult(result=df[output_column])

        df = self.calculate_start_column(df)

        logging.info("Preparing profile data")
        heartbeat_column = '{}_HEARTBEAT'.format(self.source_column)
        start = self.dataframe.index[0] - dt.timedelta(seconds=self.dataframe[heartbeat_column].iloc[0])
        start = start.tz_convert(self.datasource.timezone).tz_localize(None).tz_localize(pytz.UTC)  # correct for profile that should still be localized to the datasource timezone
        end = self.dataframe.index[-1]
        end = end.tz_convert(self.datasource.timezone).tz_localize(None).tz_localize(pytz.UTC)
        profile_df = self.load_side_input(profile_datasource_id, usage_channel_id, start, end)
        if profile_df is None or profile_df.empty or start < profile_df.index[0] or end > profile_df.index[-1]:
            logging.warn("No or not enough profile/proxy data found for profile/proxy %s with channel %s (%s) between %s and %s. Attempting to retrieve more than just the requested time period.", profile_datasource_id, usage_channel_id, usage_channel_classifier_name, start, end)
            if profile_df is not None:
            	logging.warn("Size of dataframe was: %s", len(profile_df))
            extended_start = dt.datetime(1970, 1, 1, 1, tzinfo=pytz.UTC)
            extended_end = dt.datetime(3000, 1, 1, tzinfo=pytz.UTC)
            profile_df = self.load_side_input(profile_datasource_id, usage_channel_id, extended_start, extended_end)
        if profile_df is None or profile_df.empty:
            raise FlowCancelException("No profile/proxy timeseries found for profile/proxy datasource id {} with channel {} ({}) between extended start {} and end {}".format(profile_datasource_id, usage_channel_id, usage_channel_classifier_name, extended_start, extended_end))

        if len(profile_df.columns) > 1:
            profile_df = profile_df[[profile_df.columns[0]]]
        profile_df.columns = [usage_channel_classifier_name]
        profile_df = RuleUtil.localize_profile(profile_df, self.datasource.timezone, self.datasource)

        profile_df, correction_factor, _ = self.calculate_factors(profile_df, df, usage_channel_classifier_name, self.data_filter.values)
        if profile_df is None:
            return RuleResult()
        logging.info("Start estimating flagged points.")
        for estimation_end in df[self.data_filter].index:
            estimation_start = df.loc[estimation_end, self.START]
            profile_gap_data = profile_df.loc[estimation_start + dt.timedelta(seconds=1):estimation_end, usage_channel_classifier_name]
            total_gap_usage_profile = profile_gap_data.sum()
            df.loc[estimation_end, output_column] = correction_factor * total_gap_usage_profile
        del df[self.START]
        return RuleResult(result=df[output_column])

    def get_profile_datasource_id(self, datasource_format='{metadata.market}{_}{metadata.rateclass}'):
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
            datasource_format = '{metadata.market}{_}{metadata.rateclass}'
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
                logging.warn("No '%s' tag available on datasource %s. No profile datasouce id will be returned." % (tag_name, self.datasource.id))
                continue
            raveled_tag_properties = {kv.key: kv.value for kv in current_tag.properties}
            name_piece = raveled_tag_properties.get(property_key)
            # Overwrite discocode with "ALL_ERCOT" for market ERCOT.
            if property_key == "discocode" and raveled_tag_properties.get('market') == "ERCOT":
                name_piece = 'ALL_ERCOT'
            if name_piece:
                datasource_id.append(name_piece)
            else:
                logging.warn("Tag '%s' on datasource %s does not have property key %s" % (tag_name, self.datasource.id, property_key))
        return "".join(datasource_id)

    def calculate_start_column(self, df):
        """ Calculates and adds a START column to the dataframe.
        The START column is calculated based on the index and the
        heartbeat column.

        Args:
            df (pd.DataFrame):

        Returns:
            pd.DataFrame
        """
        heartbeat_column = '{}_HEARTBEAT'.format(self.source_column)
        df[self.START] = map(lambda ts, seconds: ts - dt.timedelta(seconds=seconds), df.index, df[heartbeat_column])
        return df

    def calculate_factors(self, profile_df, scalar_df, classifier, need_prediction_booleans=None):
        """ Calculates the necessary factor to be able to apply estimations.

        Args:
            profile_df (pd.DataFrame):
            scalar_df (pd.DataFrame):
            classifier (str):
            need_prediction_booleans (np.array(list)):

        Returns:
            (pd.DataFrame, float, pd.DataFrame)
        """
        import pandas as pd
        one_hour = pd.Timedelta(hours=1)  # this timedelta is used often
        if need_prediction_booleans is not None:
            scalar_df = scalar_df[~need_prediction_booleans]
        start_dates = scalar_df[self.START]
        stop_dates = scalar_df.index
        if len(start_dates) == 0:
            logging.warn("Everything needs to be estimated for datasource %s. Can't do this obviously.", self.datasource.id)
            return None, None, None
        first_start_date = start_dates[0]
        last_stop_date = stop_dates[-1]
        if profile_df.index[0] > first_start_date:
            logging.info("Prepending to profile such that enough overlap is created.")
            profile_df = self.prepend_to_profile(profile_df, first_start_date)
        if profile_df.index[-1] < last_stop_date:
            logging.info("Appending to profile such that enough overlap is created.")
            profile_df = self.append_to_profile(profile_df, last_stop_date)

        # Determine latest profile heartbeat (the heartbeat has been calculated upon cleansing of the profile within this rule)
        heartbeat_column = heartbeat_column = '{}_HEARTBEAT'.format(classifier)
        latest_heartbeat = int(profile_df[heartbeat_column].iloc[-1])
        # Make a dataframe with all factors
        dates = pd.date_range(first_start_date, last_stop_date, freq='{}s'.format(latest_heartbeat), closed='right', tz=pytz.UTC)
        factors_df = pd.DataFrame(index=dates, columns=[self.FACTORS])
        for j in range(len(start_dates)):
            if stop_dates[j] > profile_df.index[-1]:
                logging.warn('The end date of the scalar data is after the last date of the profile. Cannot estimate all ratios between the scalar data and the profile data.')
            scalar_read_usage = scalar_df.loc[stop_dates[j], self.source_column]
            if pd.isnull(scalar_read_usage):
                # this is a gap that has to be estimated
                continue
            profile_usage = profile_df.loc[start_dates[j] + one_hour: stop_dates[j], classifier].sum()
            factors_df.loc[start_dates[j] + one_hour: stop_dates[j], self.FACTORS] = scalar_read_usage/profile_usage
        factors_with_weights = factors_df[self.FACTORS].value_counts(sort=False, dropna=True).reset_index()  # note that value_counts creates a series with counts (and an index with the factors!)
        correction_factor = (factors_with_weights[self.FACTORS] * factors_with_weights['index']).sum() / factors_with_weights[self.FACTORS].sum()
        logging.info("Correction factor is: %s" % correction_factor)
        start_index_factors_df = last_stop_date - pd.DateOffset(years=1)
        if factors_df.index[0] > start_index_factors_df:
            # extending factors_df to have enough factors if scalar data wasn't covering enough dates
            dates = pd.date_range(start_index_factors_df, factors_df.index[-1], freq='{}s'.format(latest_heartbeat), closed='right', tz=pytz.UTC)
            factors_df = factors_df.reindex(index=dates)
        factors_df.loc[pd.isnull(factors_df[self.FACTORS]), self.FACTORS] = correction_factor
        factors_df[self.FACTORS] = pd.to_numeric(factors_df[self.FACTORS], errors='coerce')
        factors_df.bfill(inplace=True)
        return profile_df, correction_factor, factors_df

    @staticmethod
    def prepend_to_profile(profile_df, prepend_till):
        """ Prepends full years of data to the dataframe.
        """
        import pandas as pd
        one_interval = dt.timedelta(hours=1)
        one_year = dt.timedelta(days=365)
        one_leap_year = dt.timedelta(days=366)
        go_forward_time = dt.timedelta(days=371)  # this number is divisible by 7 such that mondays are mapped on mondays
        # prepending profile data to existing profile data
        while profile_df.index[0] > prepend_till:
            date_first_value = profile_df.index[0]
            cast_last_date = date_first_value - one_interval
            contains_leap1 = all([cast_last_date.year % 4 == 0, cast_last_date.month > 2])
            contains_leap2 = all([(cast_last_date.year - 1) % 4 == 0, cast_last_date.month <= 2])
            if contains_leap1 or contains_leap2:
                cast = profile_df.loc[date_first_value + go_forward_time - one_leap_year: cast_last_date + go_forward_time].copy()
                cast.index = cast.index.map(lambda date: date - go_forward_time)
            else:
                cast = profile_df.loc[date_first_value + go_forward_time - one_year: cast_last_date + go_forward_time].copy()
                cast.index = cast.index.map(lambda date: date - go_forward_time)
            profile_current_length = len(profile_df)
            profile_df = pd.concat([cast, profile_df], join='inner', axis=0)
            if len(profile_df) == profile_current_length:
                logging.warning('Profile length did not increase, not prepending to profile anymore.')
                break
        return profile_df

    @staticmethod
    def append_to_profile(profile_df, append_till):
        """ Prepends full years of data to the dataframe.
        """
        import pandas as pd
        one_interval = dt.timedelta(hours=1)
        one_year = dt.timedelta(days=365)
        one_leap_year = dt.timedelta(days=366)
        go_back_time = dt.timedelta(days=371)  # this number is divisible by 7 such that mondays are mapped on mondays
        # appending timeshifted profile data to existing profile data
        while profile_df.index[-1] < append_till:
            date_last_value = profile_df.index[-1]
            contains_leap1 = (date_last_value + one_interval).year%4 == 0 and (date_last_value + one_interval).month <= 2
            contains_leap2 = ((date_last_value + one_interval).year + 1) % 4 == 0 and (date_last_value + one_interval).month > 2
            if contains_leap1 or contains_leap2:
                cast = profile_df.loc[date_last_value - go_back_time + one_interval: date_last_value - go_back_time + one_leap_year + one_interval].copy()
                cast.index = cast.index.map(lambda date: date + go_back_time)
            else:
                cast = profile_df.loc[date_last_value - go_back_time + one_interval: date_last_value - go_back_time + one_year + one_interval].copy()
                cast.index = cast.index.map(lambda date: date + go_back_time)
            profile_current_length = len(profile_df)
            profile_df = pd.concat([profile_df, cast], join='inner', axis=0)
            if len(profile_df) == profile_current_length:
                logging.warning('Profile length did not increase, not appending to profile anymore.')
                break
        return profile_df
