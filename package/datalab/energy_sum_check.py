from __future__ import absolute_import

from energyworx.domain import DictWrapper
from energyworx_public.rule import AbstractRule, RuleResult, PREPARE_DATASOURCE_IDS_KEY, HEARTBEAT_COLUMN_TEMPLATE
import logging

logger = logging.getLogger()


class EnergySumCheck(AbstractRule):
    def prepare_context(self, channel_classifier, **kwargs):
        return {'prepare_datasource_ids': [self.datasource.id]}

    def apply(self, channel_classifier, threshold=20.0, **kwargs):
        """
        Places flags at the end of a period where the IDR sum is different from the scalar energy usage.
        The difference is calculated in percentages

        Rule checks based on  (if SUM(INTERVAL) <> SCALAR USAGE +/- "X"%)

        Args:
            channel_classifier (str): id of the channel
            threshold (float): error percentage --> allowed percentage 'off' the scalar value

        Returns:
            RuleResult: object with a flag column (result) and meta data (metadata)

        Raises:
            TypeError: If the threshold is not an integer

            ValueError: If threshold is a negative number or higher than 100
        """
        import pandas as pd
        import numpy as np

        if not isinstance(threshold, int) and not isinstance(threshold, float) or np.isnan(threshold):
            raise TypeError("inappropriate type for threshold: {threshold_type}".format(threshold_type=type(threshold)))
        elif threshold < 0 or threshold > 100:
            raise ValueError("inappropriate value for threshold {threshold}".format(threshold=threshold))

        if channel_classifier is None:
            raise ValueError("No channel classifier entered")

        first_idr_date = self.dataframe.index[0]
        last_idr_date = self.dataframe.index[-1]

        # Get the other channel based on the channel_classifier
        # from the datasource between the first and last idr dates
        scalar_df = self.load_side_input(datasource_id=self.datasource.id, channel_id=channel_classifier,
                                         start=first_idr_date, end=last_idr_date)

        # Only search for the heartbeat if a data frame is found
        if scalar_df is not None and not scalar_df.empty:
            # rename the data channel to channel_classifier
            scalar_df.rename(columns={scalar_df.columns[0]: channel_classifier}, inplace=True)
            scalar_df[HEARTBEAT_COLUMN_TEMPLATE.format(channel_classifier)] = self.load_side_input(
                datasource_id=self.datasource.id,
                channel_id=HEARTBEAT_COLUMN_TEMPLATE.format(channel_classifier),
                start=first_idr_date, end=last_idr_date)

        if scalar_df is None or scalar_df.empty:
            logger.info("No data found for channel classifier {}".format(channel_classifier))
            return RuleResult(result=None)
        else:
            logger.info("Found {} scalar reads.".format(str(len(scalar_df))))

        df = self.dataframe[[self.source_column, self.source_heartbeat_column]].copy()
        flag_col = 'FLAG:energy_sum_check'
        df[flag_col] = np.nan

        scalar_heartbeat = scalar_df[HEARTBEAT_COLUMN_TEMPLATE.format(channel_classifier)]
        scalar_df['STOP'] = scalar_df.index
        scalar_df['HB_TIME'] = pd.to_timedelta(scalar_heartbeat, unit='s')
        scalar_df['START'] = scalar_df['STOP'].subtract(scalar_df['HB_TIME']).dt.tz_localize('UTC')

        # Get the sum of the idr values between each start - stop for rows in scalar_df
        scalar_df['IDR_SUM'] = map(lambda start, stop: df[self.source_column].loc[start: stop].sum(),
                                   scalar_df['START'], scalar_df['STOP'])

        first_idr_date = df.index[0]
        last_idr_date = df.index[-1]
        if last_idr_date < scalar_df['STOP'].index[0]:
            logger.warn(
                "Stopping validation energy_sum_check; the scalar values occur in the future compared to the idr values.")
            return RuleResult(result=df[flag_col])

        if scalar_df['START'][0] < first_idr_date:
            logger.warn("First idr date falls after start date scalar read; skipping this period.")
            return RuleResult(result=df[flag_col])

        # Calculate the percentage of difference between IDR_SUM and the scalar values
        scalar_df['percentage'] = (scalar_df['IDR_SUM'] - scalar_df[channel_classifier]) / scalar_df[channel_classifier] * 100

        flag_filter = ((scalar_df['percentage'] < -threshold) | (scalar_df['percentage'] > threshold)) & self.data_filter
        if not any(flag_filter):
            return RuleResult(result=df[flag_col])

        # Get the count of the idr values between each start - stop for rows in scalar_df
        scalar_df.loc[flag_filter, 'nr_idr_values'] = map(lambda start, stop:
                                                          df[self.source_column].loc[start: stop].count(),
                                                          scalar_df.loc[flag_filter, 'START'],
                                                          scalar_df.loc[flag_filter, 'STOP'])

        # Calculate expected amount of idr values based on timeframe / heartbeat
        scalar_df.loc[flag_filter, 'expected'] = map(lambda start, stop:
                                                     (stop - (start + (pd.to_timedelta(df[self.source_heartbeat_column].loc[start],unit='s')))).total_seconds() /
                                                     df[self.source_heartbeat_column].loc[stop] + 1,
                                                     scalar_df.loc[flag_filter, 'START'],
                                                     scalar_df.loc[flag_filter, 'STOP'])  # +1 since 'nr points' = 'nr intervals' + 1

        scalar_df.loc[flag_filter, 'missing_values'] = scalar_df.loc[flag_filter, 'expected'].subtract(
            scalar_df.loc[flag_filter, 'nr_idr_values'])

        # Flag each start where flag_filter = True
        for index, row in scalar_df.loc[flag_filter].iterrows():
            flag_date = df.loc[row['START']:row['STOP']].index[-1]  # find a valid date to put the annotation on
            df.loc[flag_date, 'idr_sum'] = row['IDR_SUM']
            df.loc[flag_date, 'scalar_read'] = row.loc[channel_classifier]
            df.loc[flag_date, 'start'] = str(row.loc['START'])
            df.loc[flag_date, 'stop'] = str(row.loc['STOP'])
            df.loc[flag_date, 'missing_values'] = row.loc['missing_values']
            df.loc[flag_date, flag_col] = DictWrapper(df.loc[[flag_date], ['idr_sum', 'scalar_read', 'start', 'stop', 'missing_values']].to_dict(
                orient='records')[0])  # TODO Once dicts are properly supported, the map function needs to be removed

        return RuleResult(result=df[flag_col])
