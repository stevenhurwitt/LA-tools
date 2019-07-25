
from __future__ import absolute_import

from energyworx.rules.base_rule import AbstractRule
from energyworx.domain import RuleResult, DictWrapper
import logging

logger = logging.getLogger()


class PeriodicityOrIntervalDipSpikeOutlier(AbstractRule):

    def apply(self, load_shape_type, **kwargs):
        """

        This implementation won't handle variating heartbeats well.
        Rule needs to be split in two seperate rules.

        Args:
            load_shape_type:
            **kwargs: All arguments for both rules

        Returns:

        """
        if load_shape_type == 'day_seasonal':
            logger.info("Running periodicity check because of detected day seasonal")
            rule_result = self.periodicity_check(**kwargs)
        else:
            logger.info("Running interval dip/spike check because of not seasonality detected")
            rule_result = self.interval_dip_spike_check(**kwargs)
        return rule_result

    def periodicity_check(self, nr_mads=7, window=40, no_of_occurences=12, **kwargs):
        """

        Args:
            nr_mads:
            window (int): hours
            no_of_occurences (int): hours
            informational:
            **kwargs:

        Returns:

        """
        import numpy as np
        import pandas as pd

        df = self.dataframe.copy()
        heartbeat = df[self.source_heartbeat_column].sort_values().median()
        window = window * 3600 / heartbeat
        no_of_occurences = no_of_occurences * 3600 / heartbeat
        logger.info("Performing periodicity check with arguments: nr_mads[{}], window[{}], no_of_occurences[{}]".format(nr_mads, window, no_of_occurences))
        df['rol_mean'] = df[self.source_column].rolling(window=int(60*24*3600/heartbeat), min_periods=1, center=True).mean()
        df['detrended'] = df[self.source_column] - df.rol_mean  # Do an overall detrend

        df['day_of_week'] = df.index.dayofweek
        df['hour'] = df.index.hour
        grouped_df = df.groupby(['day_of_week', 'hour'])

        def mad(arr):
            med = np.median(arr)
            return np.median(np.abs(arr - med))

        def get_annotation(x):
            return DictWrapper(
                nr_deviating_points=sum(bools),
                nr_mads=nr_mads,
                window=window)

        # using medians and median absolute deviations
        median = []
        median_absolute_deviation = []
        for group in grouped_df.groups:
            median.append(grouped_df.get_group(group)['detrended'].rolling(window=int(window), min_periods=1, center=True).median())
            median_absolute_deviation.append(grouped_df.get_group(group)['detrended'].rolling(window=int(window), min_periods=1, center=True).apply(lambda arr: mad(arr)))

        df = df.join(pd.DataFrame(dict(median=pd.concat(median), mad=pd.concat(median_absolute_deviation))), how='outer')

        df['date'] = df.tz_localize('UTC').tz_convert(self.datasource.timezone).index.date  # corrected for timezone
        df['FLAG'] = np.nan
        for date in np.unique(df['date']):
            bools = abs(df['detrended'][df.date == date] - df['median'][df.date == date]) > nr_mads * df['mad'][df.date == date]
            if sum(bools) > no_of_occurences:
                logger.debug("Checking date: {}: with {} occurrences found".format(date, sum(bools)))
                df.loc[df.date == date, 'FLAG'] = df[df.date == date].apply(lambda x: get_annotation(x), axis=1)

        return RuleResult(result=df[~df.index.duplicated(keep='first')]['FLAG'])


    def interval_dip_spike_check(self, hours=72, lookback=48, max_outlier_time=36, delta_threshold=20, min_change=30, **kwargs):
        import numpy as np

        df = self.dataframe.copy()
        heartbeat = df[self.source_heartbeat_column].sort_values().median()
        hours = hours * 3600 / heartbeat
        lookback = int(lookback * 3600 / heartbeat)
        max_outlier_time = max_outlier_time * 3600 / heartbeat * 24
        delta_threshold = delta_threshold / 100.0
        min_change = min_change / 100.0
        df['rol_mean'] = df[self.source_column].rolling(window=int(hours), min_periods=int(hours), center=False).mean()
        df['flags'] = np.nan

        def dip_spike_finder():
            nr_dip_spike = 0
            i = int(hours)  # start where we have values for the rolling mean
            while i <= df.shape[0] - hours - 1:
                # Calculate percentage difference
                delta_rol_mean = df['rol_mean'].iloc[int(i+hours)]/df['rol_mean'].iloc[i] - 1

                if delta_rol_mean > delta_threshold: #Detected a spike
                    nr_dip_spike += 1
                    for j in range(i+24,i-lookback,-1):
                        if df.rol_mean.iloc[j] - df.rol_mean.iloc[j-1] <= 0:
                            # start of spike
                            right_level = df.rol_mean.iloc[j]
                            break
                    if j==i-lookback+1:
                        # if for loop above hasn't found any point satisfying the condition
                        right_level = df.rol_mean.iloc[j]
                    max_time = i + max_outlier_time  # Has to be back at normal level within e.g. a month
                    i+=int(hours)
                    while i < max_time and i < df.shape[0]:
                        if df[self.source_column].iloc[i] <= right_level:
                            break
                        i+=1
                    if i==max_time or i==df.shape[0]:
                        # it takes to long to get back to the normal level
                        nr_dip_spike -= 1
                        i+=24*3  # skip three days in order to not capture part of the last spike
                        continue
                    elif max(df[self.source_column].iloc[j:i+1]) < df.rol_mean.iloc[j]*(1+min_change):
                        # It's not considered a spike (too small of a change)
                        nr_dip_spike -= 1
                        i+=24*3  # skip three days in order to not capture part of the last spike
                        continue
                    # it will only get here when the spike is a real spike
                    df.flags.iloc[j:i+1] = delta_rol_mean
                    i+=24*3  # skip three days in order to not capture part of the last spike

                elif delta_rol_mean < -delta_threshold:  # Detected a dip
                    nr_dip_spike += 1
                    for j in range(i+24,i-lookback,-1):
                        if df.rol_mean.iloc[j] - df.rol_mean.iloc[j-1] >= 0:
                            # start of dip
                            right_level = df.rol_mean.iloc[j]
                            break
                    if j==i-lookback+1:
                        # if for loop above hasn't found any point satisfying the condition
                        right_level = df.rol_mean.iloc[j]
                    max_time = i + max_outlier_time  # Has to be back at normal level within e.g. a month
                    i+=int(hours)
                    while i < max_time and i < df.shape[0]:
                        if df[self.source_column].iloc[i] >= right_level:
                            break
                        i+=1
                    if i==max_time or i==df.shape[0]:
                        # it takes to long to get back to the normal level
                        nr_dip_spike -= 1
                        i+=24*3  # skip three days in order to not capture part of the last spike
                        continue
                    elif min(df[self.source_column].iloc[j:i+1]) > df.rol_mean.iloc[j]*(1-min_change):
                        # It's not considered a dip (too small of a change)
                        nr_dip_spike -= 1
                        i+=24*3  # skip three days in order to not capture part of the last spike
                        continue
                    # it will only get here when the dip is a real dip
                    df.flags.iloc[j:i+1] = delta_rol_mean
                    i+=24*3  # skip three days in order to not capture part of the last spike
                i+=1
            return df, nr_dip_spike

        temp, n = dip_spike_finder()
        if n > (df.shape[0] * df[self.source_heartbeat_column].median() / 3600 / 24 / 365 * 12 / 2):  # Basically, if we find too many dips/spikes, we adjust the parameters slightly and validate again
            df['flags'] = np.nan
            temp['flags'] = np.nan
            min_change = min_change + 0.1  # add 10% to minimum change needed for flagging
            temp, n = dip_spike_finder()
            if n > 3:
                temp['flags'] = np.nan

        def annotator(x):
            a = np.nan
            percentage_change_annotation = DictWrapper(
                percentage_change=x*100,
                rolling_window=hours,
                lookback=lookback,
                max_outlier_time=max_outlier_time,
                delta_threshold=delta_threshold,
                min_change=min_change,
            )
            end_spike_annotation = DictWrapper(
                rolling_window=hours,
                lookback=lookback,
                max_outlier_time=max_outlier_time,
                delta_threshold=delta_threshold,
                min_change=min_change,
            )
            if x > delta_threshold:
                a = percentage_change_annotation
            elif x == 0.001:
                a = end_spike_annotation
            elif x < -delta_threshold:
                a = percentage_change_annotation
            elif x == -0.001:
                a = end_spike_annotation
            return a

        temp.flags = temp.flags.apply(annotator)

        return RuleResult(result=temp['flags'])