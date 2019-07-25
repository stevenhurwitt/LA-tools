
from __future__ import absolute_import

from energyworx.rules.base_rule import AbstractRule
from energyworx.domain import RuleResult, DictWrapper
import logging

logger = logging.getLogger()


class LoadFactorVsPeakCheckIdr(AbstractRule):

    def apply(self, load_factor_threshold=0.4, peak_threshold=1000, **kwargs):
        """
        This rule calculates the peak load and the load factor
        load factor is calculated by dividing the total idr sum by
        the total peak multiplied with the amount of rows

        Rows will be flagged if the peak load is above the peak_threshold or
        if the load factor(lf) is lower than the load_factor_threshold

        Args:
            load_factor_threshold (float): Threshold for the load_factor, must be a float
            peak_threshold (int): Threshold for the peak, must be an integer

        Returns:
            RuleResult : Series based on flag_column from dataframe

        Throws:
            TypeError: If load_factor_threshold is not a float
            TypeError: If peak_threshold is not an integer
        """

        if not isinstance(load_factor_threshold, float):
            raise TypeError("An invalid value was passed for load_factor_threshold")
        if not isinstance(peak_threshold, int):
            raise TypeError("An invalid value was passed for peak_threshold")

        import pandas as pd
        import numpy as np

        df = self.dataframe.copy()
        heartbeat = df["{}_HEARTBEAT".format(self.source_column)]
        flag_col = 'FLAG:load_factor_vs_peak_idr'

        df['LOAD'] = df[self.source_column] * (3600 / heartbeat)
        total_sum = df['LOAD'].sum()
        total_count = df['LOAD'].count()
        total_peak = df['LOAD'].max()
        df['lf'] = total_sum / (total_peak * total_count)

        df['flag'] = np.nan
        flag_filter = (df['lf'] < load_factor_threshold) & (total_peak > peak_threshold) & self.data_filter

        df.loc[flag_filter, 'peak'] = total_peak
        df.loc[flag_filter, 'lf_threshold'] = load_factor_threshold
        df.loc[flag_filter, 'peak_threshold'] = peak_threshold

        df.loc[flag_filter, flag_col] = map(
            DictWrapper,  # Once dicts are properly supported, only the map functions needs to be removed
            df.loc[flag_filter, ['load_factor', 'peak', 'lf_threshold', 'peak_threshold']].to_dict(orient='records'))

        return RuleResult(result=df[flag_col])
