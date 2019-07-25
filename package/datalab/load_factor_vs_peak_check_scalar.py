from __future__ import absolute_import


from energyworx.rules.base_rule import AbstractRule
from energyworx.domain import RuleResult, DictWrapper
import logging

logger = logging.getLogger()


class LoadFactorVsPeakCheckScalar(AbstractRule):
    def apply(self, load_factor_threshold=0.4, peak_threshold=1000, **kwargs):
        """
        This rule calculates the peak load and the load factor

        load factor is calculated by dividing the total idr sum by
        the total peak multiplied with the amount of rows

        Rows will be flagged if the peak load is aboven the peak_threshold or
        if the load factor(lf) is lower than the load_factor_threshold

        Args:
            load_factor_threshold (float):  Threshold for the load_factor, must be a float
            peak_threshold (int): Threshold for the peak, must be an integer

        Returns:
            RuleResult: Series based on flag_column from dataframe

        Throws:
            TypeError: If load_factor_threshold is not a float
            TypeError: If peak_threshold is not an integer
        """
        if not isinstance(load_factor_threshold, float):
            raise TypeError("A invalid value was passed for load_factor_threshold")
        if not isinstance(peak_threshold, int):
            raise TypeError("A invalid value was passed for peak_threshold")

        import numpy as np

        flag_col = 'FLAG:load_factor_vs_peak_scalar'
        dataframe = self.dataframe[[self.source_column, self.source_heartbeat_column]].copy()

        dataframe['LOAD'] = dataframe[self.source_column]  # these are already hourly values
        total_sum = dataframe['LOAD'].sum()
        total_count = dataframe['LOAD'].count()
        total_peak = dataframe['LOAD'].max()
        dataframe['lf'] = total_sum / (total_peak * total_count)

        dataframe[flag_col] = np.nan
        flag_filter = (dataframe['lf'] < load_factor_threshold) & (dataframe['LOAD'].max() > peak_threshold) & self.data_filter

        dataframe.loc[flag_filter, 'peak'] = dataframe['LOAD'].max()
        dataframe.loc[flag_filter, 'lf_threshold'] = load_factor_threshold
        dataframe.loc[flag_filter, 'peak_threshold'] = peak_threshold

        dataframe.loc[flag_filter, flag_col] = map(
                DictWrapper,  # Once dicts are properly supported, only the map functions needs to be removed
                dataframe.loc[flag_filter, ['load_factor', 'peak', 'lf_threshold', 'peak_threshold']].to_dict(orient='records'))

        return RuleResult(result=dataframe[flag_col])
