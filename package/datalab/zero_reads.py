from __future__ import absolute_import

import logging

from energyworx.domain import RuleResult
from energyworx.rules.base_rule import AbstractRule

logger = logging.getLogger()


class ZeroReads(AbstractRule):

    def apply(self, margin=0.01, common_zero_reads_threshold=1, **kwargs):
        """
        Check whether the Data frame has values equal to 0 or equal to (0 + margin).
        If there are values matching the filter they are added into a Series and returned.

        Args:
            common_zero_reads_threshold (float): threshold (0,1) for the percentage of times a value needs to be zero
                on a certain time of the week.
            margin (float): The margin for a value to be considered a zero value

        Returns:
            RuleResult : A Series with a flag
                        {'margin': 0.01}
                        or an empty Series if no flags are set

        Raises:
            TypeError: If margin is None or a negative number
        """
        import pandas as pd
        if margin is None or margin < 0:
            raise TypeError('margin: [{}] is not a valid margin'.format(margin))

        zero_reads_mask = self.dataframe[self.source_column] <= margin

        common_zero_reads = self.find_common_zero_reads(zero_reads_mask)

        if common_zero_reads_threshold < 1:
            zero_reads_mask &= common_zero_reads < common_zero_reads_threshold

        zero_reads_mask &= self.data_filter

        df = common_zero_reads[zero_reads_mask]
        df.name = 'commonness'
        df = pd.DataFrame(df)
        df['margin'] = margin

        result = pd.Series(data=df.to_dict(orient='records'), index=self.dataframe[zero_reads_mask].index)
        return RuleResult(result=result)

    def find_common_zero_reads(self, zero_reads_mask):
        """
        Finds which zero reads occur often at some time during a week.

        Args:
            zero_reads_mask (pd.Series): locations of zero reads

        Returns (pd.Series):
            A boolean mask of which indexes it is probably normal they are zero.
        """
        df = self.dataframe.copy()

        # Sum the total zero reads per moment in a week
        df['dayofweek'] = df.index.dayofweek
        df['time'] = df.index.time

        # Group by moment in week
        groupby_weekly_periodic_zero_reads = zero_reads_mask.groupby((df.index.dayofweek, df.index.time))

        # Divide by the number of weeks
        weekly_periodic_zero_reads = groupby_weekly_periodic_zero_reads.sum() / groupby_weekly_periodic_zero_reads.count()
        weekly_periodic_zero_reads.index.names = ['dayofweek', 'time']
        weekly_periodic_zero_reads.name = 'periodic'

        # Merge with original dataframe to create mask
        df.reset_index(inplace=True)
        df = df.merge(weekly_periodic_zero_reads.reset_index(), how='left', on=('dayofweek', 'time'))
        df.set_index('index', inplace=True)
        return df['periodic']
