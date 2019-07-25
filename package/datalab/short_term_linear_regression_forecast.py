from __future__ import absolute_import

from energyworx_public.rule import AbstractRule, RuleResult, PREPARE_DATASOURCE_IDS_KEY
import logging

logger = logging.getLogger()


class ShortTermLinearRegressionForecast(AbstractRule):
    """Create a short term forecast for total client portfolio"""

    def prepare_context(self, start_date_forecast, end_date_forecast, **kwargs):
        return {'prepare_datasource_ids': ['load_losses']}

    def apply(self, start_date_forecast, end_date_forecast, history_days,
              forecast_gap, **kwargs):
        """Create a 24 hour D+2 forecast for a meter

        Create a linear regression model if there is at least history_days
        of data. Otherwise, predict same consumption as same day last
        week, unless there is not enough data for that. Then forecast
        usage of the most recent day.

        Args:
            start_date_forecast (str): First date to forecast.
            end_date_forecast (str): Last date to forecast.
            history_days (int): Length of history required to create a
                forecast.
            forecast_gap (int): Gap between most recent data and first
                forecastable date.
            **kwargs:

        Returns:
            RuleResult:

        """
        import numpy as np
        import pandas as pd
        output_col = 'short_term_linear_regression_forecast'

        df = self.dataframe.loc[:, [self.source_column]].dropna(how='all')
        first_heartbeat = pd.to_timedelta(
            self.dataframe.loc[df.index.min(), self.source_heartbeat_column], unit='S')

        start_date_forecast, end_date_forecast = self.parse_forecast_dates(
            start_date_forecast, end_date_forecast)

        # Convert to local time
        df.index = df.index.tz_convert(self.datasource.timezone)

        # Convert to hourly values and scale
        hourly_df, scaler = self.convert_to_hourly_and_scale(df, self.source_column)

        # Determine last valid day to use as historical data
        last_full_day = hourly_df.index.max().floor('d')
        if last_full_day > start_date_forecast - pd.offsets.Day(forecast_gap):
            last_full_day = start_date_forecast - pd.offsets.Day(forecast_gap)

        # Can't forecast with less than 24 hours of data
        if (df.index.max() - df.index.min() + first_heartbeat).days < 1:
            logger.error('Cannot create forecast because there is not enough '
                         'historical data with requirements history_days: %s '
                         'forecast_gap: %s and start_date_forecast: %s.',
                         history_days, forecast_gap, start_date_forecast)
            df[output_col] = np.nan
        # Can't forecast further ahead than D+2 from latest data
        elif hourly_df.index.max().floor('d') < end_date_forecast - pd.offsets.Day(forecast_gap):
            logger.error('Cannot create forecast until %s because the '
                         'historical data ends at %s.',
                         end_date_forecast, hourly_df.index.max())
            df[output_col] = np.nan
        else:
            nr_days = (last_full_day - df.index.min() + first_heartbeat).days
            if nr_days >= history_days:
                # Convert data to 24 hour days
                hourly_df = self.convert_data_to_24_hour_days(hourly_df)
                # Reindex until end of forecast period if further than last data
                if end_date_forecast + pd.offsets.Day(1) > hourly_df.index.max():
                    forecast_index = pd.date_range(
                        start=hourly_df.index.min(),
                        end=end_date_forecast + pd.offsets.Day(1),
                        freq='1h')
                    hourly_df = hourly_df.reindex(forecast_index)
                # Create dataset
                X, y = self.create_dataset(hourly_df, self.source_column, 24 * forecast_gap)
                # Create forecast
                df = self.train_and_forecast(
                    X, y,
                    start_date_forecast - pd.offsets.Day(forecast_gap),
                    end_date_forecast - pd.offsets.Day(forecast_gap),
                    scaler)
                df.rename(columns={'FORECAST': output_col}, inplace=True)
            elif nr_days >= 7:
                # If there is at least a week of historical data, forecast
                # using the most recent same weekday
                forecast_index = pd.date_range(
                    start=start_date_forecast + pd.offsets.Hour(1),
                    end=end_date_forecast + pd.offsets.Day(1),
                    freq='1h')
                week_shuffle = start_date_forecast.weekday() - last_full_day.weekday()
                if week_shuffle == 0:
                    forecast_values = hourly_df.loc[last_full_day - pd.offsets.Hour(167):last_full_day, :].values
                elif week_shuffle < 0:
                    week_shuffle += 7
                    forecast_values = np.concatenate((
                        hourly_df.loc[last_full_day - pd.offsets.Hour(167 - week_shuffle * 24):last_full_day, :].values,
                        hourly_df.loc[last_full_day - pd.offsets.Hour(167):last_full_day - pd.offsets.Hour(168 - week_shuffle * 24), :].values
                    ))
                else:
                    forecast_values = np.concatenate((
                        hourly_df.loc[last_full_day - pd.offsets.Hour(167 - week_shuffle * 24):last_full_day, :].values,
                        hourly_df.loc[last_full_day - pd.offsets.Hour(167):last_full_day - pd.offsets.Hour(168 - week_shuffle * 24), :].values
                    ))
                forecast_values = scaler.inverse_transform(forecast_values)
                forecast_values = list(forecast_values) * int(np.ceil(forecast_index.shape[0] / 168.0))
                df = pd.DataFrame(
                    index=forecast_index,
                    data=forecast_values[:forecast_index.shape[0]],
                    columns=[output_col])
            elif nr_days >= 1:
                # If there's less than a week of history, forecast values
                # using the most recent day
                last_full_day = hourly_df.index.max().floor('d')
                if last_full_day > start_date_forecast - pd.offsets.Day(forecast_gap):
                    last_full_day = start_date_forecast - pd.offsets.Day(forecast_gap)
                forecast_index = pd.date_range(
                    start=start_date_forecast + pd.offsets.Hour(1),
                    end=end_date_forecast + pd.offsets.Day(1),
                    freq='1h')
                forecast_values = scaler.inverse_transform(
                    hourly_df.loc[last_full_day - pd.offsets.Hour(23):last_full_day, :].values)
                df = pd.DataFrame(
                    index=forecast_index,
                    data=list(forecast_values) * (forecast_index.shape[0] / 24),
                    columns=[output_col])
            else:
                logger.error('Cannot create forecast because there is not enough '
                    'historical data with requirements history_days: %s '
                    'forecast_gap: %s and start_date_forecast: %s.',
                    history_days, forecast_gap, start_date_forecast)
                df[output_col] = np.nan

        # Convert to UTC
        df.index = df.index.tz_convert('UTC')

        tension_lvl = self.get_property_from_tag(
            tag_name='premise', property_name='tension_lvl')
        if tension_lvl == 'BTE':
            tension_lvl = 'BT'

        load_losses_df = self.load_side_input(
            datasource_id='load_losses',
            channel_id=tension_lvl,
            start=start_date_forecast,
            end=end_date_forecast)
        df['LOAD_LOSSES'] = load_losses_df.resample(rule='H', closed='right', label='right').mean()
        # Apply load losses
        df[output_col] = (1 + df['LOAD_LOSSES']) * df[output_col]

        # Add column to
        self.dataframe = self.dataframe.merge(
            df[[output_col]], how='outer', left_index=True, right_index=True)

        return RuleResult(result=self.dataframe)

    def parse_forecast_dates(self, start_date_forecast, end_date_forecast):
        """Convert the forecast start and end to datetimes

        Args:
            start_date_forecast (str):
            end_date_forecast (str):

        Returns:
            (datetime, datetime): Start and end timestamps.

        """
        import pandas as pd
        if not start_date_forecast and not end_date_forecast:
            df = self.dataframe.loc[:, [self.source_column]].copy()
        if start_date_forecast:
            start_date_forecast = pd.to_datetime(start_date_forecast).tz_localize(
                self.datasource.timezone)
        else:
            start_date_forecast = df.index.min()
        if end_date_forecast:
            end_date_forecast = pd.to_datetime(end_date_forecast).tz_localize(
                self.datasource.timezone)
        else:
            end_date_forecast = df.index.max()
        return start_date_forecast, end_date_forecast

    def get_property_from_tag(self, tag_name, property_name):
        """Get a specific property from specific tag

        Args:
            tag_name (str): Name of the tag.
            property_name (str):  Name of the property.

        Returns:
            str: Value of the target property.

        """
        target_tag = [tag for tag in self.datasource.tags if tag.tag == tag_name]
        if len(target_tag) == 0:
            logger.warning(
                'The datasource {} does not have a {} tag specified.'.format(
                    self.datasource.id, tag_name))
            target_property = None
        else:
            target_tag = {p.key: p.value for p in target_tag[0].properties}
            if not target_tag.get(property_name):
                logger.warning(
                    'The datasource {} does not have a {} property specified.'.format(
                        self.datasource.id, property_name))
                target_property = None
            else:
                target_property = str(target_tag.get(property_name))
        return target_property

    @staticmethod
    def convert_to_hourly_and_scale(dataframe, source_col):
        """Resample data to hourly values and scale data

        Args:
            dataframe (pd.DataFrame): Dataframe with consumption data.
            source_col (str): Column with consumption data.

        Returns:
            (pd.DataFrame, MinMaxScaler): Dataframe with scaled data and
                a scaler object for inverse scaling forecast.

        """
        from sklearn.preprocessing import MinMaxScaler
        df = dataframe.resample(rule='H', closed='right', label='right').sum()
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        df[source_col] = scaler.fit_transform(
            df[source_col].values.reshape(-1, 1)).reshape(-1)
        return df, scaler

    @staticmethod
    def convert_data_to_24_hour_days(df):
        """Normalize data so that each day has 24 hours

        Remove the latter hour from fall DST transition and copy the
        hour after spring DST transition.

        Args:
            df (pd.DataFrame): Original dataframe.

        Returns:
            pd.DataFrame: Dataframe with 24 hours on each day and a
                tz-naive index.

        """
        import pandas as pd
        # Create a tz naive timestamp so we can calculate hour differences between timestamps
        df['NAIVE_TIMESTAMP'] = df.index.tz_localize(None)
        df['HOUR_DIFF'] = df['NAIVE_TIMESTAMP'] - df['NAIVE_TIMESTAMP'].shift()
        # Remove the latter hour in fall DST transition to change 25 -> 24 hours
        df = df.loc[~(df['HOUR_DIFF'] == '00:00:00'), :]
        # Add a copy of the hour after DST transition in spring to change 23 -> 24 hours
        new_rows = df.loc[df['HOUR_DIFF'] == '02:00:00', :]
        new_rows.index = new_rows.index + pd.Timedelta('1s')
        df = df.append(new_rows).sort_index()
        del df['NAIVE_TIMESTAMP']
        del df['HOUR_DIFF']
        return df

    def create_dataset(self, dataframe, source_col, forecast_gap):
        """Create a training set with features and targets

        Duration of the forecast and gap between most recent data
        and beginning of forecast are hardcoded here.

        Args:
            dataframe (pd.DataFrame): Dataframe with consumption data.
            source_col (str): Column with consumption data.
            forecast_gap (int): How many hours between latest data and
                start of forecast there should be.

        Returns:
            (pd.DataFrame, pd.DataFrame): Features and target.

        """
        forecast_hours = 24
        df = dataframe.copy()
        df = self.create_daily_forecast_features(df, forecast_gap)
        df = self.create_historical_features(df, source_col)
        df.dropna(inplace=True)
        df = self.create_forecast_target(df, source_col, forecast_hours, forecast_gap)
        df = self.select_daily_forecasts(df)
        del df[source_col]
        target_cols = [col for col in df.columns if 'FORECAST_TARGET_' in col]
        feature_cols = [col for col in df.columns if col not in target_cols]
        return df.loc[:, feature_cols], df.loc[:, target_cols]

    @staticmethod
    def create_daily_forecast_features(df, forecast_gap):
        """Create daily features used in the forecast

        Args:
            df (pd.DataFrame):
            forecast_gap (int): How many hours between latest data and
                start of forecast there should be.

        Returns:
            pd.DataFrame: Dataframe with daily features.

        """
        import pandas as pd
        from workalendar.europe import Portugal
        cal = Portugal()
        df['DT_TEMP'] = df.index - pd.Timedelta('1s')
        df['DAY_TYPE_D2'] = df['DT_TEMP'].dt.dayofweek.shift(-forecast_gap)
        df = pd.get_dummies(df, columns=['DAY_TYPE_D2'], prefix='DAY_TYPE_D2', drop_first=False)
        df['HOLIDAY_D2'] = df['DT_TEMP'].dt.date.apply(lambda x: cal.is_holiday(x)).astype('int').shift(-forecast_gap)
        del df['DT_TEMP']
        return df

    @staticmethod
    def create_historical_features(df, col):
        """Add hourly features and smoothed features

        The RBF features take an exponentially weighted average of three
        neighbouring hours during the past 4 weeks.

        Args:
            df (pd.DataFrame):
            col (str): Column with consumption data.

        Returns:
            pd.DataFrame: Dataframe with historical features.

        """
        import numpy as np
        normalizer = 4. + 8 * np.exp(-1)
        for i in range(24):
            df['HISTORICAL_CONSUMPTION_{:02d}'.format(i)] = df[col].shift(3*24 + i)
            df['RBF_{:02d}'.format(i)] = (
                np.exp(-1) * df[col].shift(3*24 + i - 1) +
                df[col].shift(3*24 + i) +
                np.exp(-1) * df[col].shift(3*24 + i + 1) +
                np.exp(-1) * df[col].shift(10*24 + i - 1) +
                df[col].shift(10*24 + i) +
                np.exp(-1) * df[col].shift(10*24 + i + 1) +
                np.exp(-1) * df[col].shift(17*24 + i - 1) +
                df[col].shift(17*24 + i) +
                np.exp(-1) * df[col].shift(17*24 + i + 1) +
                np.exp(-1) * df[col].shift(24*24 + i - 1) +
                df[col].shift(24*24 + i) +
                np.exp(-1) * df[col].shift(24*24 + i + 1)
            ) / normalizer
        return df

    @staticmethod
    def create_forecast_target(df, source_col, forecast_hours, forecast_gap):
        """Generate a target for each hour of the forecast

        Args:
            df (pd.DataFrame):
            source_col (str): Column with consumption data.
            forecast_hours (int): Number of hours to forecast.
            forecast_gap (int): How many hours between latest data and
                start of forecast there should be.

        Returns:
            pd.DataFrame: Dataframe with added target columns.

        """
        for i in range(1, forecast_hours + 1):
            df['FORECAST_TARGET_D2_H{:02d}'.format(i)] = df[source_col].shift(-(forecast_gap + i))
        return df

    @staticmethod
    def select_daily_forecasts(df):
        """Forecasts will always start from midnight

        Args:
            df (pd.DataFrame):

        Returns:
            pd.DataFrame: Dataframe with only the training examples
                that start at midnight.

        """
        return df.loc[(df.index.hour == 0) & (df.index.minute == 0), :]

    def train_and_forecast(self, X, y, start_date_forecast, end_date_forecast, scaler):
        """Train a linear regression model and create a forecast

        Args:
            X (pd.DataFrame): features
            y (pd.DataFrame): targets
            start_date_forecast: Date as timestamp where forecast starts.
            end_date_forecast: Date as timestamp where forecast ends.
            scaler (MinMaxScaler): Scaler to inverse transform the forecast
                back to original scale.

        Returns:
            pd.Dataframe: A dataframe with the forecast.

        """
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import Ridge
        X_train = X.loc[y.notnull().all(axis=1), :].loc[:start_date_forecast - pd.offsets.Day(1), :].values.astype(np.float32)
        y_train = y.loc[y.notnull().all(axis=1), :].loc[:start_date_forecast - pd.offsets.Day(1), :].values.astype(np.float32)
        X_pred = X.loc[start_date_forecast:end_date_forecast, :].values.astype(np.float32)
        model = Ridge(alpha=1.5)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        y_pred = scaler.inverse_transform(y_pred)
        # Never forecast negative usage
        y_pred = np.maximum(y_pred, 0.0)
        # Format to fit output channel
        forecast = self.reformat_predictions(
            X.loc[start_date_forecast:end_date_forecast, :], y_pred)
        return forecast

    def reformat_predictions(self, X, y_pred):
        """Reformat the predictions to a long dataframe

        Predictions come in (24, ) arrays which need to be converted to
        a long dataframe with correct timestamps. Newer forecasts should
        overwrite older ones (D+2 forecast should overwrite any other
        forecast).

        Args:
            X (pd.Dataframe):  The forecast input instances (only index
                is used).
            y_pred (list): Predictions of the forecast model.

        Returns:
            pd.Dataframe: A dataframe with the forecast.

        """
        import pandas as pd
        import numpy as np
        forecast_dfs = []
        for ind, date in enumerate(X.index):
            forecast_index = pd.date_range(
                start=date + pd.offsets.DateOffset(days=3) + pd.offsets.Hour(n=1),
                end=date + pd.offsets.DateOffset(days=4),
                freq='H',
                tz=self.datasource.timezone)
            forecast_values = y_pred[ind, :]
            if forecast_index.shape[0] > 24:
                dst_df = forecast_index.to_frame().reset_index(drop=True)
                dst_df[0] = dst_df[0].dt.tz_localize(None)
                dst_df['DIFF'] = dst_df[0] - dst_df[0].shift()
                dst_transition = dst_df.loc[dst_df.DIFF == '00:00:00', :]
                if dst_transition.shape[0] > 0:
                    dst_transition = dst_transition.index[0]
                else:
                    dst_transition = 0
                forecast_values = np.concatenate((
                    forecast_values[:dst_transition],
                    [forecast_values[dst_transition]],
                    forecast_values[dst_transition:]))
            elif forecast_index.shape[0] < 24:
                dst_df = forecast_index.to_frame().reset_index(drop=True)
                dst_df[0] = dst_df[0].dt.tz_localize(None)
                dst_df['DIFF'] = dst_df[0] - dst_df[0].shift()
                dst_transition = dst_df.loc[dst_df.DIFF == '02:00:00', :]
                if dst_transition.shape[0] > 0:
                    dst_transition = dst_transition.index[0]
                else:
                    dst_transition = 0
                forecast_values = np.concatenate((
                    forecast_values[:dst_transition],
                    forecast_values[dst_transition+1:]))
            forecast_dfs.append(pd.DataFrame(
                index=forecast_index,
                data=forecast_values,
                columns=['FORECAST']))
        forecast_df = pd.concat(forecast_dfs)
        return forecast_df.loc[~forecast_df.index.duplicated(keep='last'), :]
