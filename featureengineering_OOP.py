import pandas as pd
from functools import reduce
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


class GetDataframeFromCSV:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, path_to_analysis_dataset):
        """
        Convert signup and purchase times to pandas datetime
        """
        df = pd.read_csv(path_to_analysis_dataset) \
            .drop('Unnamed: 0', axis=1)
        return df

    def fit_transform(self, df, y=None):
        return self.fit().transform(df)


class ConvertToPandasDatetime:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, analysis_df):
        """
        Convert signup and purchase times to pandas datetime
        """
        analysis_df.signup_time = pd.to_datetime(analysis_df.signup_time, format='%m/%d/%Y %H:%M')
        analysis_df.purchase_time = pd.to_datetime(analysis_df.purchase_time, format='%m/%d/%Y %H:%M')
        return analysis_df

    def fit_transform(self, df, y=None):
        return self.fit().transform(df)


class HandleMissingValues:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, analysis_df):
        """
        Convert signup and purchase times to pandas datetime
        """
        analysis_df = analysis_df.fillna('NA')
        return analysis_df

    def fit_transform(self, analysis_df, y=None):
        return self.fit().transform(analysis_df)


class DataframeListToBeMerged:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, analysis_df):
        dataframe_list = [CalculateRatioFraud('device_id').fit_transform(analysis_df),
                          CalculateRatioFraud('country').fit_transform(analysis_df),
                          CalculateRatioFraud('sex').fit_transform(analysis_df),
                          CalculateRatioFraud('age').fit_transform(analysis_df),
                          CalculateRatioFraud('browser').fit_transform(analysis_df),
                          CalculateRatioFraud('source').fit_transform(analysis_df),
                          analysis_df[['user_id', 'purchase_value', 'class']],
                          CalculateTimeLatency().fit_transform(analysis_df)[['user_id', 'time_latency']]]
        return dataframe_list

    def fit_transform(self, analysis_df, y=None):
        return self.fit().transform(analysis_df)


class MergeMultipleDataframes:
    def __init__(self, key, method):
        self.key = key
        self.method = method

    def fit(self):
        return self

    def transform(self, dfs):
        """
        Args:
            dfs list of dataframes to be merged
            key list of column names to be used for join
            method merge-type(inner, outer, left)

        Output:
            combined dataframe
        """
        return reduce(lambda left, right: pd.merge(left, right, on=self.key, how=self.method), dfs)

    def fit_transform(self, dfs, y=None):
        return self.fit().transform(dfs)


class LabelEncodeAndConcat:
    def __init__(self, encode_columns):
        self.encode_columns = encode_columns

    def fit(self):
        return self

    def transform(self, feature_df):
        df_cat = MultiColumnLabelEncoder(columns=self.encode_columns).fit_transform(feature_df[self.encode_columns])
        encode_feature_df = pd.concat([feature_df.drop(self.encode_columns, axis=1), df_cat], axis=1)
        return encode_feature_df

    def fit_transform(self, feature_df, y=None):
        return self.fit().transform(feature_df)


class IndexDataframe:
    def __init__(self, indexby_columns):
        self.indexby_columns = indexby_columns

    def fit(self):
        return self

    def transform(self, feature_df):
        feature_df = feature_df.set_index(self.indexby_columns)
        return feature_df

    def fit_transform(self, feature_df, y=None):
        return self.fit().transform(feature_df)


class MultiColumnLabelEncoder:
    """
    Calculate ratio of fraudulent transaction by each categorical variable
    """

    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self):
        return self  # not relevant here

    def transform(self, x):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = x.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, x, y=None):
        return self.fit().transform(x)


class CalculateRatioFraud:
    def __init__(self, sel_var):
        self.sel_var = sel_var

    def fit(self, analysis_df):
        """
        Args:
            analysis_df: Dataframe with transaction level details
            sel_var: variable of interest for ratio calculation (country, device, )

        Output:
            Temporary dataframe that contains information needed to transform the input dataframe.
        """
        self.tmp = analysis_df.groupby([self.sel_var, 'class']).user_id.nunique() \
            .unstack(level=1) \
            .reset_index() \
            .rename(columns={0: 'Not Fraud', 1: 'Fraud'}).fillna(0.0)
        self.tmp['ratio_fraud_' + self.sel_var] = self.tmp['Fraud'] / (self.tmp['Fraud'] + self.tmp['Not Fraud'])
        self.tmp['num_trans_' + self.sel_var] = self.tmp['Fraud'] + self.tmp['Not Fraud']
        return self

    def transform(self, analysis_df):
        """
        Args:
            analysis_df: Dataframe with transaction level details
            sel_var: variable of interest for ratio calculation (country, device, )

        Output:
            Dataframe that merges the ratio of fraudulent transaction specific to selected variable to analysis_df
        """
        return analysis_df[['user_id', self.sel_var]] \
            .merge(self.tmp[[self.sel_var, 'ratio_fraud_' + self.sel_var, 'num_trans_' + self.sel_var]],
                   on=self.sel_var)

    def fit_transform(self, analysis_df, y=None):
        return self.fit(analysis_df).transform(analysis_df)


class CalculateTimeLatency:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, df):
        """
        Calculates the difference between sign up and purchase times
        """
        df['time_latency'] = (df.purchase_time - df.signup_time).dt.total_seconds() / 60 / 60
        return df

    def fit_transform(self, df):
        return self.fit().transform(df)


class CreateFeature:
    def __init__(self):
        self.abc = None

    def fit(self):
        return self

    def transform(self, path_to_analysis_dataset):
        createfeature_pipeline = Pipeline([('GetDataframeFromCSV', GetDataframeFromCSV()),
                                           ('ConvertToPandasDatetime', ConvertToPandasDatetime()),
                                           ('HandleMissingValues', HandleMissingValues()),
                                           ('DataframeListToBeMerged', DataframeListToBeMerged()),
                                           ('MergeMultipleDataframes',
                                            MergeMultipleDataframes(key=['user_id'], method='outer')),
                                           ('LabelEncodeAndConcat',
                                            LabelEncodeAndConcat(['country', 'sex', 'browser', 'source'])),
                                           ('IndexDataframe', IndexDataframe(['user_id', 'device_id']))
                                           ])

        feature_df = createfeature_pipeline.fit_transform(path_to_analysis_dataset)
        return feature_df

    def fit_transform(self, path_to_analysis_dataset):
        return self.fit().transform(path_to_analysis_dataset)
