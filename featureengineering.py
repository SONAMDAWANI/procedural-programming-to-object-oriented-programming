import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt
from functools import reduce
from sklearn import preprocessing


### Calculate ratio of fraudulent transaction by each categorical variable

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def calculate_ratio_fraud(analysis_df, sel_var):
    '''
    Args: 
        analysis_df: Dataframe with transaction level details
        sel_var: variable of interest for ratio calculation (country, device, )
        
    Output:
        Dataframe that merges the ratio of fraudulent transaction specific to selected variable to analysis_df
    '''
    tmp = analysis_df.groupby([sel_var, 'class']).user_id.nunique() \
        .unstack(level=1) \
        .reset_index() \
        .rename(columns={0: 'Not Fraud', 1: 'Fraud'}).fillna(0.0)
    tmp['ratio_fraud_' + sel_var] = tmp['Fraud'] / (tmp['Fraud'] + tmp['Not Fraud'])
    tmp['num_trans_' + sel_var] = tmp['Fraud'] + tmp['Not Fraud']
    return analysis_df[['user_id', sel_var]] \
        .merge(tmp[[sel_var, 'ratio_fraud_' + sel_var, 'num_trans_' + sel_var]], on=sel_var)


def calculate_time_latency(df):
    '''Calculates the difference between sign up and purchase times'''
    df['time_latency'] = (df.purchase_time - df.signup_time).dt.total_seconds() / 60 / 60
    return df


def merge_multiple_dataframes(dfs, key, method):
    '''
    Args: 
        dfs list of dataframes to be merged
        key list of column names to be used for join
        method merge-type(inner, outer, left)
        
    Output:
        combined dataframe
    '''
    return reduce(lambda left, right: pd.merge(left, right, on=key, how=method), dfs)


def apply_label_encoding(df):
    return MultiColumnLabelEncoder(columns=df.columns).fit_transform(df)


def create_features(path_to_analysis_dataset):
    '''
    Args: 
        path to analysis dataset
        
    Output:
        Dataframe transforms raw data into specific feature elements ready to be used for classfication
    '''
    analysis_df = pd.read_csv(path_to_analysis_dataset) \
        .drop('Unnamed: 0', axis=1)

    ## Convert signup and purchase times to pandas datetime
    analysis_df.signup_time = pd.to_datetime(analysis_df.signup_time, format='%m/%d/%Y %H:%M')
    analysis_df.purchase_time = pd.to_datetime(analysis_df.purchase_time, format='%m/%d/%Y %H:%M')

    ## Fill missing values with NA
    analysis_df = analysis_df.fillna('NA')

    ## Calucate fraud ratios
    fraud_by_dev = calculate_ratio_fraud(analysis_df, 'device_id')
    fraud_by_country = calculate_ratio_fraud(analysis_df, 'country')
    fraud_by_age = calculate_ratio_fraud(analysis_df, 'age')
    fraud_by_gender = calculate_ratio_fraud(analysis_df, 'sex')
    fraud_by_source = calculate_ratio_fraud(analysis_df, 'source')
    fraud_by_browser = calculate_ratio_fraud(analysis_df, 'browser')

    ## Calculate latency between sign-up and purchase time
    latency_df = calculate_time_latency(analysis_df)

    ## Merge all features
    feature_df = merge_multiple_dataframes([
        fraud_by_dev, fraud_by_country,
        fraud_by_gender,
        fraud_by_age,
        fraud_by_browser,
        fraud_by_source,
        analysis_df[['user_id', 'purchase_value', 'class']],
        latency_df[['user_id', 'time_latency']]
    ],
        key=['user_id'], method='outer')

    df_cat = apply_label_encoding(feature_df[['country', 'sex', 'browser', 'source']])
    return pd.concat([feature_df.drop(['country', 'sex', 'browser', 'source'], axis=1), df_cat], axis=1).set_index(
        ['user_id', 'device_id'])
    # return df_cat
