#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import numpy as np
import pandas as pd
from functools import reduce

from pandas.io.parsers import read_csv
from services_preprocess import services_preprocess
from location_preprocess import location_satisfaction_preprocess
from demographic_preprocess import demographic_preprocess
from sklearn import preprocessing
# info
final_folder = './'

services_path = os.path.join(final_folder, 'services.csv')
demographic_path = os.path.join(final_folder, 'demographics.csv')
location_path = os.path.join(final_folder, 'location.csv')
satisfaction_path = os.path.join(final_folder, 'satisfaction.csv')
status_path = os.path.join(final_folder, 'status.csv')
dataframe_path = os.path.join(final_folder, 'preprocess.csv')

def preprocess_data():
    # Load Data
    services_df = pd.read_csv(services_path)
    demographic_df = pd.read_csv(demographic_path)
    location_df = pd.read_csv(location_path)
    satisfaction_df = pd.read_csv(satisfaction_path)

    # Merge Data
    data_frames = [services_df, location_df, satisfaction_df, demographic_df]
    df = reduce(lambda left,right: pd.merge(left, right, on=['Customer ID'], how='outer'), data_frames)

    df = services_preprocess(df)
    df = location_satisfaction_preprocess(df)
    df = demographic_preprocess(df)
    df.to_csv(dataframe_path, index=False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--method", type=str, default='linear_model',
                            choices=['random_forest',  'linear_model', 'SVC'], help='sklearn method')
    parser.add_argument("--preprocess", action='store_true', default=False,
                        help="Preprocessing data")
    return parser.parse_args()

if __name__ == '__main__':
    arg = arg_parse()

    if arg.preprocess:
        preprocess_data()

    df = pd.read_csv(dataframe_path)

    status_df = pd.read_csv(status_path)
    # Preprocess
    # status_df['Churn Category'] = status_df['Churn Category'].astype('category').cat.codes
    status_map = {'No Churn':0, 'Competitor':1, 'Dissatisfaction':2, 'Attitude':3, 'Price':4, 'Other':5}
    status_df['Churn Category'] = status_df['Churn Category'].map(status_map)


    # Prepare Train Data
    data_frames = [df, status_df]
    train_df = reduce(lambda  left,right: pd.merge(left,right,on=['Customer ID'],
                                                how='right'), data_frames)
    # drop column
    drop_columns = ['Customer ID']
    train_df = train_df.drop(drop_columns, axis=1)

    # down sample
    No_churn_cnt = train_df[train_df['Churn Category'] == 0].count()['Churn Category']
    train_df.sort_values(by=['Churn Category'])
    train_df = train_df[int(No_churn_cnt*0.5):]

    # train and validation split
    x_label = train_df.drop(['Churn Category'], axis=1)
    y_label = train_df['Churn Category']
    x_train, x_val, y_train, y_val = train_test_split(x_label, y_label, test_size=0.2, random_state=42)

    # Prepare Test Data
    testID_path = os.path.join(final_folder, 'Test_IDs.csv')
    testID = pd.read_csv(testID_path)
    data_frames = [testID, df]
    test_df = reduce(lambda  left,right: pd.merge(left,right,on=['Customer ID'],
                                                how='left'), data_frames)
    test_df = test_df.drop(['Customer ID'], axis=1)
    test_df.head()

    if arg.method == 'linear_model':
        # Linear Model15 15 15 20
        from sklearn.neural_network import MLPClassifier
        scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_scale = scaler.transform(x_train)
        clf = MLPClassifier(alpha=1e-2, hidden_layer_sizes=(15), random_state=1)
        clf.fit(x_scale, y_train)
    
    if arg.method == 'random_forest':
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        #clf = RandomForestClassifier(max_depth=200, random_state=1, criterion='entropy')
        #clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=2, random_state=0)
        scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_scale = scaler.transform(x_train)
        clf = XGBClassifier(n_estimators=3, learning_rate= 0.5, max_depth=6)
        clf.fit(x_scale, y_train)
    
    if arg.method == 'svc':
        # SVC
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf')
        clf.fit(x_train, y_train)

    y_pred = clf.predict(test_df)
    
    y_pred_train = clf.predict(x_train)
    y_pred_val = clf.predict(x_val)
    print(f1_score(y_train, y_pred_train, average='macro'))
    print(f1_score(y_val, y_pred_val, average='macro'))
    # print(clf.score(x_train, y_train))
    # print(clf.score(x_val, y_val))

    testID['Churn Category'] = y_pred
    testID.to_csv('submission.csv', index=False)


