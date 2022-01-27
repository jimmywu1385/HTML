#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import math

def services_preprocess(df):
    #Count, Quarter, Referred a Friend, Number of Referrals
    for i in range(df.shape[0]):
        if pd.isna(df['Number of Referrals'].iloc[i]):
            if df['Referred a Friend'].iloc[i] == 'Yes':
                df['Number of Referrals'].iloc[i] = 1.0
            else:
                df['Number of Referrals'].iloc[i] = 0.0                           
    df = df.drop(columns=['Count_x', 'Quarter', 'Referred a Friend'], axis=1)

    # Tenure in Months
    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(df[['Tenure in Months', 'Total Revenue']])
    df[['Tenure in Months', 'Total Revenue']] = imp_mean.transform(df[['Tenure in Months', 'Total Revenue']])

    # offer
    offer = ['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E']
    TIM_offer = df.groupby('Offer')['Tenure in Months'].mean()
    for i in range(df.shape[0]):
        if pd.isna(df['Offer'].iloc[i]):
            min = abs(df['Tenure in Months'].iloc[i] - TIM_offer[1])
            min_ind = 1
            for j in range(len(TIM_offer))[2:]:
                if abs(df['Tenure in Months'].iloc[i] - TIM_offer[j]) < min:
                    min = abs(df['Tenure in Months'].iloc[i] - TIM_offer[j])
                    min_ind = j
            df['Offer'].iloc[i] = offer[min_ind]
    df['Offer'] = df['Offer'].astype('category').cat.codes


    # phone service
    imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    df['Phone Service'] = imp.fit_transform(df[['Phone Service']])
    df['Phone Service'] = np.where(df['Phone Service'] == 'Yes', 1, 0)

    # Avg month long charge
    for i in range(df.shape[0]):
        if pd.isna(df['Avg Monthly Long Distance Charges'].iloc[i]):
            if df['Phone Service'].iloc[i] == 'No':
                df['Avg Monthly Long Distance Charges'].iloc[i] = 0.0
    imp_mean.fit(df[['Total Long Distance Charges', 'Avg Monthly Long Distance Charges']])
    df[['Total Long Distance Charges', 'Avg Monthly Long Distance Charges']] = imp_mean.transform(df[['Total Long Distance Charges', 'Avg Monthly Long Distance Charges']])
                
    # Internet service
    for i in range(df.shape[0]):
        if pd.isna(df['Internet Service'].iloc[i]):
            if df['Internet Type'].iloc[i] == 'None':
                df['Internet Service'].iloc[i] = 'No'
            # else:
            #     df['Internet Service'][i] = 'Other'
    df['Internet Service'] = imp.fit_transform(df[['Internet Service']])
    df['Internet Service'] = np.where(df['Internet Service']=='Yes', 1, 0)

    # Multiple line
    for i in range(df.shape[0]):
        if pd.isna(df['Multiple Lines'].iloc[i]):
            if df['Internet Service'].iloc[i] == 'No' and df['Phone Service'].iloc[i] == 'No':
                df['Multiple Lines'].iloc[i] = 'No'
    df['Multiple Lines'] = imp.fit_transform(df[['Multiple Lines']])
    df['Multiple Lines'] = np.where(df['Multiple Lines']=='Yes', 1, 0)


    # Internet Type  Online Security Online Backup Device Protection Plan  Premium Tech Support
    #  Streaming TV  Streaming Movies  Streaming Music  Unlimited Data
    cols = ['Online Security', 'Online Backup', 'Device Protection Plan',  'Premium Tech Support',         'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data']
    for c in cols:
        df[c] = imp.fit_transform(df[[c]]) 
        df[c] = np.where(df[c]=='Yes', 1, 0)     

    # Avg Monthly GB Download
    avg3 = df['Avg Monthly GB Download'].mean()
    for i in range(df.shape[0]):
        if pd.isna(df['Avg Monthly GB Download'].iloc[i]):
            if df['Internet Service'].iloc[i] == 0:
                df['Avg Monthly GB Download'].iloc[i] = 0.0
            else:
                df['Avg Monthly GB Download'].iloc[i] = avg3



    # Contract Paperless Billing  Payment Method
    df['Contract'] = imp.fit_transform(df[['Contract']])
    df['Contract'] = df['Contract'].astype('category').cat.codes
    df = df.drop(['Payment Method', 'Paperless Billing', 'Internet Type'], axis=1)


    # Total Extra Data Charges  Total Refunds   
    df['Total Extra Data Charges'] = df['Total Extra Data Charges'].fillna(0.0)
    df['Total Refunds'] = df['Total Refunds'].fillna(0.0)

    # Avg month long charge
    imp_mean.fit(df[['Monthly Charge', 'Total Charges']])
    df[['Monthly Charge', 'Total Charges']] = imp_mean.transform(df[['Monthly Charge', 'Total Charges']])

    # Total Charges
    imp_mean.fit(df[['Total Revenue', 'Total Charges']])
    df[['Total Revenue', 'Total Charges']] = imp_mean.transform(df[['Total Revenue', 'Total Charges']])

    df = df.drop(['Streaming TV', 'Streaming Movies', 'Streaming Music', 'Total Long Distance Charges', 'Total Extra Data Charges', 'Total Refunds', 'Total Charges'], axis=1)

    return df