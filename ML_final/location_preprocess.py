#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def location_satisfaction_preprocess(df):
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    df["Satisfaction Score"] = imp.fit_transform(df[["Satisfaction Score"]])
    df = df.drop(['Count_y', 'Country', 'State', 'Lat Long', 'Longitude', 'Latitude'], axis=1)

    city_zip = df[['City', 'Zip Code']]
    city_zip = city_zip.drop_duplicates(subset=['City'])
    def city2zipcode(x):
        zip_code = city_zip.loc[city_zip['City']==x]['Zip Code'].values
        if len(zip_code) == 0:
            return np.nan
        return zip_code[0]
    def zipcode2city(x):
        city = city_zip.loc[city_zip['Zip Code']==x]['City'].values
        if len(city) == 0:
            return np.nan
        return city[0]

    # city -> zip code
    df.loc[np.isnan(df['Zip Code']), 'Zip Code'] = df.loc[np.isnan(df['Zip Code']), 'Zip Code'].combine_first(df['City'].map(city2zipcode))
    # zip code -> city
    df.loc[df['City'].isnull(), 'City'] = df.loc[df['City'].isnull(), 'City'].combine_first(df['Zip Code'].map(zipcode2city))

    # city & zipcode are nan
    imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    df["Zip Code"] = imp.fit_transform(df[["Zip Code"]])
    df = df.drop(['City'], axis=1)
    return df

