#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer, SimpleImputer

def demographic_preprocess(df):
    df = df.drop(columns=['Under 30', 'Count', 'Senior Citizen', 'Dependents'], axis=1)
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df["Age"] = imp.fit_transform(df[["Age"]])
    df["Number of Dependents"] = imp.fit_transform(df[["Number of Dependents"]])

    df["Gender"] = df["Gender"].map({'Male':1, 'Female':0})
    df["Married"] = df["Married"].map({'Yes':1, 'No':0})
    df["Gender"] = imp.fit_transform(df[["Gender"]])
    df["Married"] = imp.fit_transform(df[["Married"]])

    df = df.drop(['Married'], axis=1)
    return df

