import pandas as pd
import re
import numpy as np
import os
import time
import seaborn as sns

sns.set(style='darkgrid')
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
# import hvplot.dask
from mlxtend.preprocessing import minmax_scaling
# from matplotlib import pyplot as plt
from scipy import stats

os.chdir("../dataset")
columns = ['PDG ID', 'Px', 'Py', 'Pz', 'E', 'm', 'Status']

df = dd.read_parquet('myfile3.parquet', engine='pyarrow')
#df = dd.read_parquet('clustertest.parquet', engine='pyarrow')


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


feature_name = ['Px', 'Py', 'Pz', 'E', 'm']
normalized_df = normalize(df)
for feature in feature_name:

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    sns.displot(df['Px'], ax=ax, kde=True, legend=False,
            bins=18).set_titles("Original " + feature)
    plt.title("Original " + feature)
    plt.savefig('/beegfs/bashir/standalone/software/code/plots/original'+ feature + '.jpg')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    sns.displot(normalized_df['Px'], ax=ax, kde=True,
            bins=18).set_titles("Normalized " + feature)
    plt.title("Normalized " + feature)
    plt.savefig('/beegfs/bashir/standalone/software/code/plots/normalized' + feature +'.jpg')
    plt.close()
