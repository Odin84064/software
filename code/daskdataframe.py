import pandas as pd
import re
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import hvplot.dask
from mlxtend.preprocessing import minmax_scaling
#from matplotlib import pyplot as plt
from scipy import stats

os.chdir("../dataset")
columns = ['PDG ID', 'Px', 'Py', 'Pz', 'E', 'm', 'Status']

df = dd.read_parquet('myfile3.parquet', engine='pyarrow', npartitions=2)
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


feature_name = ['Px','Py','Pz' ,'E','m']
normalized_df = normalize(df)
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(df['Px'], ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Px")
sns.histplot(normalized_df['Px'], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized Px")
plt.show()
