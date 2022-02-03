import dask.dataframe as dd
import pandas as pd
import re
import numpy as np
import os
import time
import seaborn as sns
os.chdir("../dataset")
columns = ['PDG ID', 'Px', 'Py', 'Pz', 'E', 'm', 'Status']

df = dd.read_parquet('myfile3.parquet', engine='pyarrow', npartitions=2)
df = df.loc[0:1000,:]
df.to_parquet('clustertest.parquet', engine='pyarrow')