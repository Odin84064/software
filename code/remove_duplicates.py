##this file is run on cluster to create a parquet file 'myfile4.parquet' to remove
##all duplicates from 'myfile3.parquet'
# input = myfile3.parquet
# output = myfile4.parquet


import pandas as pd
import re
import numpy as np

import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("../dataset")
print("Current working directory: {0}".format(os.getcwd()))
columns = ['PDG ID','Px','Py','Pz' ,'E','m','Status']
df = pd.read_parquet('myfile3.parquet', engine="pyarrow", columns=columns)
print(df.info())
print(df.memory_usage(deep=True))
df.rename(columns={'PDG ID': 'ID'}, inplace=True)
df_no =df.drop_duplicates(keep='last')
df_no.reset_index(drop=True, inplace=True)
print('##############################################################')
df_no.to_parquet('myfile4.parquet', engine='pyarrow')
print(df_no.info())
print(df_no.memory_usage(deep=True))