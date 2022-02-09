###### stratified sampling 1 % of data to visualize things run on cluster
##input : myfile4.parquet
##output : smallforgraphs.parquet


import pandas as pd
import re
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('/beegfs/bashir/standalone/software/code')
print("Current working directory: {0}".format(os.getcwd()))
os.chdir("../dataset")
print("Current working directory: {0}".format(os.getcwd()))


columns = ['ID','Px','Py','Pz' ,'E','m','Status']
df_noduplicate = pd.read_parquet('myfile4.parquet', engine="pyarrow", columns=columns)
df3 = df_noduplicate.groupby('E', group_keys=False).apply(lambda x: x.sample(frac=0.01))
df3.to_parquet('smallforgraphs.parquet', engine='pyarrow')