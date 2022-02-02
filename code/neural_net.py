import pandas as pd
import re
import numpy as np

import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
start_time = time.time()

os.chdir("../dataset")

#dataset = pd.read_pickle('dataframe.pkl')
columns = ['PDG ID','Px','Py','Pz' ,'E','m','Status']
dataset = pd.read_parquet('myfile2.parquet', engine="pyarrow", columns=columns)
print(dataset.memory_usage(deep=True))
#print(dataset.head())
#print(dataset.info())
#print(dataset.describe())

#dataset["PDG ID"] = pd.to_numeric(dataset["PDG ID"], downcast="unsigned")

#dataset[["Px", "Py", "Pz" , "E", "m"]] = dataset[["Px", "Py", "Pz", "E", "m"]].apply(pd.to_numeric, downcast="float")
#print(dataset.memory_usage(deep=True))

print("--- %s seconds --- for reading piquet in pandas" % (time.time() - start_time))


########################################################################################################
#basic data exploration

print(dataset.describe())

# get the number of missing data points per column
missing_values_count = dataset.isnull().sum()
print(missing_values_count[0:7])
#0 missing values


### scaling and normalizing data
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
sns.histplot(dataset['m'], ax=ax, kde=True, legend=False)
ax.set_title("Original Data")



