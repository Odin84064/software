import pandas as pd
import os
os.chdir("/beegfs/bashir/standalone/software/dataset/")
columns = ['Barcode','PDG ID','Status','Px','Py','Pz' ,'E','m']
dtypes = [int, int, int, float, float, float, float, float]
dataset = pd.read_csv('dataset_cleaned_forpandas.txt',  delimiter=",",header = None,names = columns)
dataset['m'] = dataset['m'].str.replace(';', '').astype(float)
dataset.to_parquet('myfile2.parquet', engine='pyarrow')
