##cleans the text file containing the analysis results of ansys
##and convert it into a parquetfile
##input :textfile containing output from Rivet Analysis Ran on cluster
##output : parquet file(myfile2.parquet which I later changed datatypes and renamed as myfile3.parquet)


import pandas as pd
import re

import os
import time
start_time = time.time()

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
#os.chdir('/beegfs/bashir/standalone/software/code')

# Print the current working directory

os.chdir("/beegfs/bashir/standalone/software/dataset/")
print("Current working directory: {0}".format(os.getcwd()))

print("--- %s seconds --- for printing directory" % (time.time() - start_time))


# file1 = open('realdata.txt','r')
# file2 = open('realdatacleaned.txt','w')

# reading each line from original
# text file
# for line in file1.readlines():
    
#     # reading all lines that begin 
#     line = re.sub(r"^[ \t\r\n]+","",line)
#     x = re.match("^([A-Za-z]|\/|\*|\=)", line)
#     if not  x :
        
#         file2.write(line)
#         #print(line)

# file1 = open('dataset_withtext.txt','r')
# file2 = open('dataset_cleaned_forpandas.txt','w')
#
# for line in file1:
#
#     # reading all lines that begin
#
#     x = re.match("^([A-Za-z]|\/|\*|\=)", re.sub(r"^[ \t\r\n]+","",line))
#     if not  x :
#
#         file2.write(line)
#         #print(line)
#

#
#
# file1.close()
# file2.close()

#print("--- %s seconds --- for creating cleaned textfile" % (time.time() - start_time))
columns = ['Barcode','PDG ID','Status','Px','Py','Pz' ,'E','m']
dtypes = [int, int, int, float, float, float, float, float] 
dataset = pd.read_csv('dataset_cleaned_forpandas.txt',  delimiter=",",header = None,names = columns)
dataset['m'] = dataset['m'].str.replace(';', '').astype(float)
print(dataset.head())
print(dataset.shape)

print(dataset.info())
print(dataset.describe())
print("--- %s seconds --- for cleaned textfile in pandas dataframe" % (time.time() - start_time))
#dataset.to_csv ('dataframe_tocsv.csv', index = False, header=True)
#dataset.to_pickle('dataframe.pkl')
#dataset.to_parquet('myfile.parquet', engine='fastparquet')
dataset.to_parquet('myfile2.parquet', engine='pyarrow')
print("--- %s seconds --- for storing pandas dataframe to parquet" % (time.time() - start_time))
print("--- %s seconds --- total time" % (time.time() - start_time))

          
   
   
print(dataset.shape)
  
     
      
    # if  not x:
        
        
    # # #     # printing those lines
    # # #       print(line)
          
    # # #     # storing only those lines that 
    # # #     # do not begin with "TextGenerator"
    # 	file2.write(line)
          
# # close and save the files
# file1.close()
# file2.close()




# l = '539         '
# x = re.match("^([A-Za-z]|\/|\*|\=)", l)
# print(x)
# if not x:
#     print(l)

# l = '  12'
# print(l)
# l = re.sub(r"^[ \t\r\n]","",l)

# print(l)   