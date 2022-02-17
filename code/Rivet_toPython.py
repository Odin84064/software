#cleans the text file containing the analysis results of ansys and convert it into a parquetfile

##input :textfile containing output from Rivet Analysis Ran on cluster
#mc15valid.txt contains the ansys dataset for 10000 events of the first directory
#mc1513tev.txt contains the ansys dataset for 10000 events of the second directory

##output : parquet file
#1) complete events of first directory "/beegfs/hirsch/sfsscratch/PoolFiles/"mc15_valid.950214.PhPy8_ttbar_CMCpwg_Monash_valid.evgen.EVNT.e8324_tid24450207_00
# (myfile2.parquet which I later changed datatypes and renamed as myfile3.parquet)
#2) 10000 events of the first and second directory "/beegfs/hirsch/sfsscratch/PoolFiles/mc15_13TeV.950507.PhH7EG_ttbar_hdamp258p75_nonallhad_cluster_valid.evgen.EVNT.e8419
#combined and saved as final10000events.parquet to train neural network



import pandas as pd
import re
import numpy as np
import os
import time
from pathlib import Path
start_time = time.time()

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))





os.chdir("../dataset/")
print("Current working directory: {0}".format(os.getcwd()))

print("--- %s seconds --- for printing directory" % (time.time() - start_time))


def clean_text_twofiles(input_files, output_files):
    '''
    :param input_files: list of text files containing the analysis results of ansys
    :param output_files: list of  empty text files
    :return: list of text files (output_files being filled) after removal of extra words and spaces
    '''


    for input_item,output_item in zip(input_files,output_files):

        file1 = open(input_item, 'r')
        filename = Path(output_item)
        filename.touch(exist_ok=True)  # will create file, if it exists will do nothing
        file2 = open(filename,'w')

        for line in file1:

            # reading all lines that begin

            x = re.match("^([A-Za-z]|\/|\*|\=)", re.sub(r"^[ \t\r\n]+", "", line))
            if not x:
                file2.write(line)

        file1.close()
        file2.close()
    return output_files

def text_to_pandas_twofiles(input_files):
    '''

    :param input_files: list of text files containing the Rivet analysis results cleaned from clean_text_twofiles function
    :return: saves a parquet file after converting these files to pandas and merging them
    '''
    datasets = []
    for index,input_file in enumerate(input_files):
        print('The index is')
        print(index)
        columns = ['Barcode', 'PDG_ID', 'Status', 'Px', 'Py', 'Pz', 'E', 'm']
        datasets.append(pd.read_csv(input_file, delimiter=",", header=None, names=columns))
        datasets[index].drop('m', axis=1, inplace=True)
        datasets[index].drop('Barcode', axis=1, inplace=True)
        if input_file == 'mc1513tevcleaned.txt':
            datasets[index]['Status'] = 0
        datasets[index] = datasets[index].astype({'PDG_ID': np.int16, 'Status': np.int16})
    df = pd.concat(datasets)
    df.reset_index(drop=True, inplace=True)
    df_no = df.drop_duplicates(keep='last')
    df_no.reset_index(drop=True, inplace=True)

    df.to_parquet('final10000events.parquet', engine='pyarrow')
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df['Status'].value_counts())


def clean_text(input_file, output_file):
    '''

    :param input_file: text file containing the analysis results of ansys
    :param output_file: empty text file
    :return: text file (output_file being filled) after removal of extra words and spaces
    '''
    file1 = open(input_file, 'r')
    file2 = open(output_file, 'w')

    for line in file1:

        # reading all lines that begin

        x = re.match("^([A-Za-z]|\/|\*|\=)", re.sub(r"^[ \t\r\n]+", "", line))
        if not x:
            file2.write(line)

    file1.close()
    file2.close()
    return output_file


def text_to_pandas(input_file):
    '''

    :param input_file: text file containing the Rivet analysis results cleaned from clean_text function
    :return: saves a parquet file after converting this file to pandas
    '''
    columns = ['Barcode', 'PDG_ID', 'Status', 'Px', 'Py', 'Pz', 'E', 'm']
    dataset = pd.read_csv(input_file, delimiter=",", header=None, names=columns)
    dataset.drop('m', axis=1, inplace=True)
    dataset.drop('Barcode', axis=1, inplace=True)
    if input_file == 'mc1513tevcleaned.txt':
        dataset['Status'] = 0

    dataset = dataset.astype({'PDG_ID': np.int16, 'Status': np.int16})
    df_no = dataset.drop_duplicates(keep='last')
    df_no.reset_index(drop=True, inplace=True)

    dataset.to_parquet(input_file.replace('.txt', '') + '.parquet', engine='pyarrow')
    print(dataset.head())
    print(dataset.shape)
    print(dataset.info())
    print(dataset.describe())





# cleaning the rivet analysis of first and second directory of 10000 events and merging them into a pandas
#converting the pandas to final10000events.parquet to train neural network
#mc15valid.txt status column is 1 in pandas dataframe
#mc1513tev.txt status column is 0 in pandas datafram
input_files = ['mc15valid.txt','mc1513tev.txt']
output_files = ['mcvalidcleaned.txt', 'mc1513tevcleaned.txt']

events_10000 = clean_text_twofiles(input_files,output_files)
text_to_pandas_twofiles(events_10000)




#converting analyses of two directories one by one in pandas and parquet just to compare and sanity check with when
#done in a combined manner

##mc15valid.txt contains the ansys dataset for 10000 events of the first directory
#mc15validcleaned.txt is file after removing unwanted text and spaces

#first_dataset = clean_text('mc15valid.txt','mc15validcleaned.txt')
#text_to_pandas(first_dataset)


##mc1513tev.txt contains the ansys dataset for 10000 events of the second directory
#mc1513tevcleaned.txt is file after removing unwanted text and spaces
#second_dataset = clean_text('mc1513tev.txt','mc1513tevcleaned.txt')
#text_to_pandas(second_dataset)























