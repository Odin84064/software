#cleans the text file containing the analysis results of ansys and convert it into a parquetfile

##input :textfiles containing output from Rivet Analysis Ran on cluster having ids consolidated per event
#mc15validconsolidated.txt contains the ansys dataset for 100000 events of the first directory(signal)
#mc1513tevconsolidated.txt contains the ansys dataset for 100000 events of the second directory(background)

##output : parquet file
#100000eventsconsolidated.parqeut having one row as:
#n_211 ,'n_211', 'meanPx_211', 'meanPy_211', 'meanPz_211', 'meanE_211','n_22', 'meanPx_22', 'meanPy_22', 'meanPz_22', 'meanE_22'











import pandas as pd
import re
import numpy as np
import os
from itertools import zip_longest
import time
from pathlib import Path
import csv



# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))





os.chdir("../dataset/")
print("Current working directory: {0}".format(os.getcwd()))




#removes all words
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
        y = re.search('\S', line)
        if not x and y:
            file2.write(line)

    file1.close()
    file2.close()
    return output_file








def only_commonids(input_file):
    '''

    :param input_file: mcvalidconsolidatedcleaned,mctevconsolidatedcleaned
    :return: list of list with id 22 and 211 per event(as a sublist)
    '''
    #convert file to list of list
    with open(input_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    chunks = {}
    first_elements = []
    count = 0
    event_ids = []

    for item in data:

        item.pop()
        chunks[count] = [list(np.float_(item[x:x+6])) for x in range(0, len(item), 6)]
        event_ids.append(chunks[count][0])
        count += 1
    #list of dictionaries with particle ids as keys
    d = []
    i = range(len(chunks))
    for item in i:
        l = {}
        for lists in chunks[item]:
            l[lists[0]] = lists[1:]
        d.append(l)
    #particles id in each event
    particles_event = []
    for item in d:
        particles_event.append(list(item.keys()))
    # common particles in each event
    elements_in_all = list(set.intersection(*map(set, particles_event)))
    #list of dictionaries with two keys i.e. 211 and 22 for each event
    d2 = []
    for item in d:

        item = dict((k, item[k]) for k in elements_in_all if k in item)
        d2.append(item)
    #list of list with each sublist is the [211,total number ,px,py,pz,22.otal number ,px,py,pz]
    dictlist = []
    for item in d2:
        temp = []

        for key, value in item.items():
            temp = value
            temp.insert(0, key)
            dictlist.append(temp)
    x = iter(dictlist)
    dictlist= [a+b for a, b in zip_longest(x, x, fillvalue=[])]

    return  dictlist


def to_parquet(list1 , list2 ):
    '''

    :param list1: list of list of mcvalidconsolidated
    :param list2: list of list of mctevconsolidated
    :return: parquet file containing background and signal data with common IDs(211,222) consolidated per event
    '''
    columns = ['ID_211', 'n_211', 'meanPx_211', 'meanPy_211', 'meanPz_211', 'meanE_211',
               'ID_22', 'n_22', 'meanPx_22', 'meanPy_22', 'meanPz_22', 'meanE_22']
    df1 = pd.DataFrame(list1, columns=columns)
    df2 = pd.DataFrame(list2, columns=columns)
    df1['Status'] = 1
    df2['Status'] = 0
    datasets = [df1, df2]
    df = pd.concat(datasets)
    df.reset_index(drop=True, inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop(['ID_211', 'ID_22'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet('100000eventsconsolidated.parquet', engine='pyarrow')
    return df







#function call
#abc = clean_text('10cons.txt','10conscleaned.txt')
#d2 = only_commonids(abc)

#cleaned file of spaces
mcvalid = clean_text('mc15validconsolidated.txt','mc15validconsolidatedcleaned.txt')
#list of list of common ids for signal
mcvaliddic= only_commonids(mcvalid)

#cleaned files of spaces
mctev = clean_text('mc1513tevconsolidated.txt','mc1513tevconsolidatedcleaned.txt')
#list of list of common ids for noice
mctevdic = only_commonids(mctev)

#finalized dataframe and parquet file which contains consolidated common ids per event
df = to_parquet(mcvaliddic,mctevdic)








