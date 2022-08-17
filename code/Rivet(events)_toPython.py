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

    return  dictlist,data

def only_commonids_modified(input_file):
    '''
    modified to process file containing pt and eta instead of Px,Py ,Pz and E
    :param input_file: cleaned output file from function clean_text which processes the output of Rivet
    :return: list of list named as dictlist which contains only the id's common in all events in every sublist and relevant data
    '''
    #convert file to list of list after removing trailing spaces and trailing commas
    data = []

    with open(input_file) as file:
        for line in file:
            event = []
            line = line[:-1].replace(',\n', '\n')
            line = line.rstrip()
            line = line.rstrip(',')


            for numstr in line.split(","):
                if numstr:
                    try:
                        numFl = float(numstr)
                        event.append(numFl)


                    except ValueError as e:
                        print(e)
            data.append(event)

    chunks = {}
    first_elements = []
    count = 0
    event_ids = []

    for item in data:

        #item.pop()
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


def to_parquet(list1 , list2):
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

    df.to_parquet('event/7/event7.parquet', engine='pyarrow')
    return df

def to_parquet_modified(list1 , list2 , name ):
    '''
      modified to contain new features Pt and eta instead of Px,Py,Pz and E
      takes name of parquet file as an input also

    :param list1: list of list of signal i.e. status = 1 assigned here before merging with background df
    :param list2: list of list of background i.e. status = 0 assigned here before merging with signal df
    :param name : name of parquetfile to be saved
    :return: parquet file containing background and signal data with common IDs(211,222) consolidated per event
    '''
    columns = ['ID_211', 'n_211', 'meanPt_211', 'meaneta_211',
               'ID_22', 'n_22', 'meanPt_22', 'meaneta_22']
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

    df.to_parquet(name + '.parquet', engine='pyarrow')
    return df



def to_parquet_modified_var(list1 , list2 , name ):
    '''
      modified to contain new features Pt and eta and their variance
      takes name of parquet file as an input also

    :param list1: list of list of signal i.e. status = 1 assigned here before merging with background df
    :param list2: list of list of background i.e. status = 0 assigned here before merging with signal df
    :param name : name of parquetfile to be saved
    :return: parquet file containing background and signal data with common IDs(211,222) consolidated per event
    '''
    columns = ['ID_211', 'n_211', 'meanPt_211', 'meaneta_211', 'varPt_211', 'vareta_211',
               'ID_22', 'n_22', 'meanPt_22', 'meaneta_22','varPt_22', 'vareta_22']
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

    df.to_parquet(name + '.parquet', engine='pyarrow')
    return df
def to_parquet_events(signal_df, background_df, name):
    '''


    :param list1: df of signal i.e. status = 1 assigned here before merging with background df
    :param list2: df of background i.e. status = 0 assigned here before merging with signal df
    :param name : name of parquetfile to be saved
    :return: parquet file containing background and signal data with common IDs(211,222),n_11,n_13,'meanPt_211', 'meaneta_211',
    'varPt_211','vareta_211', 'n_22', 'meanPt_22', 'meaneta_22', 'varPt_22','vareta_22' consolidated per event
    '''

    # signal df
    signal_df['Status'] = 1
    # background df
    background_df['Status'] = 0
    datasets = [signal_df, background_df]
    df = pd.concat(datasets)
    df.reset_index(drop=True, inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_parquet(name + '.parquet', engine='pyarrow')
    return df
#function call
#abc = clean_text('10cons.txt','10conscleaned.txt')
#d2 = only_commonids_modified(abc)

# #cleaned file of spaces
# mcvalid = clean_text('mc15validconsolidated.txt','mc15validconsolidatedcleaned.txt')
# #list of list of common ids for signal
# mcvaliddic= only_commonids(mcvalid)
#
# #cleaned files of spaces
# mctev = clean_text('mc1513tevconsolidated.txt','mc1513tevconsolidatedcleaned.txt')
# #list of list of common ids for noice
# mctevdic = only_commonids(mctev)
#
# #finalized dataframe and parquet file which contains consolidated common ids per event
# df = to_parquet(mcvaliddic,mctevdic)




# #cleaned file of spaces
# mcvalid = clean_text('signalconsolidated.txt','signalconsolidatedcleaned.txt')
# #list of list of common ids for signal
# mcvaliddic= only_commonids(mcvalid)
#
# # #cleaned files of spaces
# mctev = clean_text('backgroundconsolidated.txt','backgroundsolidatedcleaned.txt')
# # #list of list of common ids for noice
# mctevdic = only_commonids(mctev)
# #
# # #finalized dataframe and parquet file which contains consolidated common ids per event
# df = to_parquet(mcvaliddic,mctevdic)







# #cleaned file of spaces
# mcvalid = clean_text('signalconsolidated.txt','signalconsolidatedcleaned.txt')
# #list of list of common ids for signal
# mcvaliddic= only_commonids(mcvalid)
#
# # #cleaned files of spaces
# mctev = clean_text('backgroundconsolidated.txt','backgroundsolidatedcleaned .txt')
# # #list of list of common ids for noice
# mctevdic = only_commonids(mctev)
# #
# # #finalized dataframe and parquet file which contains consolidated common ids per event
# df = to_parquet(mcvaliddic,mctevdic)
#

def rivet_to_df(input_file):
    '''

    :param input_file: cleaned output file from function clean_text without any unnecessary text
    :return: dataframe with 100000 rows as in total events and n_11 and n_13 as columns to show number of these particles in every event
             other columns are 'ID_211', 'n_211', 'meanPt_211','meaneta_211', 'varPt_211', 'vareta_211',
               'ID_22', 'n_22', 'meanPt_22', 'meaneta_22','varPt_22', 'vareta_22'


    '''
    ##part1 : creates a dataframe df1 with n_11 and n_13 as columns to show number of these particles in every event
    # contains the file as list of list of floats
    data = []

    # leading and trailing spaces and commas are removed,also string is converted to floats
    with open(input_file) as file:
        for line in file:
            event = []
            line = line[:-1].replace(',\n', '\n')
            line = line.rstrip()
            line = line.rstrip(',')
            # print(line)

            for numstr in line.split(","):
                if numstr:
                    try:
                        numFl = float(numstr)
                        event.append(numFl)


                    except ValueError as e:
                        print(e)
            data.append(event)

    # list of dictionieries of size 100000.Each dictionery store n_11 and n_13 for each event


    ##part 2: find out common particles in each event i.e. 22 and 211 and create a dataframe df2 with columns
    ##'ID_211', 'n_211', 'meanPt_211','meaneta_211', 'varPt_211', 'vareta_211',
    ##        'ID_22', 'n_22', 'meanPt_22', 'meaneta_22','varPt_22', 'vareta_22'

    chunks = {}
    first_elements = []
    count = 0
    event_ids = []

    for item in data:
        # item.pop()
        chunks[count] = [list(np.float_(item[x:x + 4])) for x in range(0, len(item), 4)]
        event_ids.append(chunks[count][0])
        count += 1
    # list of dictionaries with particle ids as keys
    d = []
    i = range(len(chunks))
    for item in i:
        l = {}
        for lists in chunks[item]:
            l[lists[0]] = lists[1:]
        d.append(l)
    # particles id in each event
    particles_event = []
    for item in d:
        particles_event.append(list(item.keys()))
    # common particles in each event
    elements_in_all = list(set.intersection(*map(set, particles_event)))
    # list of dictionaries with two keys i.e. 211 and 22 for each event
    d2 = []
    for item in d:
        item = dict((k, item[k]) for k in elements_in_all if k in item)
        d2.append(item)
    # list of list with each sublist is the [211,total number ,px,py,pz,22.otal number ,px,py,pz]
    dictlist = []
    for item in d2:
        temp = []

        for key, value in item.items():
            temp = value
            temp.insert(0, key)
            dictlist.append(temp)
    x = iter(dictlist)
    dictlist = [a + b for a, b in zip_longest(x, x, fillvalue=[])]
    columns = ['ID_211', 'n_211', 'meanPt_211', 'meaneta_211',
               'ID_22', 'n_22', 'meanPt_22', 'meaneta_22']
    df= pd.DataFrame(dictlist, columns=columns)
    df.drop(['ID_211', 'ID_22'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)



    return df
def rivet_to_df_events(input_file):
    '''

    :param input_file: cleaned output file from function clean_text without any unnecessary text
    :return: dataframe with 100000 rows as in total events and n_11 and n_13 as columns to show number of these particles in every event
             other columns are 'ID_211', 'n_211', 'meanPt_211','meaneta_211', 'varPt_211', 'vareta_211',
               'ID_22', 'n_22', 'meanPt_22', 'meaneta_22','varPt_22', 'vareta_22'


    '''
    ##part1 : creates a dataframe df1 with n_11 and n_13 as columns to show number of these particles in every event
    # contains the file as list of list of floats
    data = []

    # leading and trailing spaces and commas are removed,also string is converted to floats
    with open(input_file) as file:
        for line in file:
            event = []
            line = line[:-1].replace(',\n', '\n')
            line = line.rstrip()
            line = line.rstrip(',')
            # print(line)

            for numstr in line.split(","):
                if numstr:
                    try:
                        numFl = float(numstr)
                        event.append(numFl)


                    except ValueError as e:
                        print(e)
            data.append(event)

    # list of dictionieries of size 100000.Each dictionery store n_11 and n_13 for each event


    ##part 2: find out common particles in each event i.e. 22 and 211 and create a dataframe df2 with columns
    ##'ID_211', 'n_211', 'meanPt_211','meaneta_211', 'varPt_211', 'vareta_211',
    ##        'ID_22', 'n_22', 'meanPt_22', 'meaneta_22','varPt_22', 'vareta_22'

    chunks = {}
    first_elements = []
    count = 0
    event_ids = []

    for item in data:
        # item.pop()
        chunks[count] = [list(np.float_(item[x:x + 6])) for x in range(0, len(item), 6)]
        event_ids.append(chunks[count][0])
        count += 1
    # list of dictionaries with particle ids as keys
    d = []
    i = range(len(chunks))
    for item in i:
        l = {}
        for lists in chunks[item]:
            l[lists[0]] = lists[1:]
        d.append(l)
    # particles id in each event
    particles_event = []
    for item in d:
        particles_event.append(list(item.keys()))
    # common particles in each event
    elements_in_all = list(set.intersection(*map(set, particles_event)))
    # list of dictionaries with two keys i.e. 211 and 22 for each event
    d2 = []
    for item in d:
        item = dict((k, item[k]) for k in elements_in_all if k in item)
        d2.append(item)
    # list of list with each sublist is the [211,total number ,px,py,pz,22.otal number ,px,py,pz]
    dictlist = []
    for item in d2:
        temp = []

        for key, value in item.items():
            temp = value
            temp.insert(0, key)
            dictlist.append(temp)
    x = iter(dictlist)
    dictlist = [a + b for a, b in zip_longest(x, x, fillvalue=[])]
    columns = ['ID_211', 'n_211', 'meanPx_211', 'meanPy_211', 'meanPz_211', 'meanE_211',
               'ID_22', 'n_22', 'meanPx_22', 'meanPy_22', 'meanPz_22', 'meanE_22']
    df= pd.DataFrame(dictlist, columns=columns)
    df.drop(['ID_211', 'ID_22'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)



    return df
def to_parquet_m2(signal_df, background_df, name):
    '''


    :param list1: df of signal i.e. status = 1 assigned here before merging with background df
    :param list2: df of background i.e. status = 0 assigned here before merging with signal df
    :param name : name of parquetfile to be saved
    :return: parquet file containing background and signal data with common IDs(211,222),n_11,n_13,'meanPt_211', 'meaneta_211',
    'varPt_211','vareta_211', 'n_22', 'meanPt_22', 'meaneta_22', 'varPt_22','vareta_22' consolidated per event
    '''
    columns = ['n_211', 'meanPt_211', 'meaneta_211'
               'n_22', 'meanPt_22', 'meaneta_22'
               ]
    # signal df
    signal_df['Status'] = 1
    # background df
    background_df['Status'] = 0
    datasets = [signal_df, background_df]
    df = pd.concat(datasets)
    df.reset_index(drop=True, inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_parquet(name + '.parquet', engine='pyarrow')
    return df


# #new features Pt and eta containing files of one dataset combination
# s7 = clean_text('signal7abseta.txt','s7.txt')
# b7 = clean_text('background7abseta.txt','b7.txt')
# s7dic = rivet_to_df(s7)
# b7dic =rivet_to_df(b7)
#
# df1 = to_parquet_m2(s7dic,b7dic,'abseta7')
#
# #new features Pt and eta containing files of second dataset combination
# s69 = clean_text('signal69abseta.txt','s69.txt')
# b69 = clean_text('background69abseta.txt','b69.txt')
# s69dic = rivet_to_df(s69)
# b69dic = rivet_to_df(b69)
#
# df2 = to_parquet_m2(s69dic,b69dic,'abseta69')
#
#
#
# s7 = clean_text('signal7.txt','s7.txt')
# b7 = clean_text('background7.txt','b7.txt')
# b7 = only_commonids_modified(b7)
# s7 = only_commonids_modified(s7)
# df = to_parquet_modified(s7,b7,'abseta7')
#

#var datasets
# s1 = clean_text('var1/signal1var.txt','var1/signal1varcleaned.txt')
# b2 = clean_text('var1/background2var.txt','var1/background2varcleaned.txt')
# b2 = only_commonids_modified(b2)
# s1 = only_commonids_modified(s1)
# df = to_parquet_modified_var(s1,b2,'var1/var1_2')


# s7 = clean_text('var1/7/signal7var.txt','var1/7/signal7varcleaned.txt')
# s7 = only_commonids_modified(s7)
# b7 = clean_text('var1/7/background7var.txt','var1/7/background7varcleaned.txt')
# b7 = only_commonids_modified(b7)
# df = to_parquet_modified_var(s7,b7,'var1/7/var7')

# s69 = clean_text('var1/69/signal69var.txt','var1/69/signal69varcleaned.txt')
# s69 = only_commonids_modified(s69)
# b69 = clean_text('var1/69/background69var.txt','var1/69/background69varcleaned.txt')
# b69 = only_commonids_modified(b69)
# df = to_parquet_modified_var(s69,b69,'var1/69/var69')

#events



# s7 = clean_text('event/7/eventsignal7.txt','event/7/eventsignal7cleaned.txt')
# dfs7 = rivet_to_df_events(s7)
#
# b7 = clean_text('event/7/eventbackground7.txt','event/7/eventbackround7cleaned.txt')
# dfb7 = rivet_to_df_events(b7)
# df = to_parquet_events(signal_df=dfs7,background_df=dfb7,name='event/7/events7')

# s69 = clean_text('event/69/eventsignal69.txt','event/69/eventsignal69cleaned.txt')
# dfs69= rivet_to_df_events(s69)
#
# b69 = clean_text('event/69/eventbackground69.txt','event/69/eventbackround69cleaned.txt')
# dfb69 = rivet_to_df_events(b69)
# df = to_parquet_events(signal_df=dfs69,background_df=dfb69,name='event/69/events69')

# s1 = clean_text('event/1_2/eventsignal1.txt','event/1_2/eventsignal1cleaned.txt')
# dfs1= rivet_to_df_events(s1)
#
# b2 = clean_text('event/1_2/eventbackground2.txt','event/1_2/eventbackround2cleaned.txt')
# dfb2 = rivet_to_df_events(b2)
# df = to_parquet_events(signal_df=dfs1,background_df=dfb2,name='event/1_2/events1_2')
#

#abs_eta
s1 = clean_text('abseta_trans/1_2/absetasignal1.txt','s1.txt')
df = rivet_to_df_events(s1)
b2 = clean_text('abseta_trans/1_2/absetabackground2.txt','b2.txt')
df = rivet_to_df(b2)
s1 = only_commonids_modified(s1)
b2 = only_commonids_modified(b2)
#
# #
# df1 = to_parquet_modified(s1,b2,'abseta_trans/1_2/abseta1_2')

