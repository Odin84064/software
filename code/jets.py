import pandas as pd
import re
import numpy as np
import os
from itertools import zip_longest
import math
import time
from pathlib import Path
import csv

pd.set_option('display.max_columns', None)

print("Current working directory: {0}".format(os.getcwd()))
os.chdir("../dataset/jets/")
print("Current working directory: {0}".format(os.getcwd()))


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


def rivet_to_df_1(input_file):
    '''

    :param input_file: cleaned output file from function clean_text without any unnecessary text
    :return: dataframe with 100000 rows as in total events and n_11 n_12,n_14, n_13,n_16 as columns to show number of these particles in every event
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
    file.close()
    # list of dictionieries of size 100000.Each dictionery store n_11 and n_13 for each event
    l = []
    for item in data:
        d = {'n_11': 0, 'n_12': 0, 'n_13': 0, 'n_14': 0, 'n_16': 0}
        for x in item[::6]:

            if int(x) == 11:
                d['n_11'] = math.ceil(item[item.index(x) + 1])
            if int(x) == 12:
                d['n_12'] = math.ceil(item[item.index(x) + 1])
            if int(x) == 13:
                d['n_13'] = math.ceil(item[item.index(x) + 1])
            if int(x) == 14:
                d['n_14'] = math.ceil(item[item.index(x) + 1])
            if int(x) == 16:
                d['n_16'] = math.ceil(item[item.index(x) + 1])

        l.append(d)
    # list of dictioneries is converted to df
    df1 = pd.DataFrame(l, columns=['n_11', 'n_12', 'n_13', 'n_14', 'n_16'])

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
    columns = ['ID_211', 'n_211', 'meanPt_211', 'meaneta_211', 'varPt_211', 'vareta_211',
               'ID_22', 'n_22', 'meanPt_22', 'meaneta_22', 'varPt_22', 'vareta_22']
    df2 = pd.DataFrame(dictlist, columns=columns)
    df2.drop(['ID_211', 'ID_22'], axis=1, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    ##merging df1 and df2 together to get df
    datasets = [df1, df2]
    df = pd.concat(datasets, axis=1)
    df.reset_index(drop=True, inplace=True)

    return df


def jet_msb_var(msbfile, varfile, jetfile):
    # a = clean_text('msbsignal7.txt','msbsignal7cleaned.txt')
    file1 = msbfile + '.txt'
    file2 = msbfile + 'cleaned.txt'

    a = clean_text(file1, file2)

    dfa = pd.read_csv(a, names=['particle_multiplicity', 'transverse_momenta_sum', 'beam_thrust'])
    print(dfa.head())
    # os.chdir("../")
    file3 = varfile + '.txt'
    file4 = varfile + 'cleaned.txt'
    # file3 = '../' + paramfile + '.txt'
    # file4 = '../' + paramfile + 'cleaned.txt'
    b = clean_text(file3, file4)
    dfb = rivet_to_df_1(b)
    print(dfb.head())
    file5 = jetfile + '.txt'
    file6 = jetfile + 'cleaned.txt'
    c = clean_text(file5, file6)
    dfc = pd.read_csv(c, names=['jet_size5GeV', 'jet_size10GeV', 'jet_size15GeV', 'jet_size20GeV'], skiprows=1)
    print(dfc.head())
    datasets = [dfb, dfa, dfc]
    df = pd.concat(datasets, axis=1)
    print(df.head())
    return df
def jet50_msb_var(msbfile, varfile, jetfile):
    # a = clean_text('msbsignal7.txt','msbsignal7cleaned.txt')
    file1 = msbfile + '.txt'
    file2 = msbfile + 'cleaned.txt'

    a = clean_text(file1, file2)

    dfa = pd.read_csv(a, names=['particle_multiplicity', 'transverse_momenta_sum', 'beam_thrust'])
    print(dfa.head())
    # os.chdir("../")
    file3 = varfile + '.txt'
    file4 = varfile + 'cleaned.txt'
    # file3 = '../' + paramfile + '.txt'
    # file4 = '../' + paramfile + 'cleaned.txt'
    b = clean_text(file3, file4)
    dfb = rivet_to_df_1(b)
    print(dfb.head())
    file5 = jetfile + '.txt'
    file6 = jetfile + 'cleaned.txt'
    c = clean_text(file5, file6)
    dfc = pd.read_csv(c, names=['jet_size5GeV', 'jet_size10GeV', 'jet_size15GeV', 'jet_size20GeV','jet_size50GeV'], skiprows=1)
    print(dfc.head())
    datasets = [dfb, dfa, dfc]
    df = pd.concat(datasets, axis=1)
    print(df.head())
    return df

def jet_pt_msb_var(msbfile, varfile, jetfile):
    # a = clean_text('msbsignal7.txt','msbsignal7cleaned.txt')
    file1 = msbfile + '.txt'
    file2 = msbfile + 'cleaned.txt'

    a = clean_text(file1, file2)

    dfa = pd.read_csv(a, names=['particle_multiplicity', 'transverse_momenta_sum', 'beam_thrust'])
    print(dfa.head())
    # os.chdir("../")
    file3 = varfile + '.txt'
    file4 = varfile + 'cleaned.txt'
    # file3 = '../' + paramfile + '.txt'
    # file4 = '../' + paramfile + 'cleaned.txt'
    b = clean_text(file3, file4)
    dfb = rivet_to_df_1(b)
    print(dfb.head())
    file5 = jetfile + '.txt'
    file6 = jetfile + 'cleaned.txt'
    c = clean_text(file5, file6)
    dfc = pd.read_csv(c, names=['jet_size5GeV','sumpt_5GeV', 'jet_size10GeV','sumpt_10GeV', 'jet_size15GeV','sumpt_15GeV', 'jet_size20GeV','sumpt_20GeV','jet_size50GeV','sumpt_50GeV'], skiprows=1)
    print(dfc.head())
    datasets = [dfb, dfa, dfc]
    df = pd.concat(datasets, axis=1)
    print(df.head())
    return df

def to_parquet_1(signal_df, background_df, name):
    '''


    :param list1: df of signal i.e. status = 1 assigned here before merging with background df
    :param list2: df of background i.e. status = 0 assigned here before merging with signal df
    :param name : name of parquetfile to be saved
    :return: parquet file containing background and signal data with common IDs(211,222),n_11,n_13,'meanPt_211', 'meaneta_211',
    'varPt_211','vareta_211', 'n_22', 'meanPt_22', 'meaneta_22', 'varPt_22','vareta_22' consolidated per event
    '''
    '''columns = ['n_11', 'n_12', 'n_13', 'n_14', 'n_16', 'n_211', 'meanPt_211', 'meaneta_211', 'varPt_211',
               'vareta_211', 'n_22', 'meanPt_22', 'meaneta_22', 'varPt_22',
               'vareta_22']'''
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

#jets
dfs1 = jet_msb_var('69/msbsignal69', '69/signal69var','69/jetsignal69')
dfb2 = jet_msb_var('69/msbbackground69', '69/background69var','69/jetbackground69')
df = to_parquet_1(dfs1,dfb2,'69/jetmsbvar69')



#dfs1 = jet_msb_var('1_2/msbsignal1edit', '1_2/signal1var','1_2/jetsignal1')
#dfb2 = jet_msb_var('1_2/msbbackground2edit', '1_2/background2var','1_2/jetbackground2')
#df = to_parquet_1(dfs1,dfb2,'1_2/jetmsbvar1_2edit')


#jets50
#dfs1 = jet50_msb_var('69/msbsignal69', '69/signal69var','69/jets50signal69')
#dfb2 = jet50_msb_var('69/msbbackground69', '69/background69var','69/jets50background69')
#df = to_parquet_1(dfs1,dfb2,'69/jets50msbvar69')


#jets_pt
#dfs1 = jet_pt_msb_var('1_2/msbsignal1', '1_2/signal1var','1_2/jetsptsignal1')
#dfb2 = jet_pt_msb_var('1_2/msbbackground2', '1_2/background2var','1_2/jetsptbackground2')
#df = to_parquet_1(dfs1,dfb2,'1_2/jetsptmsbvar1_2')
