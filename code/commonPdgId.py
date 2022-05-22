'''
This script finds out stable particles per event and unique stable particles
in a data set

'''


import pandas as pd
import re
import numpy as np
import os
from itertools import zip_longest
import time
from pathlib import Path
import csv


os.chdir("../dataset/")




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
def stable_particles(input_file, name, name2):
    '''

    :param input_file: cleaned output file from function clean_text without any unnecessary text
    :return: csv file containing stable particles per event as stableparticles
             csv file containing unique stable particles as uniquestable particles


    '''

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

    l = []
    for item in data:
        d = []
        for x in item[::6]:

            if int(x) < 30:
                d.append(int(x))

        l.append(d)
    import csv
    from itertools import chain

    with open(name + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(l)
    f.close()
    elements_in_all = list(set(chain(*l)))
    with open(name2 + ".csv", "w", newline="") as f1:

        f1.write("\n".join(str(item) for item in elements_in_all))
    f1.close()

    return l, elements_in_all

b7 = clean_text('background7var.txt','b7.txt')
l,elements_in_all= stable_particles(b7,'stableparticles','uniquestableparticles')