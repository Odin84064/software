import pandas as pd
import re
import numpy as np

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ks_2samp
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib import pyplot
from timeit import default_timer as timer
from scipy.spatial import distance
print("Current working directory: {0}".format(os.getcwd()))
os.chdir("../dataset")
print("Current working directory: {0}".format(os.getcwd()))
#first parquet file of mcvalid.text and mctev.txt
#df = pd.read_parquet('100000eventsconsolidated.parquet', engine="pyarrow")
pd.set_option('display.max_columns', None)




def get_minkowski(path):
    df = pd.read_parquet(path, engine="pyarrow")
    #print(df.head(2))
    train, validate, test = np.split(df.sample(frac=1, random_state=42),
                           [int(.6*len(df)), int(.8*len(df))])
    X_train = train.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]

    X_valid = validate.iloc[:, 0:-1]
    y_valid = validate.iloc[:,-1]

    X_test = test.iloc[:,0:-1]
    y_test = test.iloc[:,-1]
    no_of_columns = len(X_train.columns)
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_cols)], remainder='passthrough')
    X_train_scaled = ct.fit_transform(X_train)
    X_valid_scaled = ct.transform(X_valid)
    X_test_scaled =  ct.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled,columns = X_train.columns)
    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns = X_valid.columns)
    X_test_scaled =  pd.DataFrame(X_test_scaled, columns = X_test.columns)



    def plot_output_distribution(model):
        predict = model.predict(X_test_scaled)
        predict_df = pd.DataFrame(predict, columns=['Predict'])
        y = pd.DataFrame(y_test)
        y.reset_index(drop=True, inplace=True)
        final = pd.concat([y, predict_df], axis=1)
        one = final[final['Status'] == 1]
        zero = final[final['Status'] == 0]
       # one = one.sample(n=19900)
        #zero = zero.sample(n=19900)
        data1 = one['Predict']
        data0 = zero['Predict']


        xb=plt.hist(data0, bins=100000)[1]


        xs=plt.hist(data1, bins=100000)[1]
        ms = distance.minkowski(xb, xs)

        return ms

    def basic_model(activation, nodes, learning_rate, X_train, y_train, X_valid, Y_valid, batch_size, epochs):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = keras.Sequential([
            layers.Dense(nodes, activation='relu', input_shape=[nodes]),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(

            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )
        early_stopping = keras.callbacks.EarlyStopping(
            patience=100,
            min_delta=0.001,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1,
        )

        history_df = pd.DataFrame(history.history)

        return (model, history_df)

    tan_fun = basic_model('tanh',no_of_columns,0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500)
    minkowski=plot_output_distribution(tan_fun[0])
    return minkowski

def get_ks(path):
    df = pd.read_parquet(path, engine="pyarrow")
    #print(df.head(2))
    train, validate, test = np.split(df.sample(frac=1, random_state=42),
                           [int(.6*len(df)), int(.8*len(df))])
    X_train = train.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]

    X_valid = validate.iloc[:, 0:-1]
    y_valid = validate.iloc[:,-1]

    X_test = test.iloc[:,0:-1]
    y_test = test.iloc[:,-1]
    no_of_columns = len(X_train.columns)
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_cols)], remainder='passthrough')
    X_train_scaled = ct.fit_transform(X_train)
    X_valid_scaled = ct.transform(X_valid)
    X_test_scaled =  ct.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled,columns = X_train.columns)
    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns = X_valid.columns)
    X_test_scaled =  pd.DataFrame(X_test_scaled, columns = X_test.columns)



    def plot_output_distribution(model):
        predict = model.predict(X_test_scaled)
        predict_df = pd.DataFrame(predict, columns=['Predict'])
        y = pd.DataFrame(y_test)
        y.reset_index(drop=True, inplace=True)
        final = pd.concat([y, predict_df], axis=1)
        one = final[final['Status'] == 1]
        zero = final[final['Status'] == 0]

        data1 = one['Predict']
        data0 = zero['Predict']


        xb=plt.hist(data0,bins=100000)[0]


        xs=plt.hist(data1,bins = 100000)[0]

        ks= ks_2samp(xs, xb)
        return ks

    def basic_model(activation, nodes, learning_rate, X_train, y_train, X_valid, Y_valid, batch_size, epochs):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = keras.Sequential([
            layers.Dense(nodes, activation='relu', input_shape=[nodes]),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(

            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )
        early_stopping = keras.callbacks.EarlyStopping(
            patience=100,
            min_delta=0.001,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1,
        )

        history_df = pd.DataFrame(history.history)

        return (model, history_df)

    tan_fun = basic_model('tanh',no_of_columns,0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500)
    ks=plot_output_distribution(tan_fun[0])
    return ks

def get_ms_ks(path):
    df = pd.read_parquet(path, engine="pyarrow")
    #print(df.head(2))
    train, validate, test = np.split(df.sample(frac=1, random_state=42),
                           [int(.6*len(df)), int(.8*len(df))])
    X_train = train.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]

    X_valid = validate.iloc[:, 0:-1]
    y_valid = validate.iloc[:,-1]

    X_test = test.iloc[:,0:-1]
    y_test = test.iloc[:,-1]
    no_of_columns = len(X_train.columns)
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_cols)], remainder='passthrough')
    X_train_scaled = ct.fit_transform(X_train)
    X_valid_scaled = ct.transform(X_valid)
    X_test_scaled =  ct.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled,columns = X_train.columns)
    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns = X_valid.columns)
    X_test_scaled =  pd.DataFrame(X_test_scaled, columns = X_test.columns)



    def plot_output_distribution(model):
        predict = model.predict(X_test_scaled)
        predict_df = pd.DataFrame(predict, columns=['Predict'])
        y = pd.DataFrame(y_test)
        y.reset_index(drop=True, inplace=True)
        final = pd.concat([y, predict_df], axis=1)
        one = final[final['Status'] == 1]
        zero = final[final['Status'] == 0]
       # one = one.sample(n=19900)
        #zero = zero.sample(n=19900)
        data1 = one['Predict']
        data0 = zero['Predict']


        xb=plt.hist(data0, bins=30)[1]


        xs=plt.hist(data1, bins=30)[1]
        ms = distance.minkowski(xb, xs)
        ks = ks_2samp(xb, xs)
        #ms = distance.minkowski(data1[:19000], data0[:19000])
        #ks = ks_2samp(data1[:19000], data0[:19000])

        return ms,ks

    def basic_model(activation, nodes, learning_rate, X_train, y_train, X_valid, Y_valid, batch_size, epochs):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = keras.Sequential([
            layers.Dense(nodes, activation='relu', input_shape=[nodes]),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(

            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )
        early_stopping = keras.callbacks.EarlyStopping(
            patience=100,
            min_delta=0.001,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1,
        )

        history_df = pd.DataFrame(history.history)

        return (model, history_df)

    tan_fun = basic_model('tanh',no_of_columns,0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500)
    ms,ks=plot_output_distribution(tan_fun[0])
    return ms,ks




#bugged dataset
paths1_2 =[
r'abseta_trans/1_2/abseta1_2.parquet',
        r'event/1_2/events1_2.parquet',

        r'var1/1_2/var1_2.parquet',
           r'n11n13/1_2/var1_2n11n13.parquet',
           r'n12/1_2/var1_2n12.parquet',
           r'msb/1_2/msbvar1_2.parquet',
           r'jets/1_2/jetmsbvar1_2.parquet',
           r'jets50/1_2/jets50msbvar1_2.parquet',
           r'jets_pt/1_2/jetsptmsbvar1_2.parquet'


           ]









#different
paths7 =[
r'abseta_trans/7/abseta7.parquet',
    r'event/7/events7.parquet',

    r'var1/7/var7.parquet',
           r'n11n13/7/var7n11n13.parquet',
           r'n12/7/var7n12.parquet',
           r'msb/7/msbvar7.parquet',
           r'jets/7/jetmsbvar7.parquet',
           r'jets50/7/jets50msbvar7.parquet',
           r'jets_pt/7/jetsptmsbvar7.parquet'


           ]






paths69 =[
    r'abseta_trans/69/abseta69.parquet',
    r'event/69/events69.parquet',

    r'var1/69/var69.parquet',
           r'n11n13/69/var69n11n13.parquet',
           r'n12/69/var69n12.parquet',
           r'msb/69/msbvar69.parquet',
           r'jets/69/jetmsbvar69.parquet',
           r'jets50/69/jets50msbvar69.parquet',
           r'jets_pt/69/jetsptmsbvar69.parquet'


           ]






def minkowski_excel(paths1_2,paths7,paths69):
    minkowski_list1_2 = ['N/A', 'N/A']
    for item in paths1_2[2:]:
        minkowski_list1_2.append(get_minkowski(item))
    df1_2 = pd.DataFrame(list(zip(paths1_2, minkowski_list1_2)), columns=['Bugged', 'Minkowski'])
    minkowski_list7 = []
    for item in paths7:
        minkowski_list7.append(get_minkowski(item))

    df7 = pd.DataFrame(list(zip(paths7, minkowski_list7)), columns=['Different', 'Minkowski'])

    minkowski_list69 = []
    for item in paths69:
        minkowski_list69.append(get_minkowski(item))
    df69 = pd.DataFrame(list(zip(paths69, minkowski_list69)), columns=['Same', 'Minkowski'])

    df = pd.concat([df1_2, df7, df69], axis=1)
    df.to_excel('../minkowski/minkowski_scores_100000bins.xlsx')

def ks_excel(paths1_2,paths7,paths69):
    ks_list1_2 = ['N/A','N/A']
    for item in paths1_2[2:]:
        ks_list1_2.append(get_ks(item))
    df1_2 = pd.DataFrame(list(zip(paths1_2,ks_list1_2)), columns=['Bugged','ks'])
    ks_list7 = []
    for item in paths7:
        ks_list7.append(get_ks(item))

    df7 = pd.DataFrame(list(zip(paths7,ks_list7)), columns=['Different','ks'])

    ks_list69 = []
    for item in paths69:
        ks_list69.append(get_ks(item))
    df69 = pd.DataFrame(list(zip(paths69,ks_list69)), columns=['Same','ks'])
    df = pd.concat(df7, df69, df1_2)
    df.to_excel('../ks/ks_scores_100000bins.xlsx')

def ms_ks_excel(paths1_2,paths7,paths69):

    """Bugged Dataset"""
    ms_list1_2 = ['N/A', 'N/A']
    ks_list1_2 = ['N/A', 'N/A']
    for item1 in paths1_2[2:]:
        ms,ks =get_ms_ks(item1)
        ms_list1_2.append(ms)
        ks_list1_2.append(ks)
    df1_2ms = pd.DataFrame(list(zip(paths1_2, ms_list1_2)), columns=['Bugged', 'ms'])
    df1_2ks = pd.DataFrame(list(zip(paths1_2, ks_list1_2)), columns=['Bugged', 'ks'])

    """Different Dataset"""
    ms_list7 = []
    ks_list7 = []
    for item2 in paths7:
        ms, ks = get_ms_ks(item2)
        ms_list7.append(ms)
        ks_list7.append(ks)
    df7ms = pd.DataFrame(list(zip(paths7, ms_list7)), columns=['Different', 'ms'])
    df7ks = pd.DataFrame(list(zip(paths7, ks_list7)), columns=['Different', 'ks'])

    """Same Dataset"""
    ms_list69 = []
    ks_list69 = []
    for item3 in paths69:
        ms, ks = get_ms_ks(item3)
        ms_list69.append(ms)
        ks_list69.append(ks)
    df69ms = pd.DataFrame(list(zip(paths7, ms_list69)), columns=['Different', 'ms'])
    df69ks = pd.DataFrame(list(zip(paths69, ks_list69)), columns=['Same', 'ks'])


    dfms = pd.concat(df7ms, df69ms, df1_2ms)
    dfms.to_excel('../ms/ms_scores_100000bins.xlsx')
    dfks = pd.concat(df7ks, df69ks, df1_2ks)
    dfks.to_excel('../ks/ks_scores_100000bins.xlsx')
    return dfks,dfms

def ms_ks_dataset_df(paths,column_name):

    """Bugged Dataset"""
    ms_list = []
    ks_list = []
    if column_name == 'Bugged':

        ms_list = ['N/A']
        ks_list = ['N/A']
        for item in paths[1:]:
            ms,ks =get_ms_ks(item)
            ms_list.append(ms)
            ks_list.append(ks)
    else:
        for item in paths:
            ms,ks =get_ms_ks(item)
            ms_list.append(ms)
            ks_list.append(ks)


    dfms = pd.DataFrame(list(zip(paths, ms_list)), columns=[column_name, 'ms'])
    dfks = pd.DataFrame(list(zip(paths, ks_list)), columns=[column_name, 'ks'])

    return dfks,dfms


dfks7,dfms7 = ms_ks_dataset_df(paths7,"Different")

dfks69,dfms69 = ms_ks_dataset_df(paths69,"Same")

dfks1_2,dfms1_2 = ms_ks_dataset_df(paths1_2,"Bugged")
#
dfms = pd.concat([dfms7, dfms69, dfms1_2 ],axis=1)
dfms.to_excel('../ms/ms_scores_bins.xlsx')
# dfks = pd.concat([dfks7, dfks69, dfks1_2],axis=1)
# dfks.to_excel('../ks/ks_scores_100000bins.xlsx')

