'''
ran experiments on number of layers for the bugged dataset  msbvar1_2.parquet  as shown in kw-24 presentation
'''


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
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt

print("Current working directory: {0}".format(os.getcwd()))
os.chdir("../dataset")
print("Current working directory: {0}".format(os.getcwd()))
#first parquet file of mcvalid.text and mctev.txt
#df = pd.read_parquet('100000eventsconsolidated.parquet', engine="pyarrow")
pd.set_option('display.max_columns', None)
#parquetfile of  444102.PhPy8EG_A14_ttbar_hdamp258p75_fullrun_nonallhad.21.6.32 and 444101.PhPy8EG_A14_ttbar_hdamp258p75_fullrun_nonallhad.21.6.17 stored as signalconsolidated.txt and backgroundconsolidated.txt
df = pd.read_parquet('msb/1_2/msbvar1_2.parquet', engine="pyarrow")
print(df.head(5))
print(df.shape)

# Create training, validation and test splits

train, validate, test = np.split(df.sample(frac=1, random_state=42),
                       [int(.6*len(df)), int(.8*len(df))])
X_train = train.iloc[:, 0:-1]
y_train = train.iloc[:, -1]

X_valid = validate.iloc[:, 0:-1]
y_valid = validate.iloc[:,-1]

X_test = test.iloc[:,0:-1]
y_test = test.iloc[:,-1]

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_cols)], remainder='passthrough')
X_train_scaled = ct.fit_transform(X_train)
X_valid_scaled = ct.transform(X_valid)
X_test_scaled =  ct.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled,columns = X_train.columns)
X_valid_scaled = pd.DataFrame(X_valid_scaled, columns = X_valid.columns)
X_test_scaled =  pd.DataFrame(X_test_scaled, columns = X_test.columns)


def dense_model(learning_rate, X_train, y_train, X_valid, Y_valid, batch_size, epochs,neurons):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.Sequential([
        layers.Dense(neurons, activation='relu', input_shape=[18]),


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
    predict = model.predict(X_test_scaled)
    predict_df = pd.DataFrame(predict, columns=['Predict'])
    y = pd.DataFrame(y_test)
    y.reset_index(drop=True, inplace=True)
    final = pd.concat([y, predict_df], axis=1)
    final.loc[:, 'Predict'][final['Predict'] <= 0.5] = 0
    final.loc[:, 'Predict'][final['Predict'] > 0.5] = 1
    y_true = final['Status'].to_numpy()
    y_predict = final['Predict'].to_numpy()

    accuracy = accuracy_score(final['Status'], final['Predict'])
    f1 = f1_score(final['Status'].to_numpy(), final['Predict'].to_numpy())

    true_pos = final[(final.Status == 1) & (final.Predict == 1)]
    false_pos = final[(final.Status == 0) & (final.Predict == 1)]
    true_neg = final[(final.Status == 0) & (final.Predict == 0)]
    false_neg = final[(final.Status == 1) & (final.Predict == 0)]

    tp = (len(true_pos) / len(final)) * 100
    fp = (len(false_pos) / len(final)) * 100
    tn = (len(true_neg) / len(final)) * 100
    fn = (len(false_neg) / len(final)) * 100

    tp1 = len(true_pos)
    fp1 = len(false_pos)
    fn1 = len(false_neg)
    tn1 = len(true_neg)
    sensitivity = (tp1 / (tp1 + fn1))
    specificity = (tn1 / (tn1 + fp1))

    # writing performance parameters to a file

    return f1, accuracy, sensitivity, specificity




def plot_output_distribution(model):
    predict = model.predict(X_train_scaled)
    predict_df = pd.DataFrame(predict, columns=['Predict'])
    y = pd.DataFrame(y_train)
    y.reset_index(drop=True, inplace=True)
    final = pd.concat([y, predict_df], axis=1)
    one = final[final['Status'] == 1]
    zero = final[final['Status'] == 0]

    data1 = one['Predict']
    data0 = zero['Predict']
    plt.figure(figsize=(15, 8))
    sns.histplot(data0, label='Background', kde= 'True',color = 'b')
    sns.histplot(data1,label='Signal', kde= 'True',alpha=0.7, color='r')
    plt.xlabel('Probabilitiy Distribution ', fontsize=25)
    plt.ylabel('Number of Datapoints', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    pat1 = '../code/plots/eventsconsolidated/' + 'probability_distribution' + '.png'
    plt.savefig(pat1)
    plt.show()
    plt.close()

    plt.figure()
    plt.figure(figsize=(15, 8))
    sns.histplot(data0, bins=50, label='Background',kde = True,color='b')
    plt.xlabel('Probabilitiy Distribution ', fontsize=25)
    plt.ylabel('Frequency of Datapoints', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    pat2 = '../code/plots/eventsconsolidated/' + 'probability_distribution_background' + '.png'
    plt.savefig(pat2)
    plt.show()
    plt.close()

    plt.figure()
    plt.figure(figsize=(15, 8))
    sns.histplot(data1, bins=50, label='Signal',alpha = 0.7,color = 'r',kde = True)
    plt.xlabel('Probabilitiy Distribution ', fontsize=25)
    plt.ylabel('Frequency of Datapoints', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    pat2 = '../code/plots/eventsconsolidated/' + 'probability_distribution_signal' + '.png'
    plt.savefig(pat2)
    plt.show()
    plt.close()

def run_multiple_model():
    '''
    :param: parquet file i.e. msbvar7,msbvar69 and masbvar1_2.parquet
    :return: run model 50 times and returns accuracy.f1-score,sensitivity and specificity as excel file
    '''
    accuracy_list = []
    f1_list = []
    sensitivity_list = []
    specificity_list = []
    for i in range(10):
        f1, accuracy, sensitivity, specificity =dense_model( 0.001,X_train_scaled,y_train,X_valid_scaled,y_valid, 1024, 500,10)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    df=pd.DataFrame(list(zip(f1_list, accuracy_list, specificity_list,sensitivity_list)),
                     columns=['f1_score', 'accuracy', 'sensitivity','specificity'])
    df.to_excel('msb/1_2/var1_2_10times_dense_10.xlsx')
    return df



#tan_fun = dense_model(0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500)

#f1,accuracy,sensitivity,specificity = f1_accuracy_confusion(tan_fun[0])
#plot_output_distribution(tan_fun[0])
df = run_multiple_model()


#f1, accuracy, sensitivity, specificity = dense_model(0.001,X_train_scaled,y_train,X_valid_scaled,y_valid, 1024, 500,64)
#print((accuracy))