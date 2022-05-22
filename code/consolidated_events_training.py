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
from matplotlib import pyplot
from timeit import default_timer as timer

print("Current working directory: {0}".format(os.getcwd()))
os.chdir("../dataset/")
#first parquet file of mcvalid.text and mctev.txt
#df = pd.read_parquet('100000eventsconsolidated.parquet', engine="pyarrow")
pd.set_option('display.max_columns', None)
#parquetfile of  444102.PhPy8EG_A14_ttbar_hdamp258p75_fullrun_nonallhad.21.6.32 and 444101.PhPy8EG_A14_ttbar_hdamp258p75_fullrun_nonallhad.21.6.17 stored as signalconsolidated.txt and backgroundconsolidated.txt
df = pd.read_parquet('var69.parquet', engine="pyarrow")
print(df.head(5))


# Create training and validation splits
# features = df.iloc[:,0:-1]
# labels = df.iloc[:,-1]
# X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, random_state=42)
# numerical_cols = [cname for cname in features.columns if features[cname].dtype in ['int64', 'float64']]
# ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_cols)], remainder='passthrough')
# X_train_scaled = ct.fit_transform(X_train)
# X_valid_scaled = ct.transform(X_valid)
# X_train_scaled = pd.DataFrame(X_train_scaled,columns = X_train.columns)
# X_valid_scaled = pd.DataFrame(X_valid_scaled, columns = X_valid.columns)
# nt = ColumnTransformer([("only numeric", Normalizer(), numerical_cols)], remainder='passthrough')
# X_train_norm = nt.fit_transform(X_train)
# X_valid_norm = nt.transform(X_valid)
# X_train_norm = pd.DataFrame(X_train_norm,columns = X_train.columns)
# X_valid_norm = pd.DataFrame(X_valid_norm, columns = X_valid.columns)



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


def basic_perceptron(activation, learning_rate, X_train, y_train, X_valid, y_valid, batch_size, epochs):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.Sequential([
        layers.Dense(1, activation='sigmoid', input_shape=[15]),

    ])

    model.compile(
        optimizer=opt,
        # optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],

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
    # Start the plot at epoch 5
    history_df.loc[1:, ['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title(
        'Loss of perceptron with {} epochs,{} batch '.format(
            epochs,learning_rate, batch_size))

    plt.show()

    plt.close()
    history_df.loc[1:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title(
        'Accurcay of perceptron with {} epochs,{} batch'.format(
            epochs,
            learning_rate,
            batch_size))

    plt.show()

    plt.close()

    print((
              "Best Validation Loss of perceptorn with early stopping: {:0.4f} \nBest Validation Accuracy of perceptron with early stopping: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))
    return (model, history_df)

def basic_model(activation, learning_rate, X_train, y_train, X_valid, Y_valid, batch_size, epochs):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.Sequential([
        layers.Dense(15, activation='relu', input_shape=[15]),
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
    name1 = 'Binary Cross Entropy Loss with {} epochs,{} batch'.format(
        epochs, batch_size)
    name2 = 'Accurcay with {} epochs,{} batch'.format(epochs,activation,learning_rate,batch_size)
    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[1:, ['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title(
        'Binary Cross Entropy Loss with {} epochs,{} batch'.format(
            epochs, batch_size))
    pat1 = '../code/plots/eventsconsolidated/'+ name1 + '.png'
    plt.savefig(pat1)
    plt.show()

    plt.close()
    history_df.loc[1:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title('Accurcay with {} epochs,{} batch'.format(epochs,batch_size))
    pat2 = '../code/plots/eventsconsolidated/' + name2 + '.png'
    plt.savefig(pat2)
    plt.show()

    plt.close()

    print(("Best Validation Loss with early stopping: {:0.4f} \nBest Validation Accuracy with early stopping: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))
    return (model, history_df)

def plot_distributions(df):
    for column in df.columns:
        plt.figure()             # <==================== here!
        sns.distplot(df[column])
        plt.savefig('../code/plots/eventsconsolidated/distribution_' + column)
        plt.close()
def plot_scatter(df):
    y = df.iloc[:,-1]
    alpha = 0.5
    plt.scatter(df[y == 0]['meanE_211'], df[y == 0]['meanE_22'], label='Noise', s=50, lw=0, color=[1, alpha, alpha])
    plt.scatter(df[y == 1]['meanE_211'], df[y == 1]['meanE_22'], label='Signal', s=50, lw=0, color=[alpha, alpha, 1])
    plt.legend()
    plt.xlabel('meanE_211')
    plt.ylabel('meanE_22')
    plt.savefig('../code/plots/eventsconsolidated/scatter_meanE')
    plt.show()


    plt.scatter(df[y == 0]['meanPx_211'], df[y == 0]['meanPx_22'], label='Particle 211', s=50, lw=0,
                color=[1, alpha, alpha])
    plt.scatter(df[y == 1]['meanPx_211'], df[y == 1]['meanPx_22'], label='Particle 22', s=50, lw=0,
                color=[alpha, alpha, 1])
    plt.legend()
    plt.xlabel('meanPx_211')
    plt.ylabel('meanPx_22')
    plt.savefig('../code/plots/eventsconsolidated/scatter_meanPx')
    plt.show()

    plt.scatter(df[y == 0]['meanPy_211'], df[y == 0]['meanPy_22'], label='Noise', s=50, lw=0, color=[1, alpha, alpha])
    plt.scatter(df[y == 1]['meanPy_211'], df[y == 1]['meanPy_22'], label='Signal', s=50, lw=0, color=[alpha, alpha, 1])
    plt.legend()
    plt.xlabel('meanPy_211')
    plt.ylabel('meanPy_22')
    plt.savefig('../code/plots/eventsconsolidated/scatter_meanPy')
    plt.show()

    plt.scatter(df[y == 0]['meanPz_211'], df[y == 0]['meanPz_22'], label='Noise', s=50, lw=0, color=[1, alpha, alpha])
    plt.scatter(df[y == 1]['meanPz_211'], df[y == 1]['meanPz_22'], label='Signal', s=50, lw=0, color=[alpha, alpha, 1])
    plt.legend()
    plt.xlabel('meanPz_211')
    plt.ylabel('meanPz_22')
    plt.savefig('../code/plots/eventsconsolidated/scatter_meanPz')
    plt.show()


def plot_output_predictions(model):
    predict = model.predict(X_valid_scaled)

    predict_df = pd.DataFrame(predict, columns=['Predict'])
    y = pd.DataFrame(y_test)
    y.reset_index(drop=True, inplace=True)
    final = pd.concat([y, predict_df], axis=1)
    final.loc[:, 'Predict'][final['Predict'] <= 0.5] = 0
    final.loc[:, 'Predict'][final['Predict'] > 0.5] = 1
    one = final[final['Status'] == 1]
    zero = final[final['Status'] == 0]

    data1 = one['Predict']
    data0 = zero['Predict']
    # plt.hist(data1, weights=np.ones(len(data1)) / len(data1))
    #
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.ylabel('Frequency')
    # plt.xlabel('Prediction')
    # plt.title('Histogram of Signal')
    # plt.show()
    # pat1 = '../code/plots/eventsconsolidated/' + 'predictions_for_signal' + '.png'
    # plt.savefig(pat1)
    # plt.close()
    #
    # plt.hist(data0, weights=np.ones(len(data0)) / len(data0))
    #
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.ylabel('Frequency')
    # plt.xlabel('Prediction')
    # plt.title('Histogram of Noise')
    # plt.show()
    # pat2 = '../code/plots/eventsconsolidated/' + 'predictions_for_noise' + '.png'
    # plt.savefig(pat2)
    # plt.close()

    fig, ax = plt.subplots()
    ax.hist(data1, weights=np.ones(len(data1)) / len(data1), color='yellow', alpha=1, label='prediction')
    ax.hist(one['Status'], weights=np.ones(len(one['Status'])) / len(one['Status']), color='blue', alpha=0.7,
            label='actual')

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    ax.set(title='Actual vs Prediction (Signal)', ylabel='% of Label')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    plt.xlim([0, 1.1])
    plt.show()
    pat1 = '../code/plots/eventsconsolidated/' + 'predictions_for_signal'
    plt.savefig(pat1)
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(zero['Status'], weights=np.ones(len(zero['Status'])) / len(zero['Status']), color='red', alpha=0.7,
            label='actual')
    ax.hist(data0, weights=np.ones(len(data0)) / len(data0), color='lightblue', alpha=1, label='prediction')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    ax.legend()
    ax.set(title='Actual vs Prediction (Noise)', ylabel='% of Label')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    plt.xlim([0, 1])
    plt.show()
    pat2 = '../code/plots/eventsconsolidated/' + 'predictions_for_noise'
    plt.savefig(pat2)
    plt.close()

    return one, zero, final, predict


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
    plt.figure(figsize=(15, 7))
    sns.histplot(data0, label='Background', kde= 'True',color = 'b')
    sns.histplot(data1,label='Signal', kde= 'True',alpha=0.7, color='r')
    plt.xlabel('Probabilitiy Distribution ', fontsize=25)
    plt.ylabel('Number of Datapoints', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show()
    plt.close()

    plt.figure()
    plt.figure(figsize=(15, 7))
    sns.histplot(data0, bins=50, label='Background',kde = True,color='b')
    plt.xlabel('Probabilitiy Distribution ', fontsize=25)
    plt.ylabel('Frequency of Datapoints', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show()
    plt.close()

    plt.figure()
    plt.figure(figsize=(15, 7))
    sns.histplot(data1, bins=50, label='Signal',alpha = 0.7,color = 'r',kde = True)
    plt.xlabel('Probabilitiy Distribution ', fontsize=25)
    plt.ylabel('Frequency of Datapoints', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show()
    plt.close()

def f1_accuracy_confusion(model):
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

        cm = confusion_matrix(y_true, y_predict)

        true_pos = final[(final.Status == 1) & (final.Predict == 1)]
        false_pos = final[(final.Status == 0) & (final.Predict == 1)]
        true_neg = final[(final.Status == 0) & (final.Predict == 0)]
        false_neg = final[(final.Status == 1) & (final.Predict == 0)]

        tp = (len(true_pos) / len(final)) * 100
        fp = (len(false_pos) / len(final)) * 100
        tn = (len(true_neg) / len(final)) * 100
        fn = (len(false_neg) / len(final)) * 100

        ax = plt.subplot()
        sns.heatmap(cm / np.sum(cm), annot=True, ax=ax, fmt='.2%', cmap="Blues")
        ax.set_xlabel('Prediction');
        ax.set_ylabel('Actual');
        ax.set_title('Confusion Matrix blue');
        ax.xaxis.set_ticklabels(['Background', 'Signal']);
        ax.yaxis.set_ticklabels(['Background', 'Signal'])
        plt.show()
        pat3 = '../code/plots/eventsconsolidated/output_parameters/' + 'confusion_matrix'
        plt.savefig(pat3)
        plt.close()
        tp1 = len(true_pos)
        fp1 = len(false_pos)
        fn1 = len(false_neg)
        tn1 = len(true_neg)
        sensitivity = (tp1/(tp1+fn1))
        specificity = (tn1/(tn1+fp1))
        x = np.array([[tp1, fp1], [fn1, tn1]], np.float64)
        ax = plt.subplot()
        sns.heatmap(x / len(final), annot=True, ax=ax, fmt='.2%', cmap="Greens")
        ax.set_ylabel('Prediction');
        ax.set_xlabel('Actual');
        ax.set_title('Confusion Matrix');
        ax.xaxis.set_ticklabels(['Signal', 'Background']);
        ax.yaxis.set_ticklabels(['Signal', 'Background'])
        plt.show()
        pat4 = '../code/plots/eventsconsolidated/output_parameters/' + 'confusion_matrix_green'
        plt.savefig(pat4)
        return f1, accuracy,sensitivity,specificity

def roc_auc_curve(model):
    r_probs = [0 for _ in range(len(y_test))]
    predict = model.predict(X_test_scaled)
    r_auc = roc_auc_score(y_test, r_probs)
    rf_auc = roc_auc_score(y_test, predict)
    r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, predict)
    plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
    plt.plot(rf_fpr, rf_tpr, marker='.', label=' Model (AUROC = %0.3f)' % rf_auc)


    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() #
    # Show plot
    plt.show()
    plt.show()
    pat4 = '../code/plots/eventsconsolidated/output_parameters/' + 'roc'
    plt.savefig(pat4)
    plt.close()
def roc_auc_curve_two(model, perceptron):
        r_probs = [0 for _ in range(len(y_test))]
        predict = model.predict(X_test_scaled)
        predict_per = perceptron.predict(X_test_scaled)
        r_auc = roc_auc_score(y_test, r_probs)
        rf_auc = roc_auc_score(y_test, predict)
        rp_auc = roc_auc_score(y_test, predict_per)
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        rf_fpr, rf_tpr, _ = roc_curve(y_test, predict)
        rp_fpr, rp_tpr, _ = roc_curve(y_test, predict_per)
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(rf_fpr, rf_tpr, marker='.', label=' Model (AUROC = %0.3f)' % rf_auc)
        plt.plot(rp_fpr, rp_tpr, marker='p', label=' Perceptron (AUROC = %0.3f)' % rp_auc)

        # Title
        plt.title('ROC Plot')
        # Axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # Show legend
        plt.legend()  #
        # Show plot
        plt.show()
        plt.show()


#tan_fun = []
#sig_fun = []


# plot_distributions(X_train_scaled)
# plot_scatter(df)
#
# tan_fun.append(basic_model('tanh',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500))
# tan_fun.append(basic_model('tanh',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,512,500))
# tan_fun.append(basic_model('tanh',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,256,500))
# tan_fun.append(basic_model('tanh',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,128,500))
#
#
#
#
# sig_fun.append(basic_model('sigmoid',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500))
# sig_fun.append(basic_model('sigmoid',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,512,500))
# sig_fun.append(basic_model('sigmoid',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,256,500))
# sig_fun.append(basic_model('sigmoid',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,128,500))
#
#
# plot_output_predictions(tan_fun[0][0])



tan_fun = basic_model('tanh',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500)
#
#
# one,zero,final,predict = plot_output_predictions(tan_fun[0])
#
# plot_output_distribution(tan_fun[0])
#
# f1,accuracy,sensitivity,specificity = f1_accuracy_confusion(tan_fun[0])
#
# roc_auc_curve(tan_fun[0])

##perceptron
perceptron = basic_perceptron('tanh',0.001,X_train_scaled,y_train,X_valid_scaled,y_valid,1024,500)


f1,accuracy,sensitivity,specificity = f1_accuracy_confusion(tan_fun[0])

roc_auc_curve_two(tan_fun[0],perceptron[0])

plot_output_distribution(tan_fun[0])