
#printing model prediction also
#basic model with 5 input layer and no early stopping

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


import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from timeit import default_timer as timer

start = timer()



print("Current working directory: {0}".format(os.getcwd()))
os.chdir("../dataset/")

print("Current working directory: {0}".format(os.getcwd()))
df = pd.read_parquet('100000eventsconsolidated.parquet', engine="pyarrow")

# Create training and validation splits
features = df.iloc[:,0:-1]
labels = df.iloc[:,-1]
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, random_state=42)
numerical_cols = [cname for cname in features.columns if features[cname].dtype in ['int64', 'float64']]
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_cols)], remainder='passthrough')
X_train_scaled = ct.fit_transform(X_train)
X_valid_scaled = ct.transform(X_valid)
X_train_scaled = pd.DataFrame(X_train_scaled,columns = X_train.columns)
X_valid_scaled = pd.DataFrame(X_valid_scaled, columns = X_valid.columns)
nt = ColumnTransformer([("only numeric", Normalizer(), numerical_cols)], remainder='passthrough')
X_train_norm = nt.fit_transform(X_train)
X_valid_norm = nt.transform(X_valid)
X_train_norm = pd.DataFrame(X_train_norm,columns = X_train.columns)
X_valid_norm = pd.DataFrame(X_valid_norm, columns = X_valid.columns)
def basic_model():
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=[10]),
        layers.Dense(1, activation='tanh'),
    ])

    model.compile(
        optimizer = opt,
        #optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=100,
        min_delta=0.001,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_valid_scaled, y_valid),
        batch_size=128,
        epochs=500,
       callbacks=[early_stopping],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[1:, ['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss with 500 epochs ,early stopping,tanh activation of consolidated events')
    plt.show()
    plt.savefig('../code/plots/(events_cons)loss_with_200 Epochs.jpg')
    plt.close()
    history_df.loc[1:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title('Accurcay with 500 epochs ,early stopping,tanh activation  of consolidated events')
    plt.show()
    plt.savefig('../code/plots/(events_cons)accuracy_with_200 Epochs.jpg')
    plt.close()

    print(("Best Validation Loss with early stopping: {:0.4f} \nBest Validation Accuracy with early stopping: {:0.4f}")\
              .format(history_df['val_loss'].min(),
                      history_df['val_binary_accuracy'].max()))
    return model,history_df


def plot_output_predictions( model):
    predict = model[0].predict(X_train_scaled)
    predict_df = pd.DataFrame(predict,columns=['Predict'])
    y =pd.DataFrame(y_train)
    y.reset_index(drop=True, inplace=True)
    final =pd.concat([y, predict_df], axis=1)
    one = final[final['Status'] == 1]
    zero = final[final['Status'] == 0]
    plt.title('Noise vs Signal Prediction of consolidated events')
    bins = np.linspace(-10, 10, 50)
    plt.hist(one['Predict'],bins, alpha = 0.5, label='Signal',density = False,color = 'm')
    plt.hist(zero['Predict'],bins, alpha = 0.5 ,label='Noise', density = False,color = 'r')
    pyplot.legend(loc='upper right')
    plt.xlim([0, 1])
    plt.show()
    return one,zero



model = basic_model()
plot_output_predictions(model)

end = timer()
print((end - start))
