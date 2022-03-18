##Basic neural networks with
# 1)Standardscalar,,Batchnormalization,Dropout,Early stopping
# 2) Normalizer,Batchnormalization,Dropout,Early stopping

# Params
# batch size =512
# epochs = 1000
# 3 dense layer :128/64 neurons
# 1 output layer







import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

os.chdir("../dataset")
df = pd.read_parquet('final10000events.parquet', engine="pyarrow")


features = df.drop('Status', axis=1)
labels = df['Status']



# Create training and validation splits
features_train, features_valid, labels_train, labels_valid \
    = train_test_split(features, labels, test_size=0.33, random_state=42)

print(list(features_train.columns))

##  Standardize
my_ct = ColumnTransformer([('scale', StandardScaler(), list(features_train.columns))]
, remainder='passthrough')
features_train_scale = my_ct.fit_transform(features_train)
features_valid_scale = my_ct.transform(features_valid)
features_train_scale = pd.DataFrame(features_train_scale
,columns = features_train.columns)
features_valid_scale = pd.DataFrame(features_valid_scale
,columns = features_valid.columns)

#Normalize
# my_ct = ColumnTransformer([('norm', Normalizer(), list(features_train.columns))]
# , remainder='passthrough')
# features_train_norm = my_ct.fit_transform(features_train)
# features_valid_norm = my_ct.transform(features_valid)
# features_train_norm = pd.DataFrame(features_train_norm
# ,columns = features_train.columns)
# features_valid_norm = pd.DataFrame(features_valid_norm
# ,columns = features_valid.columns)

max_ = features_train.max(axis=0)
min_ = features_train.min(axis=0)
features_train_norm = (features_train - min_) / (max_ - min_)
features_valid_norm = (features_valid - min_) / (max_ - min_)

def model_Standarization():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=20,
        min_delta=0.001,
        restore_best_weights=True,
        monitor='val_loss', mode='min'
    )
    history = model.fit(
        features_train_scale, labels_train,
        validation_data=(features_valid_scale, labels_valid),
        batch_size=512,
        epochs=1000,
        callbacks=[early_stopping],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss with Standardization')
    plt.savefig('../code/plots/loss_with_Standardization.jpg')
    plt.close()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Accurcay with Standardization')
    plt.savefig('../code/plots/accuracy_with_Standardization.jpg')
    plt.close()

    print(("Best Validation Loss with Standarization: {:0.4f} \nBest Validation Accuracy with Standardization: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))



def model_Normalization():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[5]),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=20,
        min_delta=0.001,
        restore_best_weights=True,
        monitor='val_loss', mode='min'
    )
    history = model.fit(
        features_train_norm, labels_train,
        validation_data=(features_valid_norm, labels_valid),
        batch_size=512,
        epochs=1000,
        callbacks=[early_stopping],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss with Normalization')
    plt.savefig('../code/plots/loss_with_Normalization.jpg')
    plt.close()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Accurcay with Normalization')
    plt.savefig('../code/plots/accuracy_with_Normalization.jpg')
    plt.close()

    print(("Best Validation Loss with Normalization: {:0.4f} \nBest Validation Accuracy with Normalization: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

def model_Normalization_NoEarlyStopping():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    history = model.fit(
        features_train_norm, labels_train,
        validation_data=(features_valid_norm, labels_valid),
        batch_size=512,
        epochs=100,

        verbose=1,
    )

    history_df = pd.DataFrame(history.history)

    history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.ylim(0,100)
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss with Normalization and no Stopping')
    plt.savefig('../code/plots/loss_with_NormalizationandnoStopping.jpg')
    plt.close()
    history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.ylim(0, 100)
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Accurcay with Normalization and no Stopping')
    plt.savefig('../code/plots/accuracy_with_NormalizationandnoStopping.jpg')
    plt.close()

    print(("Best Validation Loss with NormalizationandnoStopping: {:0.4f} \nBest Validation Accuracy with NormalizationandnoStopping: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

#print(model_Standarization())
#print(model_Normalization())
print(model_Normalization_NoEarlyStopping())