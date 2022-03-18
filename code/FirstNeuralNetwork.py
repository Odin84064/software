##First neural networks with
# 1)early stopping
# 2) early stopping ,Batch Normalization and dropout

# Params
# batch size =512
# epochs = 1000
# 3 dense layer :128/64neurons
# 1 output layer

import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

os.chdir("../dataset")
df = pd.read_parquet('final10000events.parquet', engine="pyarrow")

# Create training and validation splits
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
# Split features and target
X_train = df_train.drop('Status', axis=1)
X_valid = df_valid.drop('Status', axis=1)
y_train = df_train['Status']
y_valid = df_valid['Status']


##model with early stopping
def model_early_stopping():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[5]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1,activation = 'sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=300,
        min_delta=0.001,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
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
    plt.title('Binary Cross Entropy Loss with Early Stopping')
    plt.savefig('../code/plots/loss_with_earlystopping.jpg')
    plt.close()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title('Accurcay with Early Stopping')
    plt.savefig('../code/plots/accuracy_with_earlystopping.jpg')
    plt.close()

    print(("Best Validation Loss with early stopping: {:0.4f} \nBest Validation Accuracy with early stopping: {:0.4f}")\
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))
    plt.hist(model.predict(X_train),bins = 50)
    plt.show()


def model_batch_normalization():
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
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=512,
        epochs=1000,
        callbacks=[early_stopping],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    plt.savefig('../code/plots/loss_with_batchnormalization.jpg')
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss with BatchNormalization and Dropout')
    plt.close()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title('Binary Cross Entropy Loss with BatchNormalization and Dropout')
    plt.savefig('../code/plots/accuracy_with_batchnormalization.jpg')
    plt.close()

    print(("Best Validation Loss with early stopping: {:0.4f} \nBest Validation Accuracy with early stopping: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

print(model_early_stopping())

print(model_batch_normalization())



plt.hist(model.predict())