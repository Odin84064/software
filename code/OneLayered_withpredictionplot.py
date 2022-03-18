
#printing model prediction also
#basic model with 5 input layer and no early stopping

import pandas as pd
import re
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from timeit import default_timer as timer

start = timer()



print("Current working directory: {0}".format(os.getcwd()))
os.chdir("dataset/")

print("Current working directory: {0}".format(os.getcwd()))
df = pd.read_parquet('final1000events.parquet', engine="pyarrow")

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


def basic_model():

    model = keras.Sequential([
        layers.Dense(5, activation='relu', input_shape=[5]),
        layers.Dense(1, activation='sigmoid'),
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
        batch_size=256,
        epochs=200,
       # callbacks=[early_stopping],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[1:, ['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss with 200 epochs')
    plt.show()
    plt.savefig('../code/plots/loss_with_200 Epochs.jpg')
    plt.close()
    history_df.loc[1:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title('Accurcay with 200 Epochs')
    plt.show()
    plt.savefig('../code/plots/accuracy_with_200 Epochs.jpg')
    plt.close()

    print(("Best Validation Loss with early stopping: {:0.4f} \nBest Validation Accuracy with early stopping: {:0.4f}")\
              .format(history_df['val_loss'].min(),
                      history_df['val_binary_accuracy'].max()))
    return model,history_df
model = basic_model()

def plot_output_predictions( model):
    predict = model[0].predict(X_train)
    predict_df = pd.DataFrame(predict,columns=['Predict'])
    y =pd.DataFrame(y_train)
    y.reset_index(drop=True, inplace=True)
    final =pd.concat([y, predict_df], axis=1)
    one = final[final['Status'] == 1]
    zero = final[final['Status'] == 0]
    plt.title('Noise vs Signal Prediction')
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
print((end - start)*60)
