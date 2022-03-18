### Using PDG_ID = 211 and 22 as validation set

import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer



os.chdir("../dataset")
df = pd.read_parquet('final1000events.parquet', engine="pyarrow")


##removing PDG_OD 22  from data frame
# df_new = df.drop(df[df['PDG_ID']==22 ].index)
# df_new.reset_index(drop = True,inplace = True)

##removing PDG_OD 211  from data frame
df_new = df.drop(df[df['PDG_ID']==211 ].index)
df_new.reset_index(drop = True,inplace = True)




##validation set1 only PDG_ID =22
df_22 = df[df['PDG_ID'] == 22]
df_22.reset_index(drop = True,inplace = True)

#validation set2 only PDG_ID =22
# df_22 = df[df['PDG_ID'] == 22]
# df_22.reset_index(drop = True,inplace = True)

#validation set2 only PDG_ID =211
df_211 = df[df['PDG_ID'] == 211]
df_211.reset_index(drop = True,inplace = True)



def preprocessing(features,valid):
    features_train = features.drop('Status', axis=1)
    labels_train = features['Status']

    features_valid = valid.drop('Status', axis=1)
    labels_valid = valid['Status']

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
    return features_train_norm,features_valid_norm,labels_train,labels_valid

def basic_model(features_train,features_valid,labels_train,labels_valid):

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
        features_train, labels_train,
        validation_data=(features_valid, labels_valid),
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
    plt.title('Binary Cross Entropy Loss with 200 epochs(211)')
    plt.show()
   # plt.savefig('../code/plots/loss_with_200 Epochs.jpg')
    plt.close()
    history_df.loc[1:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title('Accurcay with 200 Epochs(211)')
    plt.show()
    #plt.savefig('../code/plots/accuracy_with_200 Epochs.jpg')
    plt.close()

    print(("Best Validation Loss with early stopping: {:0.4f} \nBest Validation Accuracy with early stopping: {:0.4f}")\
              .format(history_df['val_loss'].min(),
                      history_df['val_binary_accuracy'].max()))
    return model,features_valid, labels_valid,history_df

def model_without_earlystopping(features_train,features_valid,labels_train,labels_valid):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', input_shape=[5]),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    history = model.fit(
        features_train, labels_train,
        validation_data=(features_valid, labels_valid),
        batch_size=128,
        epochs=50,

        verbose=1,
    )

    history_df = pd.DataFrame(history.history)

    history_df.loc[1:,['loss', 'val_loss']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Binary Cross Entropy Loss 6layers(22) ')
    plt.show()
    plt.close()

    history_df.loc[1:,['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.xlabel("# of epochs")
    plt.ylabel("loss(cross_entropy")
    plt.title('Accurcay 6 Layers(22)')
    plt.show()
    plt.close()



    print((
              "Best Validation Loss: {:0.4f} \nBest Validation Accuracy : {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

    return model, features_valid, labels_valid,history_df
def plot_output_predictions( model,features_valid,labels_valid):
    predict = model.predict(features_valid)
    predict_df = pd.DataFrame(predict,columns=['Predict'])
    y =pd.DataFrame(labels_valid)
    y.reset_index(drop=True, inplace=True)
    final =pd.concat([y, predict_df], axis=1)
    one = final[final['Status'] == 1]
    zero = final[final['Status'] == 0]
    plt.title('Noise vs Signal Prediction(211)')
    bins = np.linspace(-10, 10, 50)
    plt.hist(one['Predict'],bins, alpha = 0.5, label='Signal',density = False,color = 'g')
    plt.hist(zero['Predict'],bins, alpha = 0.5 ,label='Noise', density = False,color = 'r')
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    plt.show()
    return one,zero

# features_train_norm,features_valid_norm,labels_train,labels_valid = preprocessing(df_new,df_22)
# print(model_without_earlystopping(features_train_norm,features_valid_norm,labels_train,labels_valid))


features_train_norm,features_valid_norm,labels_train,labels_valid = preprocessing(df_new,df_211)
model,features_valid,labels_valid,history_df = basic_model(features_train_norm,features_valid_norm,labels_train,labels_valid)
one,zero =plot_output_predictions(model,features_valid,labels_valid)

#features_train_norm,features_valid_norm,labels_train,labels_valid = preprocessing(df_new,df_22)
#model,features_valid,labels_valid = model_without_earlystopping(features_train_norm,features_valid_norm,labels_train,labels_valid)
#one,zero =plot_output_predictions(model,features_valid,labels_valid)