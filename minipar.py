import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

#Reading the dataseet
df = pd.read_csv("pmsm_temperature_data.csv")
col_list = df.columns.tolist()
# print(len(col_list))

#Assigning features and label
profile_id = ['profile_id']
target_list = ['pm', 'torque', 'stator_yoke', 'stator_tooth', 'stator_winding']
feature_list = [col for col in col_list if col not in target_list and col not in profile_id]
target_list = ['pm']

# df_dict = {}
# for id_ in df.profile_id.unique():
#     df_dict[id_] = df[df['profile_id']==id_].reset_index(drop = True)

features = df[feature_list]
target = df[target_list]


#ANNmodel

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = features.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
print(NN_model.summary())

NN_model.fit(features, target, epochs=10, batch_size=32, validation_split = 0.2)

