# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy
import numpy as np
# import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# from pandas.core.window import online

# accessing data from csv file

def create_dataset(dataset, timestep=1):
    dataX, dataY = [], []
    #print('a')
    for i in range(len(dataset) - timestep):
        #print(i,len(dataset)-timestep)
        p = dataset[i:(i + timestep), 0]
        dataX.append(p)
        dataY.append(dataset[(i + timestep), 0])
    return numpy.array(dataX), numpy.array(dataY)


data = pd.read_csv('dataset/ADANIPORTS.csv')
data.drop(['Symbol', 'Series', 'Last', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble',
           'Prev Close'], axis=1, inplace=True)
print(data.index, data.head())
print(len(data.index))
data['Date'] = pd.to_datetime(data['Date'])

index = int(len(data.index) * 0.8)
# accessing training and testing data from csv
train_set = data.iloc[:index, :]
test_set = data.iloc[index:, :]
print(train_set.shape, test_set.shape, train_set.info())

# creating model
# model = Sequential()
print(train_set.head(3))
train_data = train_set.iloc[:, 4:5]
test_data = test_set.iloc[:, 4:5]
print(train_data)
# Press the green button in the gutter to run the script.

sc = MinMaxScaler(feature_range=(0, 1))
train_data = sc.fit_transform(np.array(train_data).reshape(-1, 1))

test_data = sc.fit_transform(np.array(test_data).reshape(-1, 1))

timestep = 100
train_X, train_Y = create_dataset(train_data, timestep)
test_X, test_Y = create_dataset(test_data, timestep)

#reshape
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(timestep, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=1, batch_size=64, verbose=1)
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)
train_predict = sc.inverse_transform(train_predict)
test_predict = sc.inverse_transform(test_predict)

x = math.sqrt(mean_squared_error(train_Y,train_predict))
y = math.sqrt(mean_squared_error(test_Y,test_predict))

print(x, y)
