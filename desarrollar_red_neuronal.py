import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense, input_dim=8, activation='relu')
model.add(12, Dense, activation='relu')
model.add(8,Dense, activation = 'relu')
model.add(1,Dense, activation = 'sigmoid')