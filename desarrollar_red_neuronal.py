
#obtener datos

import pandas as pd

dataset = pd.read_csv('Datasets/pima-indians-diabetes.csv')
print(dataset.head(10))
print(dataset.shape)
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

#dividir datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

import keras
from keras.models import Sequential
from keras.layers import Dense

#Modelizacion
model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#Compilar Modelo

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Entrenamiento
history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.1)
