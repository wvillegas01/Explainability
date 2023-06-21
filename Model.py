import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Data collection and preparation
drivers_data = 'drivers.csv'
qualifying_data = 'qualifying.csv'
lap_times_data = 'lap_times.csv'

# Load the data from the CSV files
drivers = pd.read_csv(drivers_data)
qualifying = pd.read_csv(qualifying_data)
lap_times = pd.read_csv(lap_times_data)

print(drivers.columns)  # Show the columns in the DataFrame drivers
print(qualifying.columns)  # Display the columns in the DataFrame qualifying results
print(lap_times.columns)  # Show the columns in the DataFrame lap_times

# Check for duplicates in 'driverId' columns
qualifying_duplicates = qualifying[qualifying['driverId'].duplicated()]
drivers_duplicates = drivers[drivers.duplicated(subset=['driverId'], keep=False)]

# Perform the union of the data using the appropriate columns
# (make any necessary adjustments based on your data structure)
data = pd.merge(drivers, qualifying, on='driverId', how='inner')
data = pd.merge(data, lap_times, on=['raceId', 'driverId'], how='inner')

# Perform data cleansing and transformation
# (make the necessary adjustments according to your needs)
data = data.dropna()  # Elimina filas con valores faltantes

# Select the relevant columns from each table
drivers_selected = drivers[['driverId', 'number']]
qualifying_results_selected = qualifying[['driverId', 'position']]
lap_times_selected = lap_times[['raceId', 'driverId', 'lap', 'position']]
#"""
print(drivers_selected.head())  # Muestra las primeras filas del DataFrame drivers_selected
print(qualifying_results_selected.head())  # Muestra las primeras filas del DataFrame qualifying_results_selected
print(lap_times_selected.head())  # Muestra las primeras filas del DataFrame lap_times_selected

print('driverId' in drivers_selected.columns)  # Verifica si la columna 'driverId' existe en drivers_selected
print('driverId' in qualifying_results_selected.columns)  # Verifica si la columna 'driverId' existe en qualifying_results_selected
#"""
# Rename the column in qualifying results select using loc
qualifying_results_selected.loc[:, 'driverId'] = qualifying_results_selected['driverId'].rename('driverId_new')

# Create an explicit copy of drivers_selected using the copy method
drivers_selected_copy = drivers_selected[['driverId', 'number']].copy()

# Perform data union again
data = pd.merge(data, qualifying_results_selected, on='driverId', how='inner')
data = pd.merge(data, drivers_selected_copy, on='driverId', how='inner')

#print(data.columns)

# Divide los datos en conjuntos de entrenamiento y prueba
X = data[['driverId', 'position', 'lap']]  # Selecciona las características relevantes
y = data['position']  # Define la variable objetivo como 'position'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 2: Construction of the LSTM model
model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))


# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')


# Step 3: Training the model
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, epochs=5, batch_size=32)

# Save the trained model
model.save('modelo_entrenado.h5')

