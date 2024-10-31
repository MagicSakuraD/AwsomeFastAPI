import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop

# Load the data
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

reds = np.array([entry['reds'] for entry in json_data])
blues = np.array([entry['blue'] for entry in json_data])
combined_data = np.hstack((reds, blues.reshape(-1, 1)))  # Combine reds and blue

# Define Reference values for standardization
Referencevalue = np.array([4, 9, 14, 19, 24, 29, 8])

# Standardize data by subtracting Referencevalue
standardized_data = combined_data - Referencevalue

# Create sliding window sequences on standardized data
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 12
X, y = create_sequences(standardized_data, n_steps)

# Build the optimized LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16, return_sequences=True))  # New LSTM layer
model.add(LSTM(8))  # Last LSTM layer with return_sequences=False
model.add(Dense(y.shape[1]))  # Output layer

# Compile the model
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10)
model.fit(X, y, epochs=270, batch_size=12, callbacks=[early_stopping], verbose=0) # type: ignore

# Prepare the new_data input and standardize it
new_data = np.array([combined_data[-12:] - Referencevalue])  # Standardized recent data for prediction

# Predict and then de-standardize by adding back the Referencevalue
predicted = model.predict(new_data)
predicted_destandardized = predicted + Referencevalue  # De-standardize

# Print standardized prediction and de-standardized (final) prediction
print("Standardized prediction (difference):", predicted)
print("De-standardized prediction (final reds and blue):", predicted_destandardized)

#  {"issue":"24124","reds":[2,14,15,17,25,30],"blue":11},