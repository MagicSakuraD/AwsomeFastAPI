import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Load the data
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

reds = np.array([entry['reds'] for entry in json_data])
blues = np.array([entry['blue'] for entry in json_data])
combined_data = np.hstack((reds, blues.reshape(-1, 1)))  # Combine reds and blue

# Create sliding window sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 5
X, y = create_sequences(combined_data, n_steps)

# Build the optimized LSTM model
model = Sequential()
model.add(LSTM(32, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(15, activation='tanh'))  # Second LSTM layer
# model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Dense(y.shape[1]))  # Output layer
model.compile(optimizer='RMSprop', loss='mse')  # Using RMSprop optimizer

# Print model summary
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10)
model.fit(X, y, epochs=300, batch_size=5, callbacks=[early_stopping], verbose=0) # type: ignore

# Predict using the last 5 periods of data
new_data = np.array([combined_data[-5:]])  # Use 5 periods of recent data
predicted = model.predict(new_data)
print("Predicted reds and blue:", predicted)


# {
#       "issue": "24115",
#       "reds": [
#         3,
#         10,
#         11,
#         19,
#         27,
#         28
#       ],
#       "blue": 7
#     },

#  {
#       "issue": "24114",
#       "reds": [
#         7,
#         11,
#         18,
#         24,
#         27,
#         32
#       ],
#       "blue": 4
#     },