import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load the data
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

# Extract the blue data, keeping the latest data at the beginning
blues = np.array([entry['blue'] for entry in json_data])

# Define Reference value for standardization
Referencevalue = 16

# Standardize data by subtracting Referencevalue
standardized_data = blues - Referencevalue

# Create sliding window sequences on standardized data from the tail to the head
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 1, -1, -1):  # 从尾部往前滑动
        X.append(data[i+1:i+n_steps+1])  # 过去的16个数据作为输入窗口
        y.append(data[i])  # 预测的目标是窗口前一个数据点
    return np.array(X), np.array(y)

# Parameters
n_steps = 16
X, y = create_sequences(standardized_data, n_steps)
# Build an optimized LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    LSTM(64, return_sequences=True, dropout=0.1),
    BatchNormalization(),
    LSTM(32),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='linear')
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=15)
model.fit(X, y, epochs=200, batch_size=16, callbacks=[early_stopping], verbose=0) # type: ignore

# Prepare the new_data input and standardize it
# Note: using the latest 16 values as input
new_data = np.array([standardized_data[:16]])  # 使用最新的 16 个数据
new_data = new_data.reshape((new_data.shape[0], new_data.shape[1], 1))

# Predict and then de-standardize by adding back the Referencevalue
predicted = model.predict(new_data)
predicted_destandardized = predicted + Referencevalue  # De-standardize

# Print standardized prediction and de-standardized (final) prediction
print("Standardized prediction (difference):", predicted)
# Convert the de-standardized prediction to integers
predicted_destandardized_rounded = np.round(predicted_destandardized, 2)
print("De-standardized prediction (final blue):", predicted_destandardized_rounded)