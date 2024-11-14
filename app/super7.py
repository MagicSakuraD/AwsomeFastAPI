import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam

# Load the data
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

# Extract the reds and blue data, keeping the latest data at the beginning
reds = np.array([entry['reds'] for entry in json_data])
blues = np.array([entry['blue'] for entry in json_data])
combined_data = np.hstack((reds, blues.reshape(-1, 1)))  # Combine reds and blue
# Define Reference values for standardization
Referencevalue = np.array([4, 9, 17, 21, 24, 29, 9])

# Standardize data by subtracting Referencevalue
standardized_data = combined_data - Referencevalue

# Create sliding window sequences on standardized data from the tail to the head
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 1, -1, -1):  # 从尾部往前滑动
        X.append(data[i+1:i+n_steps+1])  # 过去的12个数据作为输入窗口
        y.append(data[i])  # 预测的目标是窗口前一个数据点
    return np.array(X), np.array(y)

# Parameters
n_steps = 16
X, y = create_sequences(standardized_data, n_steps)

# Build an optimized LSTM model
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), dropout=0.2),
    BatchNormalization(),
    LSTM(128, return_sequences=True, dropout=0.3),
    BatchNormalization(),
    LSTM(64),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='linear')
])
# Compile with adjusted learning rate and optimizer
optimizer = RMSprop(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='huber')
# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10)
model.fit(X, y, epochs=270, batch_size=12, callbacks=[early_stopping], verbose=0) # type: ignore

# Prepare the new_data input and standardize it
# Note: using the latest 12 values as input
new_data = np.array([combined_data[:16] - Referencevalue])  # 使用最新的 16 个数据

# Predict and then de-standardize by adding back the Referencevalue
predicted = model.predict(new_data)
predicted_destandardized = predicted + Referencevalue  # De-standardize

# Print standardized prediction and de-standardized (final) prediction
print("Standardized prediction (difference):", predicted)
# Convert the de-standardized prediction to integers
predicted_destandardized_int = np.round(predicted_destandardized).astype(int)
print("De-standardized prediction (final reds and blue):", predicted_destandardized_int)


# 7 - 11 - 15 - 21 - 24 - 30 - 7   5 - 9 - 11 - 21 - 22 - 26 - 8    5 - 9 - 14 - 19 - 20 - 28 - 8    5 - 9 - 13 - 17 - 20 - 29 - 10
# 8 - 12 - 14 - 19 - 22 - 29 - 7   5 - 10 - 13 - 18 - 20 - 26 - 12  8 - 14 - 17 - 22 - 26 - 30 - 9   3 - 8 - 14 - 18 - 22 - 27 - 11
# 5 - 10 - 15 - 20 - 23 - 29 - 10   5 - 11 - 14 - 18 - 21 - 29 - 9   5 - 9 - 14 - 19 - 24 - 29 - 9   4 - 9 - 14 - 19 - 24 - 29 - 5
# 4 - 10 - 15 - 20 - 23 - 27 - 12   5 - 10 - 13 - 16 - 21 - 28 - 7


# Top 10 Red Balls: 5 (8), 14 (7), 29 (6), 9 (5), 20 (5), 10 (4), 19 (4), 21 (4), 22 (4), 8 (3)
# Top 5 Blue Balls: 7 (3), 9 (3), 8 (2), 10 (2), 12 (2)
# Total red balls: 84, Total blue balls: 14