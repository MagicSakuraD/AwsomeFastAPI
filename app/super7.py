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
Referencevalue = np.array([4, 10, 15, 19, 24, 28, 8])

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
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    LSTM(128, return_sequences=True, dropout=0.1),
    BatchNormalization(),
    LSTM(128),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(7, activation='linear')
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=15)
model.fit(X, y, epochs=300, batch_size=16, callbacks=[early_stopping], verbose=0) # type: ignore

# Prepare the new_data input and standardize it
# Note: using the latest 12 values as input
new_data = np.array([combined_data[:16] - Referencevalue])  # 使用最新的 16 个数据

# Predict and then de-standardize by adding back the Referencevalue
predicted = model.predict(new_data)
predicted_destandardized = predicted + Referencevalue  # De-standardize

# Print standardized prediction and de-standardized (final) prediction
print("Standardized prediction (difference):", predicted)
# Convert the de-standardized prediction to integers
predicted_destandardized_rounded = np.round(predicted_destandardized, 2)
print("De-standardized prediction (final reds and blue):", predicted_destandardized_rounded)



# Top 10 Red Balls: 5 (8), 14 (7), 29 (6), 9 (5), 20 (5), 10 (4), 19 (4), 21 (4), 22 (4), 8 (3)
# Top 5 Blue Balls: 7 (3), 9 (3), 8 (2), 10 (2), 12 (2)
# Total red balls: 84, Total blue balls: 
#   {"issue":"24131","reds":[4,5,11,15,20,32],"blue":13},
#   { "issue": "24130", "reds": [1, 8, 12, 17, 19, 24], "blue": 16 },
#   { "issue": "24129", "reds": [9, 10, 13, 19, 24, 32], "blue": 1 },
