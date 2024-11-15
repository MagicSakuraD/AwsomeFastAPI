import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop

# Load the data
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

reds = np.array([entry['reds'] for entry in json_data])
blues = np.array([entry['blue'] for entry in json_data])
combined_data = np.hstack((reds, blues.reshape(-1, 1)))  # Combine reds and blue

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 1, -1, -1):  # 从尾部往前滑动
        X.append(data[i+1:i+n_steps+1])  # 过去的12个数据作为输入窗口
        y.append(data[i])  # 预测的目标是窗口前一个数据点
    return np.array(X), np.array(y)

n_steps = 32
X, y = create_sequences(combined_data, n_steps)

model = Sequential()

# First LSTM layer with bidirectional processing
model.add(LSTM(256, return_sequences=True, 
               dropout=0.3, recurrent_dropout=0.2,
               input_shape=(X.shape[1], X.shape[2])))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))

# Second LSTM layer with reduced complexity
model.add(LSTM(128, return_sequences=True))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

# Final LSTM layer
model.add(LSTM(64))
model.add(LeakyReLU(alpha=0.2))

# Dense layers with gradual dimension reduction
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1]))  # Output layer

# Compile with adjusted learning rate
optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# Train with increased epochs and patience
early_stopping = EarlyStopping(monitor='loss', patience=20)
model.fit(X, y, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0) # type: ignore

# Predict using the last 32 periods of data to match model's expected input shape
new_data = np.array([combined_data[-n_steps:]])  # Use n_steps periods of recent data
predicted = model.predict(new_data)
print("Predicted reds and blue:", predicted)

#   { "issue": "24130", "reds": [1, 8, 12, 17, 19, 24], "blue": 16 },
#   { "issue": "24129", "reds": [9, 10, 13, 19, 24, 32], "blue": 1 },