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

# Create sliding window sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 12
X, y = create_sequences(combined_data, n_steps)

# Build the optimized LSTM model
model = Sequential()
# 构建模型
model.add(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
model.add(LSTM(16, dropout=0.3, recurrent_dropout=0.3))  # 最后一层LSTM
model.add(Dense(y.shape[1]))  # Output layer

# model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
# model.add(LSTM(32, return_sequences=True))
# model.add(LSTM(16, return_sequences=True))  # 新增的LSTM层
# model.add(LSTM(8))  # 最后一层LSTM，return_sequences=False
# model.add(Dense(y.shape[1]))  # Output layer


# 编译模型
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')  # 必须先compile
# Print model summary
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10)
model.fit(X, y, epochs=270, batch_size=12, callbacks=[early_stopping], verbose=0) # type: ignore

# Predict using the last 5 periods of data
new_data = np.array([combined_data[-12:]])  # Use 5 periods of recent data
predicted = model.predict(new_data)
print("Predicted reds and blue:", predicted)

#  {"issue":"24124","reds":[2,14,15,17,25,30],"blue":11},
#  
