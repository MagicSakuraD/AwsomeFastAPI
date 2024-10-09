import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 从本地文件读取数据
with open('history.json', 'r') as file:
    json_data = json.load(file)

# 提取 reds 和 blue 组合特征
reds = np.array([entry['reds'] for entry in json_data])
blues = np.array([entry['blue'] for entry in json_data])

# 合并 reds 和 blue 作为特征
combined_data = np.hstack((reds, blues.reshape(-1, 1)))  # 将蓝球合并到红球后

# 构建滑动窗口时序数据（例如使用前 3 期预测下 1 期）
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# 我们使用前 3 期来预测下一期
n_steps = 3
X, y = create_sequences(combined_data, n_steps)

# 创建 LSTM 模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))  # 输入形状为 (时间步, 特征)
model.add(Dense(y.shape[1]))  # 输出层，预测 reds 和 blue 的值（7 个数值）
model.compile(optimizer='adam', loss='mse')

# 打印模型总结
model.summary()

# 训练模型
model.fit(X, y, epochs=300, batch_size=1, verbose=1) # type: ignore

# 使用模型进行预测
new_data = np.array([combined_data[-3:]])  # 最近的 3 期数据
predicted = model.predict(new_data)
print("Predicted reds and blue:", predicted)