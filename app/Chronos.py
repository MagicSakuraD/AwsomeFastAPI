import numpy as np
import torch
# import pandas as pd
import json
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

# Step 1: 读取 JSON 数据
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

# Step 2: 提取 reds 和 blue 数据
reds = np.array([entry['reds'] for entry in json_data])  # Shape: (N, 6)
blues = np.array([entry['blue'] for entry in json_data])  # Shape: (N,)
print(torch.cuda.is_available())  # 如果返回 True，说明 CUDA 可用
# 合并 reds 和 blues 数据，生成一个单独的序列
combined_data = np.hstack((reds, blues.reshape(-1, 1))).flatten()  # 展平成 1D 序列

# Step 3: 定义参考值并进行标准化
Referencevalue = np.array([4, 10, 14, 19, 24, 28, 8])  # 参考值
standardized_data = combined_data - np.tile(Referencevalue, len(combined_data) // len(Referencevalue))  # 标准化

# Step 4: 转换为 PyTorch 张量
context = torch.tensor(standardized_data, dtype=torch.float32)

# Step 5: 加载 Chronos 模型
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",  # 使用 Chronos-T5-tiny 版本
    device_map="cuda",         # 如果有 GPU
    torch_dtype=torch.bfloat16 # 使用 bfloat16 加速
)

# Step 6: 设置预测长度 (比如预测未来 7 个时间点)
prediction_length = 7

# 使用最近的 16 个时间点作为输入
recent_context = context[-16:]

# Step 7: 进行预测
forecast = pipeline.predict(recent_context.unsqueeze(0), prediction_length=prediction_length)  # 输出形状 [1, num_samples, 7]

# Step 8: 从预测中提取分位数 (中位数、10%-90%置信区间等)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# Step 9: 可视化结果
forecast_index = range(len(combined_data), len(combined_data) + prediction_length)
plt.figure(figsize=(8, 4))
plt.plot(range(len(combined_data)), combined_data, color="royalblue", label="Historical Data")
plt.plot(forecast_index, median, color="tomato", label="Median Forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% Prediction Interval")
plt.legend()
plt.grid()
plt.show()

# Step 10: De-standardize predictions (add back the Referencevalue)
predicted_destandardized = median + Referencevalue
print("De-standardized prediction (final reds and blue):", predicted_destandardized)
