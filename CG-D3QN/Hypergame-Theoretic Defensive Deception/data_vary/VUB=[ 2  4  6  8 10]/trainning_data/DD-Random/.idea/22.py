import matplotlib.pyplot as plt
import pickle
import numpy as np

# 加载数据
path = 'all_result_after_each_game_10-06_24-03-2024.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

# 创建一个图形
plt.figure(figsize=(10, 6))

# 准备存储每个线的数据
lines_data = [[] for _ in range(8)]

# 遍历数据并将每组数据的第一个元素放入相应的线的数据列表中
for key, array in data.items():
    for i in range(min(len(array), 8)):  # 遍历每个数组的前8个元素，如果数组长度不足8，则取数组的长度
        lines_data[i].append(array[i])

# 绘制每条线
for i, line_data in enumerate(lines_data):
    plt.plot(line_data, label=f'Line {i+1}')  # 绘制每条线，并添加标签

# 添加标题和标签
plt.title('Line Plot of First Elements for Each Group')
plt.xlabel('Group Index')
plt.ylabel('Value')
plt.legend()  # 添加图例

# 显示图形
plt.show()
