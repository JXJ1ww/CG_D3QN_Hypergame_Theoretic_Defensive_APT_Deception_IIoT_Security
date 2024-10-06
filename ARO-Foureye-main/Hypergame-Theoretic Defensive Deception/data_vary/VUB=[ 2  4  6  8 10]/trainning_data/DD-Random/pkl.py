import matplotlib.pyplot as plt

# 您提供的数据
# show_pkl.py

import pickle

path = 'all_result_after_each_game_21-18_10-03-2024.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)
print(data)
print(len(data))

#data = [0.125] * 8

# 创建一个新的图形
plt.figure(figsize=(8, 6))

# 绘制条形图
plt.bar(range(len(data)), data)

# 添加标题和标签
plt.title('Bar Chart of Data')
plt.xlabel('Index')
plt.ylabel('Value')

# 显示图形
plt.show()
