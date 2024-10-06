# show_pkl.py

import pickle

path = 'all_result_after_each_game_10-06_23-03-2024.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))

print(type(data))
