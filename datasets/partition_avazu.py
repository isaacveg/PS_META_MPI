import csv 
import numpy as np
import logging
from collections import defaultdict
import pandas as pd

dataset_path = '/data/slwang/datasets/avazu/mini_set.csv'
device_id_partition_path = '/data/slwang/datasets/avazu/partition/device_id_partition.csv'
data_partition_path = '/data/slwang/datasets/avazu/partition/data_partition.csv'
device_id_data_path = '/data/slwang/datasets/avazu/partition/device_id_data.csv'

worker_num = 100
data_num = 90000

partition_type = 11  # device_id: 11, device_ip: 12


# 取出数据集中所有的device_id
with open(dataset_path, 'r') as f: 
    reader = csv.reader(f)  
    device_ids = [row[partition_type] for row in reader]
    device_ids = device_ids[: data_num + 1]
del device_ids[0]
device_ids = set(device_ids)
device_ids = list(device_ids)
print(len(device_ids))

# 将所有device_id平均分配到所有客户端
device_ids_partition = np.array_split(device_ids, worker_num)
f = open(device_id_partition_path,'w',encoding='utf8',newline='')
csvwrite = csv.writer(f)
for i in range(worker_num):
    csvwrite.writerow(device_ids_partition[i])
print(len(device_ids_partition[0]))

# 每一个device_id属于某一个客户端
device_id_worker_dict = defaultdict(int)
for i in range(worker_num):
    for j in range(len(device_ids_partition[i])):
        device_id_worker_dict[device_ids_partition[i][j]] = i

# 将每一条数据根据device_id划分到对应的客户端，并得出每一个device_id包含哪些数据
data_idx_partition = [[] for k in range(worker_num)]
cnt = 0
device_id_datas = defaultdict(list)
with open(dataset_path, encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        device_id = row[partition_type]
        if device_id != 'device_id':
            worker_id = device_id_worker_dict[device_id]
            data_idx_partition[worker_id].append(cnt)
            device_id_datas[device_id].append(cnt)
            cnt += 1
            if cnt >= data_num:
                break
f.close()
f=open(data_partition_path,'w',encoding='utf8',newline='')
csvwrite=csv.writer(f)
for i in range(worker_num):
    csvwrite.writerow(data_idx_partition[i])
f=open(device_id_data_path,'w',encoding='utf8',newline='')
csvwrite=csv.writer(f)
for i in range(len(device_ids)):
    new_array = []
    new_array.append(device_ids[i])
    new_array.extend(device_id_datas[device_ids[i]])
    csvwrite.writerow(new_array)