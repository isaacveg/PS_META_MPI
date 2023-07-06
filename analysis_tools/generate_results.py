import os
import shutil
import glob
from datetime import datetime


# 获取最新的 model_save 文件夹
model_save = max(glob.glob('./model_save/*/'), key=os.path.getctime)

# 获取最新的 server_log 文件
server_log = max(glob.glob('./server_log/*server.log'), key=os.path.getctime)

# 获取最新的 config 文件
config_save = max(glob.glob('./config_save/*'), key=os.path.getctime)

# 获取最新的 clients_log 子文件夹
clients_log = max(glob.glob('./clients_log/*/'), key=os.path.getctime)

# 获取时间戳
timestamp = datetime.strptime(os.path.basename(server_log)[:-11], '%Y-%m-%d-%H_%M_%S')
print(timestamp)

# 在 results 文件夹中创建一个新文件夹
try:
    result_dir = os.path.join('./results/', timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(result_dir)

    # 将文件夹移动到 results 文件夹中
    shutil.move(model_save, os.path.join(result_dir, 'model_save'))
    shutil.move(server_log, result_dir)
    shutil.move(clients_log, os.path.join(result_dir, 'clients_log'))
    shutil.move(config_save, os.path.join(result_dir, 'config.yml'))
except Exception as e :
    print("Error: ", e)

# 如果移动后原文件夹是空的，把它们删除
if not os.listdir('./model_save'):
    os.rmdir('./model_save')
    print("model_save removed")
if not os.listdir('./server_log'):
    os.rmdir('./server_log')
    print("server_log removed")
if not os.listdir('./clients_log'):
    os.rmdir('./clients_log')
    print("clients_log removed")
if not os.listdir('./config_save'):
    os.rmdir('./config_save')
    print("config_save removed")
    