import os
import shutil
import glob
from datetime import datetime


# 获取最新的 model_save 文件夹
model_save = max(glob.glob('./model_save/*/'), key=os.path.getctime)

# 获取最新的 server_log 文件夹
server_log = max(glob.glob('./server_log/*server.log'), key=os.path.getctime)

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
    # 复制一份config文件
    shutil.copy('./config.yml', result_dir)
except Exception as e :
    print("Error: ", e)