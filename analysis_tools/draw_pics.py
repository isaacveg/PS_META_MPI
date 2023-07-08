import os
from os.path import join as ospj
import yaml

result_root = './results/'

draw_requirements = {
    'meta_method': ['fedavg', 'fomaml', 'mamlhf']
}

draw_contents = [
    'avg_'
]


def main():
    # 获取文件夹下的所有一级子文件夹名称
    subfolders = [ospj(result_root, f) for f in os.listdir(result_root)]
    # print(subfolders)
    # 获取每次结果对应的config文件信息
    cfgs = [get_config(f) for f in subfolders]
    # 获取每次结果对应的server_log文件目录
    server_logs = [ospj(subfolder, os.path.basename(subfolder) + "_server.log") for subfolder in subfolders]
    # print(server_logs)




# 
def get_config(path_name):
    filename = ospj(path_name,'config.yml')
    with open(filename, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg



if __name__ == "__main__":
    main()
