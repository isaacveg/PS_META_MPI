import os, fnmatch
from os.path import join as ospj
import yaml
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')
matplotlib.rcParams['savefig.format'] = 'png'

result_root = './results/'
plot_root = './plot/'

if not os.path.exists(plot_root):
    os.makedirs(plot_root)
if not os.path.exists(result_root):
    os.makedirs(result_root)

draw_requirements = {
    'meta_method': ['fedavg', 'fomaml', 'mamlhf']
    ,'inner_lr': [0.1, 0.01]
    ,'outer_lr': [0.01, 0.05]
    ,'epoch_num': [200]
}

draw_contents = [
    'train_acc',
    ,'train_loss'
    ,'eval_acc'
    ,'eval_loss'
    ,'test_acc'
    ,'test_loss'
]


def main():
    # 获取文件夹下的所有一级子文件夹名称
    subfolders = [ospj(result_root, f) for f in os.listdir(result_root)]
    # print(subfolders)
    # 获取每次结果对应的config文件信息
    cfgs = [get_config(f) for f in subfolders]
    # 获取每次结果对应的server_log文件目录
    server_logs = [ospj(subfolder, filename) for subfolder in subfolders for filename in os.listdir(subfolder) if fnmatch.fnmatch(filename, '*server.log')]    
    print(server_logs)

    # 获取符合 draw_requirements 的 config 以及对应的信息
    acquired_cfgs = []
    # 遍历 config
    for idx, cfg in enumerate(cfgs):
        for key, value in draw_requirements.items():
            # 如果有一个不满足要求则跳过，否则加入列表
            if cfg[key] not in value:
                continue
        acquired_cfgs.append((idx, cfg))
    
    print([cfg[0] for cfg in acquired_cfgs])

    ## 画准确率部分

    # 读取server_log中的信息并画图
    # fig_train, fig_eval, fig_test = plt.figure(), plt.figure(), plt.figure()
    for idx, cfg in acquired_cfgs:
        if 'train_acc' in draw_contents:
            # plt.figure()
            # 获取每个epoch的eval_acc_before和eval_acc_after的值和epoch_cnt
            t, _, _ = process_server_log(server_logs[idx])
            draw_and_save(t, 'train_acc', cfg)
            # eval_acc_before = [(d.get('eval_acc_before')[0], d.get('eval_acc_before')[1]) for d in t]
            # eval_acc_after = [(d.get('eval_acc_after')[0], d.get('eval_acc_after')[1]) for d in t]

            # # 分离epoch_cnt和eval_acc_before和eval_acc_after的值
            # epochs, eval_acc_before_values = zip(*eval_acc_before)
            # _, eval_acc_after_values = zip(*eval_acc_after)

            # # 绘制eval_acc_before和eval_acc_after
            # plt.plot(epochs, eval_acc_before_values, label='eval_acc_before')
            # plt.plot(epochs, eval_acc_after_values, label='eval_acc_after')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # plt.title('Training Evaluation Accuracy')
            # plt.legend()
            # result_str = "_".join([ str(cfg[item]) for item in draw_requirements.keys()])
            # plt.savefig(plot_root+'/train_acc_{}.png'.format(result_str))
        
    

def draw_train_acc(t, requirement, cfg):
    plt.figure()
    # 获取每个epoch的eval_acc_before和eval_acc_after的值和epoch_cnt
    # t, _, _ = process_server_log(server_logs[idx])
    eval_acc_before = [(d.get('eval_acc_before')[0], d.get('eval_acc_before')[1]) for d in t]
    eval_acc_after = [(d.get('eval_acc_after')[0], d.get('eval_acc_after')[1]) for d in t]

    # 分离epoch_cnt和eval_acc_before和eval_acc_after的值
    epochs, eval_acc_before_values = zip(*eval_acc_before)
    _, eval_acc_after_values = zip(*eval_acc_after)

    # 绘制eval_acc_before和eval_acc_after
    plt.plot(epochs, eval_acc_before_values, label='eval_acc_before')
    plt.plot(epochs, eval_acc_after_values, label='eval_acc_after')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Evaluation Accuracy')
    plt.legend()
    result_str = "_".join([ str(cfg[item]) for item in draw_requirements.keys()])
    plt.savefig(plot_root+'/train_acc_{}.png'.format(result_str))


def process_server_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()   
        train_data, eval_data, global_data = [],[],[]
        epoch_cnt = 0
        for line in lines:
            if 'Epoch:' in line:
                epoch_cnt = int(line.split(': ')[1])
            elif 'Selected client idxes' in line:
                if 'Eval' in lines[lines.index(line)+1]:
                    train_data.append({})
                    eval_acc_before = float(lines[lines.index(line)+1].split(': ')[1])
                    eval_acc_after = float(lines[lines.index(line)+2].split(': ')[1])
                    eval_loss_before = float(lines[lines.index(line)+3].split(': ')[1])
                    eval_loss_after = float(lines[lines.index(line)+4].split(': ')[1])
                    train_data[-1]['eval_acc_before'] = (epoch_cnt, eval_acc_before)
                    train_data[-1]['eval_acc_after'] =  (epoch_cnt, eval_acc_after)
                    train_data[-1]['eval_loss_before'] =  (epoch_cnt, eval_loss_before)
                    train_data[-1]['eval_loss_after'] =  (epoch_cnt, eval_loss_after)
            elif 'Evaling clients' in line:
                eval_data.append({})
                eval_acc_before = float(lines[lines.index(line)+1].split(': ')[1])
                eval_acc_after = float(lines[lines.index(line)+2].split(': ')[1])
                eval_loss_before = float(lines[lines.index(line)+3].split(': ')[1])
                eval_loss_after = float(lines[lines.index(line)+4].split(': ')[1])
                eval_data[-1]['eval_acc_before'] =(epoch_cnt, eval_acc_before)
                eval_data[-1]['eval_acc_after'] = (epoch_cnt, eval_acc_after)
                eval_data[-1]['eval_loss_before'] =  (epoch_cnt, eval_loss_before)
                eval_data[-1]['eval_loss_after'] = (epoch_cnt, eval_loss_after)
            elif 'Test_Loss' in line:
                global_data.append({})
                test_loss = float(line.split(': ')[1])
                test_acc = float(lines[lines.index(line)+1].split(': ')[1])
                global_data[-1]['test_loss'] = test_loss
                global_data[-1]['test_acc'] = test_acc
    return train_data, eval_data, global_data

# 
def get_config(path_name):
    filename = ospj(path_name,'config.yml')
    with open(filename, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg



if __name__ == "__main__":
    main()
