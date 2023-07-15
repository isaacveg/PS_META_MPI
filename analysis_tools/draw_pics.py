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

# Add fileters below that you don't want to draw
result_filters = {
    'meta_method': ['fomaml', 'mamlhf', 'fedavg']
    ,'inner_lr': [0.01]
    ,'outer_lr': [0.05, 0.1]
    ,'epoch_num': [200]
    ,'data_partition_pattern':[1]
    ,'non_iid_ratio': [7]
}

# Comment following lines that you don't want to draw 
draw_contents = [
    'train_acc'
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

    # 获取符合 result_filters 的 config 以及对应的信息
    acquired_cfgs = []
    # 遍历 config
    for idx, cfg in enumerate(cfgs):
        flag = 0
        for key, value in result_filters.items():
            # 如果有一个不满足要求则跳过，否则加入列表
            if cfg[key] not in value:
                flag = 1
                break
        if flag == 0:acquired_cfgs.append((idx, cfg))
    
    print([cfg[0] for cfg in acquired_cfgs])

    ## 画准确率部分
    if 'train_acc' in draw_contents:
        plt.figure()
        for idx, cfg in acquired_cfgs:
            # 获取每个epoch的eval_acc_before和eval_acc_after的值和epoch_cnt
            t, _, _ = process_server_log(server_logs[idx])
            # draw_and_save(t, 'train_acc', cfg)
            eval_acc_before = [(d.get('eval_acc_before')[0], d.get('eval_acc_before')[1]) for d in t]
            eval_acc_after = [(d.get('eval_acc_after')[0], d.get('eval_acc_after')[1]) for d in t]

            # 分离epoch_cnt和eval_acc_before和eval_acc_after的值
            epochs, eval_acc_before_values = zip(*eval_acc_before)
            _, eval_acc_after_values = zip(*eval_acc_after)

            result_str = "_".join([ str(cfg[item]) for item in result_filters.keys()])
            # 绘制eval_acc_before和eval_acc_after
            plt.plot(epochs, eval_acc_before_values, label='bf_'+result_str)
            plt.plot(epochs, eval_acc_after_values, label='af_'+result_str)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Evaluation Accuracy')
        plt.legend() 
        plt.savefig(plot_root+'/train_acc.png')

    ## 画eval_acc部分
    if 'eval_acc' in draw_contents:
        plt.figure()
        for idx, cfg in acquired_cfgs:
            # 获取每个epoch的eval_acc_before和eval_acc_after的值和epoch_cnt
            _, t, _ = process_server_log(server_logs[idx])
            # draw_and_save(t, 'train_acc', cfg)
            eval_acc_before = [(d.get('eval_acc_before')[0], d.get('eval_acc_before')[1]) for d in t]
            eval_acc_after = [(d.get('eval_acc_after')[0], d.get('eval_acc_after')[1]) for d in t]
            # print(cfg['meta_method'],eval_acc_after)
            # 分离epoch_cnt和eval_acc_before和eval_acc_after的值
            epochs, eval_acc_before_values = zip(*eval_acc_before)
            _, eval_acc_after_values = zip(*eval_acc_after)

            # print(cfg['meta_method'],eval_acc_after_values)

            result_str = "_".join([ str(cfg[item]) for item in result_filters.keys()])
            # 绘制eval_acc_before和eval_acc_after
            plt.plot(epochs, eval_acc_before_values, label='bf_'+result_str)
            plt.plot(epochs, eval_acc_after_values, label='af_'+result_str)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Eval_clients Evaluation Accuracy')
        plt.legend() 
        plt.savefig(plot_root+'/eval_acc.png')
    
    ## 画 test acc部分
    if 'test_acc' in draw_contents:
        plt.figure()
        for idx, cfg in acquired_cfgs:
            _, _, t = process_server_log(server_logs[idx])
            test_acc_before_values = [d.get('test_acc') for d in t]
            result_str = "_".join([ str(cfg[item]) for item in result_filters.keys()])
            epochs = range(1, len(test_acc_before_values)+1)
            plt.plot(epochs, test_acc_before_values, label=result_str)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Global_meta_model Accuracy')
        plt.legend() 
        plt.savefig(plot_root+'/test_acc.png')
    



def process_server_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()   
        train_data, eval_data, global_data = [],[],[]
        epoch_cnt = 0
        line_cnt = 0
        for line in lines:
            if 'Epoch:' in line:
                epoch_cnt = int(line.split(': ')[1])
            elif 'Selected client idxes' in line:
                if 'Eval' in lines[lines.index(line)+1]:
                    train_data.append({})
                    eval_acc_before = float(lines[lines.index(line, line_cnt)+1].split(': ')[1])
                    eval_acc_after = float(lines[lines.index(line, line_cnt)+2].split(': ')[1])
                    eval_loss_before = float(lines[lines.index(line, line_cnt)+3].split(': ')[1])
                    eval_loss_after = float(lines[lines.index(line, line_cnt)+4].split(': ')[1])
                    train_data[-1]['eval_acc_before'] = (epoch_cnt, eval_acc_before)
                    train_data[-1]['eval_acc_after'] =  (epoch_cnt, eval_acc_after)
                    train_data[-1]['eval_loss_before'] =  (epoch_cnt, eval_loss_before)
                    train_data[-1]['eval_loss_after'] =  (epoch_cnt, eval_loss_after)
            elif 'Evaling clients:' in line:
                eval_data.append({})
                eval_acc_before = float(lines[lines.index(line, line_cnt)+1].split(': ')[1])
                eval_acc_after = float(lines[lines.index(line, line_cnt)+2].split(': ')[1])
                eval_loss_before = float(lines[lines.index(line, line_cnt)+3].split(': ')[1])
                eval_loss_after = float(lines[lines.index(line, line_cnt)+4].split(': ')[1])
                eval_data[-1]['eval_acc_before'] =(epoch_cnt, eval_acc_before)
                eval_data[-1]['eval_acc_after'] = (epoch_cnt, eval_acc_after)
                eval_data[-1]['eval_loss_before'] =  (epoch_cnt, eval_loss_before)
                eval_data[-1]['eval_loss_after'] = (epoch_cnt, eval_loss_after)
            elif 'Test_Loss' in line:
                global_data.append({})
                test_loss = float(line.split(': ')[1])
                test_acc = float(lines[lines.index(line, line_cnt)+1].split(': ')[1])
                global_data[-1]['test_loss'] = test_loss
                global_data[-1]['test_acc'] = test_acc
            line_cnt += 1
    return train_data, eval_data, global_data

# 
def get_config(path_name):
    filename = ospj(path_name,'config.yml')
    with open(filename, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg



if __name__ == "__main__":
    main()
