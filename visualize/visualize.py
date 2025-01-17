import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import argparse

def plot_test_acc(exp_dir, save_dir):
    global prefix
    plt.figure(figsize=(10, 6))
    color_dict = {
        'FP32': '#7EC8E3',  # 浅蓝色
        'FP16-AMP': '#FFB37A',  # 浅橙色
        'FP16-AMP-LOSS_SCALE': '#A3D9A5',  # 浅绿色
        'BF16-AMP': '#F28C8C',  # 浅红色
        'FP16': '#B59DD7',  # 浅紫色
        'BF16': '#D9A69A'  # 浅棕色
    }

    all_data = []
    for dir in exp_dir:
        df = pd.read_csv(f"{dir}/test_epoch.csv")
        if 'fp32' in dir:
            label = 'FP32'
        elif 'fp16-amp-lossscale' in dir:
            label = 'FP16-AMP-LOSS_SCALE'
        elif 'fp16-amp' in dir:
            label = 'FP16-AMP'
        elif 'bf16-amp' in dir:
            label = 'BF16-AMP'
        elif 'fp16' in dir:
            label = 'FP16'
        elif 'bf16' in dir:
            label = 'BF16'
        plt.plot(df['epoch'], df['test_accuracy'], label=label, color=color_dict[label])

        all_data.append((df['epoch'], df['test_accuracy'], label))

    plt.title(f'Test Accuracy vs Epoch for Different Precision Models(epoch: {args.epochs}, batch-size:{args.batch_size})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 获取最后的几个 epoch 数据来做放大镜
    zoom_start = 180  # 假设最后 20 个 epoch 需要放大
    zoom_end = 200    # 调整这个范围来控制放大区域

    # 创建 inset_axes 用于显示放大镜区域
    axins = inset_axes(plt.gca(), width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.65, 0.4, 1, 1), bbox_transform=plt.gca().transAxes, borderpad=2)

    # 在放大镜区域绘制最后的 epoch 数据
    for epoch_data, accuracy_data, label in all_data:
        axins.plot(epoch_data[zoom_start:zoom_end], accuracy_data[zoom_start:zoom_end], label=label, color=color_dict[label])

    # 放大镜标记
    mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")  # loc1和loc2指定放大区域的边框位置

    # 放大镜图例和标注
    axins.set_title('Zoomed In Area')
    axins.set_xlabel('Epoch')
    axins.set_ylabel('Test Accuracy')

    plt.savefig(f'{save_dir}/{prefix}test_acc.png')

    
def plot_train_loss_acc(exp_dir, save_dir):
    global prefix
    plt.figure(figsize=(10, 6))
    color_dict = {
        'FP32': '#7EC8E3',  # 浅蓝色
        'FP16-AMP': '#FFB37A',  # 浅橙色
        'FP16-AMP-LOSS_SCALE': '#A3D9A5',  # 浅绿色
        'BF16-AMP': '#F28C8C',  # 浅红色
        'FP16': '#B59DD7',  # 浅紫色
        'BF16': '#D9A69A'  # 浅棕色
    }

    all_data = []
    for dir in exp_dir:
        df = pd.read_csv(f"{dir}/train.csv")
        if 'fp32' in dir:
            label = 'FP32'
        elif 'fp16-amp-lossscale' in dir:
            label = 'FP16-AMP-LOSS_SCALE'
        elif 'fp16-amp' in dir:
            label = 'FP16-AMP'
        elif 'bf16-amp' in dir:
            label = 'BF16-AMP'
        elif 'fp16' in dir:
            label = 'FP16'
        elif 'bf16' in dir:
            label = 'BF16'
        df['step'] = range(len(df))
        plt.plot(df['step'], df['train_loss'], label=label, color=color_dict[label])

        all_data.append((df['step'], df['train_loss'], label))

    plt.title(f'Train Loss vs Step for Different Precision Models(epoch: {args.epochs}, batch-size:{args.batch_size})')
    plt.xlabel('Step')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid(True)

    # 获取最后的几个 epoch 数据来做放大镜
    zoom_start = len(df)-20  # 假设最后 20 个 epoch 需要放大
    zoom_end = len(df)    # 调整这个范围来控制放大区域

    # 创建 inset_axes 用于显示放大镜区域
    axins = inset_axes(plt.gca(), width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.65, 0.4, 1, 1), bbox_transform=plt.gca().transAxes, borderpad=2)

    # 在放大镜区域绘制最后的 epoch 数据
    for epoch_data, accuracy_data, label in all_data:
        axins.plot(epoch_data[zoom_start:zoom_end], accuracy_data[zoom_start:zoom_end], label=label, color=color_dict[label])

    # 放大镜标记
    mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")  # loc1和loc2指定放大区域的边框位置

    # 放大镜图例和标注
    axins.set_title('Zoomed In Area')
    axins.set_xlabel('Step')
    axins.set_ylabel('Train Loss')

    plt.savefig(f'{save_dir}/{prefix}train_loss.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 VISUALIZE')
    parser.add_argument('--exp-dir', type=str, default='../results')
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model', type=str, default='VGG19')
    parser.add_argument('--dataset', type=str, default='Cifar10')
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    suffix_list = ["fp32", "fp16", "bf16", "fp16-amp", "bf16-amp", "fp16-amp-lossscale"]
    prefix = f'{args.dataset}-{args.model}-e{args.epochs}-b{args.batch_size}-'
    exp_list = [f'{args.exp_dir}/{prefix+suffix_list[i]}' for i in range(len(suffix_list))]

    plot_test_acc(exp_list, args.save_dir)
    plot_train_loss_acc(exp_list, args.save_dir)

