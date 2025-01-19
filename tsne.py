import torch
from random import random
import torch.nn as nn
import matplotlib.pyplot as plt
import os, time, argparse,glob
import numpy as np
from util.data_loader import get_data_loaders
from sklearn.manifold import TSNE

# 音频模型
from model_method.deepConv import M18
from model_method.TSNA import TSLANet
from model_method.waveCnn_recog import Res1dNet31
from model_method.WaveMsNet import WaveMsNet
from model_method.wave_transformer import restv2_tiny
# 谱图模型
from model_method.SimPFs import SimPFs_model
from model_method.wave_VIT import wavevit_s
from model_method.mpvit import mpvit_tiny
from model_method.Res2net import res2net18
from model_method.spectrogram_recog import ResNet38
# 音频+谱图模型
from model_method.wave_mel import Wavegram_Logmel
# design model
from SinConv.audio_model import audio_classification

import torch.optim as optim
from config import get_args_parser

import warnings
warnings.filterwarnings("ignore")

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 初始网络模型
def return_model(args):
    
    #音频模型
    if args.model_name == 'M18':
        model = M18(num_classes=args.num_classes).to(device)
    elif args.model_name == 'TSNA':
        model = TSLANet(num_classes=args.num_classes).to(device)
    elif args.model_name == 'Res1d':
        model = Res1dNet31(num_classes=args.num_classes).to(device)
    if args.model_name == 'WaveMsNet':
        model = WaveMsNet(num_classes=args.num_classes).to(device)
    elif args.model_name == 'WTS':
        model = restv2_tiny(num_classes=args.num_classes).to(device)
    
    #谱图模型
    elif args.model_name == 'SimPFs':
        model = SimPFs_model(num_classes=args.num_classes).to(device)
        
    elif args.model_name == 'wavevit':
        model = wavevit_s(num_classes=args.num_classes).to(device)
    elif args.model_name == 'mpvit':
        model = mpvit_tiny(num_classes=args.num_classes).to(device)
    elif args.model_name == 'Res2net':
        model = res2net18(num_classes=args.num_classes).to(device)
    elif args.model_name == 'Res2d':
        model = ResNet38(sample_rate=args.sample_rate, window_size=args.window_size,
                         hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                         classes_num=args.num_classes).to(device)
    # 时域+谱图
    elif args.model_name == 'Wavegram_Logmel':
        model = Wavegram_Logmel(sample_rate=args.sample_rate, window_size=args.window_size,
                                hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                                classes_num=args.num_classes).to(device)
    # design mpodel
    
    elif args.model_name == 'sinc':
        model = audio_classification(dim=64,L = 16000,classes_num=args.num_classes).to(device)
        
    return model

# 初始化数据集
def return_data(args):
    if args.no_noise:
        if args.data_type == 'ESC':
            return r'ori_dataSet/Cut_ESC'
        elif args.data_type == 'Cut_ShipEar':
            return r"ori_dataSet/Cut_ShipEar"
        elif args.data_type == 'Deepship':
            return r"ori_dataSet/Cut_deepShip"
        elif args.data_type == 'Whale':
            return r"ori_dataSet/Cut_whale"
        else:
            return None
    else:
        if args.data_type == 'ESC':
            return f'dataSet/{args.noise_path}/ESC'
        elif args.data_type == 'Cut_ShipEar':
            return f"dataSet/{args.noise_path}/Cut_ShipEar"
        elif args.data_type == 'Cut_deepShip':
            return f"dataSet/{args.noise_path}/Cut_deepShip"
        elif args.data_type == 'Cut_whale':
            return f"dataSet/{args.noise_path}/Cut_whale"
        else:
            return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()
    # -------------------------加载模型------------------------#
    model = return_model(args)
    model.load_state_dict(torch.load(
        f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}/{args.data_type}_{args.model_name}.pkl'))
    
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()

    data_set = return_data(args)

    train_loader, test_loader = get_data_loaders(data_set,args.batch_size, num_workers=8)
    # 假设 model 是你训练好的模型，test_data_loader 是待测试的数据集的 DataLoader

    model.eval()  # 切换到评估模式

    features = []
    labels = []

    with torch.no_grad():  # 不需要梯度计算
        for inputs, label in test_loader:
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            # labels = torch.LongTensor(labels.numpy()).to(device)
            output = model(inputs)  # 获取模型的输出或特征
            features.append(output.cpu().numpy())  # 将特征从 GPU 移到 CPU 并转为 NumPy
            labels.append(label.cpu().numpy())

    features = np.concatenate(features)  # 合并所有批次的特征
    labels = np.concatenate(labels)  # 合并所有批次的标签

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)  # n_components=2 表示降维到2D
    reduced_features = tsne.fit_transform(features)

    # 绘制 t-SNE 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title(f't-SNE Visualization of Test Data Features under {args.noise_level}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.show()
    plt.savefig(f'{args.noise_level}/{args.data_type}/{args.model_name}/tsne_visualize.png')
