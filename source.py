import torch
from random import random
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os, time, argparse, glob, copy
import numpy as np
from util.data_loader import get_data_loaders

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


def train(model, optimizer, loss_fn, train_loader, n_epoch, train_path):
    start = time.time()
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    batch_num = len(train_loader)
    print("总的batch数量", batch_num)
    train_batch_num = round(batch_num * 0.8)
    print("训练使用的batch数量", train_batch_num)
    # 复制模型的参数
    best_acc = 0.0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    
    for epoch in range(1, n_epoch + 1):
        # exp_lr_scheduler.step()
      
        running_loss = 0
        running_correct = 0

        train_num = 0
        val_loss = 0
        val_corrects = 0
        val_num = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels.numpy()).to(device)
            if i < train_batch_num:
                model.train()
                output = model(inputs).to(device)
                loss = loss_fn(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, pred = torch.max(output.data, 1)  # get the index of the max log-probability
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(pred == labels)
                train_num += inputs.size(0)

            else:
                model.eval()
                output = model(inputs).to(device)
                loss = loss_fn(output, labels)
                _, pred = torch.max(output.data, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(pred == labels)
                val_num += inputs.size(0)
        
        train_loss = running_loss / train_num
        train_acc = 100.0 * running_correct.double().item() / train_num
        
        val_loss = val_loss / val_num
        val_acc = 100.0 * val_corrects.double().item() / val_num

        elapse = time.time() - start

        log_message = (f'Epoch: {epoch}/{n_epoch} lr: {optimizer.param_groups[0]["lr"]:.4g} '
                       f'samples: {len(train_loader.dataset)} TrainLoss: {train_loss:.3f} TrainAcc: {train_acc:.2f}% '
                       f'ValLoss: {val_loss:.3f} ValAcc: {val_acc:.2f}%')

        print(log_message)

        with open(train_path, "a") as file:
            file.write(log_message + "\n")

        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        ##拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            # -------------------------保存在测试集上的最佳模型------------------------#
            save_path = f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}'
            if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f'{save_path}/{args.data_type}_{args.model_name}.pkl')
    # print(f"model train time:{elapse:.1f}s")
    with open(train_path, "a") as file:
            file.write(f"model train time:{elapse:.1f}s" + "\n")
    ##使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch": range(n_epoch),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        }
    )
    return model, train_process


def main_fold(model_name, model, train_loader, train_path, args):
    print('------------------------- Train Start --------------------------------')
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3,
                               weight_decay=0.0005)  # by default, l2 regularization is implemented in the weight decay.
    
    exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
    
    model, train_process = train(model, optimizer, loss_fn, train_loader, args.n_epoch, train_path)

    return model, train_process


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
        model = wavevit_s().to(device)
    elif args.model_name == 'mpvit':
        model = mpvit_tiny().to(device)
    elif args.model_name == 'Res2net':
        model = res2net18(num_classes=args.num_classes).to(device)
    elif args.model_name == 'Res2d':
        model = ResNet38(sample_rate=args.sample_rate, window_size=args.window_size,
                         hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                         num_classes=args.num_classes).to(device)
    # 时域+谱图
    elif args.model_name == 'Wave_mel':
        model = Wavegram_Logmel(sample_rate=args.sample_rate, window_size=args.window_size,
                                hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                                num_classes=args.num_classes).to(device)
    # design mpodel
    
    elif args.model_name == 'sinc':
        model = audio_classification(dim=64,L = 16000,num_classes=args.num_classes).to(device)

    return model


# 初始化数据集
def return_data(args):
    if args.no_noise:
        if args.data_type == 'ESC':
            return r'ori_dataSet/Cut_ESC'
        elif args.data_type == 'Cut_ShipEar':
            return r"ori_dataSet/Cut_ShipEar"
        elif args.data_type == 'Cut_deepShip':
            return r"ori_dataSet/Cut_deepShip"
        elif args.data_type == 'Cut_whale':
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

    # Setting random seed
    random_name = str(random())
    random_seed = 123
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()

    data_set = return_data(args)
    train_loader, test_loader = get_data_loaders(data_set, args.batch_size, train_ratio=0.9, random_seed=random_seed, num_workers=8)

    print(f"----------------------------- Load data: {args.data_type} -----------------------------")

    # 当shuffle为True时, random_state影响标签的顺序。设置random_state=整数，可以保持数据集划分的方式每次都不变，便于不同模型的比较

    model = return_model(args)  ## Reinitialize model for each fold

    if not os.path.exists('{}/{}/{}'.format(args.noise_level, args.data_type, args.model_name)): os.makedirs(
        '{}/{}/{}'.format(args.noise_level, args.data_type, args.model_name))
    train_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/loss_accuracy.txt"

    # 在这里可以创建 DataLoader 或者进行模型训练

    print(f"Train: {len(train_loader.dataset)} samples")
    
    # 模型训练
    model_ft, train_process = main_fold(args.model_name, model, train_loader, train_path, args)

    ##可视化模型训练过程
    plt.figure(figsize=(12, 4))
    ##损失函数
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss", markersize=5)
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss", markersize=5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    ##精度
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc", markersize=5)
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc", markersize=5)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    # 保存训练结果
    PATH_fig = os.path.join(f"{args.noise_level}/{args.data_type}/{args.model_name}" + '.pdf')
    plt.savefig(PATH_fig)

