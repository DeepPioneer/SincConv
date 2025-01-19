import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from timm.models.layers import DropPath, trunc_normal_
import os, math
import numpy as np
import torchinfo
from thop.profile import profile

from config import get_args_parser

parser = get_args_parser()
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)

class SincConv(nn.Module):

    @staticmethod
    def adaptive_to_mel(frequencies, beta=1.5):

        return 2595 * np.log10(1 + beta * (frequencies / 700))
        # return 2595 * np.log10(1 + np.sqrt(beta) * (frequencies / 700))

    @staticmethod
    def adaptive_to_hz(mel_frequencies, beta):

        frequencies = (700 / beta) * (10 ** (mel_frequencies / 2595) - 1)
        # frequencies = (700 / np.sqrt(beta)) * (10 ** (mel_frequencies / 2595) - 1)
        return frequencies

    #     def hz_to_imel(hz):
    #         return 2195.286 - 2595 * np.log10(1 + (8031.25 - hz) / 700)

    #     @staticmethod
    #     def imel_to_hz(imel):
    #         return 8031.25 - 700 * (10 ** ((2195.286 - imel) / 2595) - 1)
    @staticmethod
    def hz_to_imel(hz):
        return 2195.286 - 1927 * np.log10(1 + (8031.25 - hz) / 700)

    @staticmethod
    def imel_to_hz(imel):
        return 8031.25 - 700 * (10 ** ((2195.286 - imel) / 1927) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        # 在生成Mel频率的过程中，使用self.out_channels + 1是因为np.linspace函数会生成out_channels + 1个点，
        # 这样可以得到out_channels个频带边界（即每个带通滤波器的低频和高频边界）。
        # 如果使用self.out_channels + 2，则会多出一个边界点，导致在分配频带时出现不必要的频带重叠或间隙。
        # 因此，self.out_channels + 1的选择是为了确保每个滤波器都有一个明确的频带定义，而不产生冗余的边界。
        # mel = np.linspace(self.to_mel(low_hz),
        #                   self.to_mel(high_hz),
        #                   self.out_channels + 1)
        # hz = self.to_hz(mel)
        # 定义分界频率为 sr/4
        f_split = self.sample_rate / 4 - (self.min_low_hz + self.min_band_hz)

        # Mel 滤波器的边界
        mel_min = self.adaptive_to_mel(low_hz, beta=1.5)
        mel_max = self.adaptive_to_mel(f_split, beta=1.5)

        # IMel 滤波器的边界
        imel_min = self.hz_to_imel(f_split)
        imel_max = self.hz_to_imel(high_hz)

        # 滤波器数量分配
        n_mels_half = self.out_channels // 2

        # 低频段的 Mel 三角形滤波器
        mel_points = np.linspace(mel_min, mel_max, n_mels_half + 1)
        hz_points_mel = self.adaptive_to_hz(mel_points, beta=1.5)

        # 高频段的 IMel 三角形滤波器
        imel_points = np.linspace(imel_min, imel_max, n_mels_half + 1)
        hz_points_imel = self.imel_to_hz(imel_points)

        # 合并两个频段的中心频率
        hz = np.concatenate((hz_points_mel, hz_points_imel[1:]))

        # print(hz)

        # filter lower frequency (out_channels, 1)
        # 将过滤器的低频边界值（即每个带通滤波器的低频部分）转换为一个可学习的参数。hz[:-1]提取的是计算得到的赫兹频率范围的低边界。
        # 这个参数是通过等间隔划分 Mel 频率范围得到的初始值。
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        # print("self.low_hz_",self.low_hz_)
        # filter frequency band (out_channels, 1)
        # 计算每个频带的带宽（高频边界减去低频边界），并将其也转换为一个可学习的参数。np.diff(hz)计算的是相邻频率之间的差值，从而得到带宽信息。
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        # n_ 是一个形状为 [1, kernel_size//2] 的张量，表示滤波器的频率响应，可以用于后续的信号处理
        self.n_ = self.n_.to(waveforms.device)  # 生成一个用于滤波的频率向量
        # print("self.n_.shape",self.n_.shape) torch.Size([1, 12])
        # 窗函数  # torch.Size([1, 12])
        self.window_ = self.window_.to(waveforms.device)
        # 低频（low）和高频（high）的定义是为了确保带通滤波器的设计遵循特定的约束
        # self.min_low_hz 是一个最小低频值，确保 low 不会低于这个值。这样做的目的是避免滤波器工作在不合适的频率范围。
        low = self.min_low_hz + torch.abs(self.low_hz_)
        # print(low)
        # self.band_hz_ 同样是可学习的参数，表示每个带通滤波器的带宽。将带宽加到 low 上，得到 high。
        # 使用 torch.clamp 确保 high 不超过采样频率的一半（self.sample_rate / 2），以遵循奈奎斯特定理，避免混叠。
        # 这样的定义确保了每个带通滤波器的频率范围合理，同时能通过训练过程动态调整，以适应输入信号的特性。
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        # print(high)
        band = (high - low)[:, 0]
        # print(band)

        # 这两个结果用于后续的带通滤波器设计，帮助构建滤波器的脉冲响应

        # 将每个滤波器的低频边界 low 与时间向量 self.n_ 相乘，得到低频部分的频率响应。这表示低频信号在给定时间上的相位变化。
        f_times_t_low = torch.matmul(low, self.n_)
        # 计算高频边界 high 在时间域上的频率响应，表示高频信号在同一时间上的相位变化。
        f_times_t_high = torch.matmul(high, self.n_)

        # 带通滤波器的脉冲响应通常分为三个部分：左侧部分、中心部分和右侧部分
        # 计算带通滤波器左侧部分的脉冲响应 计算高频和低频边界对应的正弦值的差。这表示了带通滤波器的频率响应
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        # 翻转得到另一侧
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        # 构建带通滤波器的完整脉冲响应并进行归一化
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        # 将构建的带通滤波器脉冲响应重塑为适合卷积操作的形状。
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)

class DynamicMaskingModule(nn.Module):
    def __init__(self, input_dim):
        super(DynamicMaskingModule, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, energy):
        feature = self.fc(energy)
        adaptive_mask = torch.sigmoid(feature)  # Soft mask based on learned features
        return adaptive_mask

class mul_block(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate, in_channels):
        super(mul_block, self).__init__()
        self.sin_conv1 = SincConv(out_channels, kernel_size, sample_rate, in_channels)
        # self.conv1 = nn.Conv2d(1, 64,kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)  # this is used to normalize
        self.pool1 = nn.MaxPool1d(4)

        # 合并第二层卷积的多个块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool1d(4)

        # 合并第三层卷积的多个块
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 51, padding=25),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 51, padding=25),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool1d(4)

        # 合并第四层卷积的多个块
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, 101, padding=50),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 101, padding=50),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.pool4 = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.sin_conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        return x

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, L):
        super().__init__()

        # 创建复权重参数，适应最后一维频率分量的变换
        self.complex_weight_high = nn.Parameter(
            torch.randn(L // 2 + 1, 2, dtype=torch.float32) * 0.02)  # 频域中的大小为 10//2 + 1
        self.complex_weight = nn.Parameter(torch.randn(L // 2 + 1, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)
        self.smooth_param = nn.Parameter(torch.rand(1))  # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        # Calculate energy in the frequency domain
        # 计算频域中的能量
        B, _, _ = x_fft.shape  # [1, 1, 6] (6 是傅里叶变换后频率分量)
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)  # 能量按频率维度求和

        # 计算能量的中位数
        median_energy = energy.median(dim=-1, keepdim=True)[0]  # 计算频域中每个音频段的中位能量

        # 归一化能量，避免除以 0
        epsilon = 1e-6  # 防止除以零的小常量
        normalized_energy = energy / (median_energy + epsilon)  # 归一化能量

        # Frequency domain attention mechanism
        # adaptive_mask = nn.Softmax(dim=1)(normalized_energy)  # Apply softmax to normalize attention weights

        # Sigmoid-based soft mask to avoid hard cutoff
        adaptive_mask = torch.sigmoid(
            (normalized_energy - self.threshold_param) * self.smooth_param)  # smooth_param controls smoothness

        # dynamic_masking = DynamicMaskingModule(input_dim=energy.size(1)).to(device)
        # adaptive_mask = dynamic_masking(normalized_energy)

        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        # print("x_in",x_in.shape) # torch.Size([32, 4000, 128])
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)  # weight torch.Size([128])
        x_weighted = x_fft * weight

        # Adaptive High Frequency Mask (no need for dimensional adjustments)
        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = x_fft * freq_mask.to(x.device)

        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high

        x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, dim=-1, norm='ortho')

        x = x.to(dtype)

        return x

class audio_classification(nn.Module):
    def __init__(self, dim=64, L=16000, num_classes=4):
        super().__init__()

        self.asb = Adaptive_Spectral_Block(L=L)
        self.sinconv = mul_block(out_channels=dim, kernel_size=251, sample_rate=args.sample_rate, in_channels=1)

        # self.avgPool = nn.AvgPool1d(25)      #replaced with ADaptive + flatten
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, num_classes)  # this is the output layer.

    def forward(self, x):
        x = self.asb(x)  # 32,1,16000
        x = self.sinconv(x)
        # Flatten the output for the fully connected layer
        x = self.flatten(self.avgPool(x))
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    inputs = torch.randn(1, 1, 16000)
    model = audio_classification(dim=64, L=inputs.shape[2])
    output = model(inputs)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    flops, params = profile(model, inputs=(inputs,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
