import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F

# 低截止频率和高截止频率是从数据中学习到的滤波器的唯一参数。
# 在设计滤波器时，我们通常需要确定滤波器的两个关键参数：低截止频率（表示滤波器允许通过的最低频率）和高截止频率（表示滤波器允许通过的最高频率）。
# 而在传统的滤波器设计中，这些参数通常是人为设定的，但在SincConv卷积层中，这些参数是通过模型训练过程从数据中自动学习的。

#SincConv 卷积层设计的是一种特殊的带通滤波器，这种滤波器的低频边界（low_hz_）和频带宽度（band_hz_）作为可学习的参数（nn.Parameter）被初始化。
# 通过模型训练，这两个参数会根据输入数据的特性进行更新和优化，从而自动确定每个滤波器的低截止频率和高截止频率。
# 这意味着滤波器会根据输入信号的特征自动调整，达到自适应设计的效果。


# 代码中的主要关联性如下：
#
# 低截止频率 (low_hz_)：在初始化时，low_hz_ 参数表示每个滤波器的低截止频率。这个参数是通过等间隔划分 Mel 频率范围得到的初始值。
# 在前向传播（forward）中，low 是通过 self.min_low_hz + torch.abs(self.low_hz_) 计算得到的，
# 确保低频不会低于 self.min_low_hz，这也意味着该参数会在训练过程中通过反向传播不断优化，从数据中学习到合适的低截止频率。

# 高截止频率 (high)：高截止频率是通过将低截止频率加上可学习的频带宽度 (self.band_hz_) 来计算的：
# high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)。
# band_hz_ 代表了每个滤波器的带宽，它也是一个可学习的参数。在训练过程中，这个带宽值会不断调整，从而确定滤波器的高截止频率。
# 进一步解释：
# 学习到的唯一参数：在 SincConv 中，滤波器的唯一需要学习的参数就是低截止频率和频带宽度。
# 与传统的卷积核不同，这种设计不需要学习整个滤波器的权重，而是通过数学公式（如 sin 函数和时间域的频率响应）来生成滤波器的脉冲响应。
# 在代码的 forward 方法中，f_times_t_low 和 f_times_t_high 通过计算低、高截止频率对应的时间域频率响应构建滤波器，最后通过卷积操作对输入信号进行处理。
# 这些滤波器是根据学习到的低截止频率和高截止频率自适应设计的。
# 因此，代码中低截止频率 (low_hz_) 和高截止频率 (high) 的学习过程，就是从输入数据中自动调整滤波器频率响应的核心逻辑。


class SincConv(nn.Module):

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

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
        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        # 将过滤器的低频边界值（即每个带通滤波器的低频部分）转换为一个可学习的参数。hz[:-1]提取的是计算得到的赫兹频率范围的低边界。
        # 这个参数是通过等间隔划分 Mel 频率范围得到的初始值。
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        print("self.low_hz_",self.low_hz_)
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
        self.n_ = self.n_.to(waveforms.device) # 生成一个用于滤波的频率向量
        # print("self.n_.shape",self.n_.shape) torch.Size([1, 12])
        # 窗函数  # torch.Size([1, 12])
        self.window_ = self.window_.to(waveforms.device)
        # 低频（low）和高频（high）的定义是为了确保带通滤波器的设计遵循特定的约束
        # self.min_low_hz 是一个最小低频值，确保 low 不会低于这个值。这样做的目的是避免滤波器工作在不合适的频率范围。
        low = self.min_low_hz + torch.abs(self.low_hz_)
        print(low)
        # self.band_hz_ 同样是可学习的参数，表示每个带通滤波器的带宽。将带宽加到 low 上，得到 high。
        # 使用 torch.clamp 确保 high 不超过采样频率的一半（self.sample_rate / 2），以遵循奈奎斯特定理，避免混叠。
        # 这样的定义确保了每个带通滤波器的频率范围合理，同时能通过训练过程动态调整，以适应输入信号的特性。
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        print(high)
        band = (high - low)[:, 0]
        print(band)

        #这两个结果用于后续的带通滤波器设计，帮助构建滤波器的脉冲响应

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

if __name__ == "__main__":
    # path = "../../dataSet/1-11687-A-47_1.wav"
    # data,fs = librosa.load(path,sr=None)
    # data = torch.from_numpy(data)
    # print(data.shape)
    data = torch.randn(1,1,400)
    model = SincConv(3,25,400,1,1) # 80 251 16000
    output = model(data)
    print(output.shape)