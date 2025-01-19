import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from config import get_args_parser
 # Setting the config for each stage
parser = get_args_parser()
args = parser.parse_args()

def do_mixup(x, mixup_lambda):
    out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] + \
           x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    return out

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class Wavegram_Logmel(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, num_classes):

        super(Wavegram_Logmel, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(args.mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # self.fc1 = nn.Linear(2048, 2048, bias=True)
        # self.fc_audioset = nn.Linear(2048, num_classes, bias=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 5)

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        # init_layer(self.fc_audioset)

    def forward(self, inputs, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram # [32,16000]
        inputs = inputs.squeeze(1)
        # a[:, None, :]: Adds an extra dimension, changing the shape of a from [32, 16000] to [32, 1, 16000].
        # # [32, 64, 3200]
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(inputs[:, None, :])))
        # torch.Size([32, 64, 800])
        a1 = self.pre_block1(a1, pool_size=4)
        # torch.Size([32, 128, 200])
        a1 = self.pre_block2(a1, pool_size=4)
        # torch.Size([32, 128, 50])
        a1 = self.pre_block3(a1, pool_size=4)
        # torch.Size([32, 4, 32, 50]) -> torch.Size([32, 4, 50, 32]) B C T F
        a1 = a1.reshape((a1.shape[0], -1, args.mel_bins//2, a1.shape[-1])).transpose(2, 3)
        # torch.Size([32, 64, 25, 32])
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        x = self.spectrogram_extractor(inputs)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        # # torch.Size([32, 1, 51, 64])
        x = x.transpose(1, 3)
       
        # torch.Size([32, 64, 25, 32])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        # torch.Size([32, 64, 25, 32])
        x = torch.cat((x, a1), dim=1)
        # print(x.shape)
        x = F.dropout(x, p=0.2, training=self.training)
        pool_input = (1,1)
        x = self.conv_block2(x, pool_size=pool_input, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=pool_input, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=pool_input, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=pool_input, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=pool_input, pool_type='avg')
        # torch.Size([32, 2048, 25, 32])
        x = F.dropout(x, p=0.2, training=self.training)
        # torch.Size([32, 2048, 25])
       
        # x = torch.mean(x, dim=3)
        # # x1 # torch.Size([32, 2048])
        # (x1, _) = torch.max(x, dim=2)
        # # x2 # torch.Size([32, 2048])
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        # x = torch.sigmoid(self.fc_audioset(x))

        x = self.avgpool(x)
        x = self.flatten(x) 
        x = self.fc1(x)

        return x

if __name__ == "__main__":
    data = torch.randn(32,1,16000)
    model = Wavegram_Logmel(sample_rate=16000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000,num_classes=2)
    output = model(data)
    print(output.shape)
