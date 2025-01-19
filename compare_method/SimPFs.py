import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair
from thop.profile import profile
from torchlibrosa.augmentation import SpecAugmentation
from config import get_args_parser
 # Setting the config for each stage
parser = get_args_parser()
args = parser.parse_args()

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
        elif pool_type == 'None':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x

class Audio_Frontend(nn.Module):
    """
    Wav2Mel transformation & Mel Sampling frontend
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, sampler=None):
        super(Audio_Frontend, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.sampler = sampler

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        r"""Calculate (log) mel spectrogram from spectrogram.

             Args:
                 input: (*, n_fft), spectrogram

             Returns: 
                 output: (*, mel_bins), (log) mel spectrogram
             """
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        init_bn(self.bn0)

    def forward(self, inputs):
        """
        Input: (batch_size, data_length)
        """
        inputs = inputs.squeeze(1)
        x = self.spectrogram_extractor(inputs)  # (batch_size, 1, time_steps, freq_bins)
        # print("self.spectrogram_extractor(X)",x.shape) #torch.Size([8, 1, 32, 513])
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) #torch.Size([8, 1, 32, 64])
        # print("self.logmel_extractor(X)", x.shape)
        x = x.transpose(1, 3) # torch.Size([8, 64, 32, 1])
        x = self.bn0(x)
        x = x.transpose(1, 3) # torch.Size([8, 1, 32, 64])
        # 谱图增强方法
        # if self.training:
            # x = self.spec_augmenter(x)
        if self.sampler is not None:
            x = self.sampler(x)

        return x

class Cnn10(nn.Module):
    def __init__(self,num_classes):

        super(Cnn10, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, num_classes, bias=True)
        
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(512, 5)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(1,1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1,1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1,1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1,1))
        x = F.dropout(x, p=0.2, training=self.training)# [8,512,3,4]
        
        x = torch.mean(x, dim=3) # [8,512,3]
        (x1, _) = torch.max(x, dim=2) #[8,512]
        x2 = torch.mean(x, dim=2)#[8,512]
        x = x1 + x2 # [8,512]
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = self.fc_audioset(x)
        
        # x = self.avgpool(x)
        # x = self.flatten(x) 
        # x = self.fc1(x)
        
        return x

def _spectral_crop(input, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    top_combined = torch.cat((top_left, top_right), dim=-1)
    bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
    all_together = torch.cat((top_combined, bottom_combined), dim=-2)

    return all_together

def _spectral_pad(input, output, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    pad = torch.zeros_like(input)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):] = output[:, :, -(cutoff_freq_h - 1):,
                                                                      -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    return pad

def DiscreteHartleyTransform(input):
    # fft = torch.rfft(input, 2, normalized=True, onesided=False)
    # for new version of pytorch
    fft = torch.fft.fft2(input, dim=(-2, -1), norm='ortho')
    fft = torch.stack((fft.real, fft.imag), -1)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht

class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, oheight, owidth):
        ctx.oh = oheight
        ctx.ow = owidth
        ctx.save_for_backward(input)

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(input)

        # frequency cropping
        all_together = _spectral_crop(dht, oheight, owidth)
        # inverse Hartley transform
        dht = DiscreteHartleyTransform(all_together)
        return dht

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(grad_output)
        # frequency padding
        grad_input = _spectral_pad(input, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = DiscreteHartleyTransform(grad_input)
        return grad_input, None, None

class SpectralPool2d(nn.Module):
    def __init__(self, scale_factor):
        super(SpectralPool2d, self).__init__()
        self.scale_factor = _pair(scale_factor)

    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        # print(self.scale_factor[0],self.scale_factor[1]) # 0.25 1
        h, w = math.ceil(H * self.scale_factor[0]), math.ceil(W * self.scale_factor[1])
        # print(h,w) # 8 64
        return SpectralPoolingFunction.apply(input, h, w)

class Audio_Encoder(nn.Module):
    def __init__(self, frontend, backbone):
        super(Audio_Encoder, self).__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        return self.backbone(self.frontend(input))

class Pooling_layer(nn.Module):
    def __init__(self, factor=0.75):
        super(Pooling_layer, self).__init__()
        self.factor = factor
        self.SpecPool2d = SpectralPool2d(scale_factor=(factor, 1))

    def forward(self, x):
        """
        args:
            x: input mel spectrogram [batch, 1, time, frequency]
        return:
            out: reduced features [batch, 1, factor*time, frequency]
        """
        out = self.SpecPool2d(x)
        return out

class MIx_Pooling_layer(nn.Module):
    def __init__(self):
        super(MIx_Pooling_layer, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(2,2))
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        out = x1 + x2
        return out

class Max_Pooling_layer(nn.Module):
    def __init__(self):
        super(Max_Pooling_layer, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2))#scale_factor=(factor, 1)

    def forward(self, x):
        out = self.max_pool(x)
        return out

class Avg_Pooling_layer(nn.Module):
    def __init__(self):
        super(Avg_Pooling_layer, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.avg_pool(x)
        return out

def SimPFs_model(num_classes=5):
    panns_params = {
        'sample_rate': 16000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 8000}

    pool_type = args.pool_type
    if pool_type == 'max':
        sampler = Max_Pooling_layer()
    elif pool_type == 'avg':
        sampler = Avg_Pooling_layer()
    elif pool_type == 'avg+max':
        sampler = MIx_Pooling_layer()
    elif pool_type == "SpectralPool2d":
        sampler = Pooling_layer(factor=0.5)
    frontend = Audio_Frontend(**panns_params, sampler=sampler)
    backbone = Cnn10(num_classes)
    model = Audio_Encoder(frontend=frontend, backbone=backbone)
    return model


if __name__ == '__main__':
    panns_params = {
        'sample_rate': 16000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 8000}

    pool_type = 'SpectralPool2d'
    if pool_type == 'max':
        sampler = Max_Pooling_layer()
    elif pool_type == 'avg':
        sampler = Avg_Pooling_layer()
    elif pool_type == 'avg+max':
        x1 = MIx_Pooling_layer()
    elif pool_type == "SpectralPool2d":
        sampler = Pooling_layer(factor=0.5)

    frontend = Audio_Frontend(**panns_params, sampler=sampler)
    backbone = Cnn10(num_classes=5)

    model = Audio_Encoder(frontend=frontend, backbone=backbone)

    inputs = torch.randn(1, 1, 16000)
    output = model(inputs)
    print(output.shape)
    
    flops, params = profile(model, inputs=(inputs,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))