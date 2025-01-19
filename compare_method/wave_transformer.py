import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)
        # 用于将低分辨率图像转换为高分辨率图像
        self.up = nn.Sequential(
            nn.Conv1d(dim, sr_ratio * sr_ratio * dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.PixelShuffle(upscale_factor=sr_ratio)
        )
        self.up_norm = nn.LayerNorm(dim, eps=1e-6)

        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(0)

    def forward(self, x, L):
        B, N, C = x.shape
        # torch.Size([32, 4000, 32])
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, L)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.sr_norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print("k的形状",k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        self.apply_transform = False
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x的形状",x.shape) # [32,3136,96]
        identity = v.transpose(-1, -2).reshape(B, C, L // self.sr_ratio)
        # identity = self.up(identity).transpose(1, 2)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, L):
        # [32,3136,96] h=56,w=56
        x = x + self.drop_path(self.attn(self.norm1(x), L))  # pre_norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv1d(dim, dim, kernel_size=11, padding=11//2, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class ConvStem(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, patch_size=4, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv1d(in_dim, out_dim, kernel_size=11, stride=2, padding=11//2,bias=False))
            stem.append(nn.BatchNorm1d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv1d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)  # [32,32,4000]
        if self.with_pos:
            x = self.pos(x)  # [32,32,4000]
        x = x.transpose(1, 2)  # BCHW -> BNC [32,4000,32]
        x = self.norm(x)
        L = L // self.patch_size
        return x, L

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=11, stride=patch_size, padding=11 // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        # torch.Size([32, 64, 2000])
        B, C, L = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        L = L // self.patch_size
        return x, L

class ResTV2(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8], drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i])
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)  # final norm layer
        # classification head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):  # input x->[32,1,16000]
        B, _, L = x.shape
        # x->[32,32,4000]->[32,4000,32] L=4000
        x, L = self.stem(x)
        # stage 1
        for blk in self.stage1:
            x = blk(x, L)  # output x->[32,4000,32]
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        # stage 2
        x, L = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, L)  # # output x->[32,784,192]
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        # stage 3
        x, L = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x, L)  # [32,196,384]
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        # stage 4
        x, L = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x, L)  # [32,49,768]
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        x = self.avg_pool(x).flatten(1)
        x = self.head(x)
        return x

def restv2_tiny(**kwargs):  # 82.3|4.7G|24M -> |3.92G|30.37M   4.5G|30.33M
    model = ResTV2(embed_dims=[32, 64, 128, 256], depths=[1, 2, 2, 2], **kwargs)
    return model

from thop.profile import profile
if __name__ == "__main__":
    inputs = torch.randn(1, 1, 16000)
    model = restv2_tiny(num_classes=2)
    output = model(inputs)
    print(output.shape)  # [32,2]

     
    flops, params = profile(model, inputs=(inputs,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))