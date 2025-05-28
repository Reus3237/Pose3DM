import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math


from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2,  # 通道翻倍
                      kernel_size=1,        # 关键修改：1x1卷积
                      stride=1,
                      padding=0,            # 必须设为0以保持H/W
                      bias=False),          # 与原代码一致
            nn.BatchNorm2d(n_feat * 2),     # 建议添加归一化
            nn.ReLU(inplace=True)           # 建议添加激活函数
        )

    def forward(self, x):
        return self.body(x)  # 输入输出尺寸: [B, C, H, W] -> [B, 2C, H, W]

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            # 关键修改1：移除上采样层
            nn.Conv2d(n_feat, n_feat // 2,   # 通道数减半
                     kernel_size=1,           # 关键修改2：1x1卷积
                     stride=1,
                     padding=0,              # 关键修改3：必须设为0
                     bias=False),
            nn.BatchNorm2d(n_feat // 2),     # 建议添加
            nn.ReLU(inplace=True)            # 建议添加
        )

    def forward(self, x):
        return self.body(x)
import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys
# 获取当前工作目录
current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    with torch.no_grad():
        # Generate a truncated normal distribution.
        return tensor.normal_(mean, std).clamp_(a, b)

from timm.models.layers import DropPath
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks import BiSTSSMBlock
class  Poseblock(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=48, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth
        self.STEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim_ratio,
                mlp_ratio = mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs'
                )
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim,
                mlp_ratio = mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs'
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def STE_forward(self, x):   #首先通过空间变换器块处理关节的空间信息。
        b, c, f, n = x.shape
        x = rearrange(x, 'b c f n -> b f n c')
        x = rearrange(x, 'b f n c  -> (b f) n c', )

        x = self.Spatial_patch_to_embedding(x)

        x += self.Spatial_pos_embed  # 是一个位置编码，它将关节的位置编码加到输入数据上。位置编码的作用是为每个关节提供位置信息
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)

        return x

    def TTE_foward(self, x):   #接着通过时间变换器块处理关节在时间序列中的变化。
        # assert len(x.shape) == 3, "shape is equal to 3"
        b, f, n, c  = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:,:f,:]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = rearrange(x, 'b f n c -> b c f n')
        return x

##########################################################################

class Pose3DM(nn.Module):
    def __init__(self,
                 inp_channels=2,
                 out_channels=3,
                 dim=16,
                 num_blocks=[1, 1, 1],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=10, mlp_ratio=2., drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=None,
                 ):
        super(Pose3DM, self).__init__()

        self.encoder = True

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = Poseblock(num_frame=num_frame, num_joints=num_joints, in_chans=16,
                                    embed_dim_ratio=16, depth=depth, mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                              )

            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = Poseblock(num_frame=num_frame, num_joints=num_joints, in_chans=32,
                                    embed_dim_ratio=32, depth=depth, mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                              )

            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = Poseblock(num_frame=num_frame, num_joints=num_joints, in_chans=64,
                                    embed_dim_ratio=128, depth=depth, mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                              )

            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = Poseblock(num_frame=num_frame, num_joints=num_joints, in_chans=128,
                                    embed_dim_ratio=64, depth=depth, mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                              )

            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = Poseblock(num_frame=num_frame, num_joints=num_joints, in_chans=32,
                                    embed_dim_ratio=32, depth=depth, mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                              )

            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = Poseblock(num_frame=num_frame, num_joints=num_joints, in_chans=16,
                                    embed_dim_ratio=16, depth=depth, mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                              )

            self.decoder_level1.add_module(f"block{i}", block)
        out_dim = 3
        embed_dim = 16
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
    def forward(self, inp_img, img1=None):
        b, f, n, c = inp_img.shape
        inp_img1 = rearrange(inp_img, 'b f j c -> b c f j')
        inp_enc_level1 = self.patch_embed(inp_img1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_dec_level3 = self.decoder_level3(out_enc_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = rearrange(out_dec_level1, 'b c f j -> b f j c')
        x = self.head(out_dec_level1)
        x = x.view(b, f, n, -1)

        return x



if __name__ == "__main__":
    torch.cuda.set_device(0)
    model = Pose3DM(num_frame=243, mlp_ratio = 2, depth = 6).cuda()
    from thop import profile, clever_format
    input_shape = (1, 243, 17, 2)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))