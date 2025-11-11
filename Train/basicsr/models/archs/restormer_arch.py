## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


    
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchTimePoseEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, out_dim=48, bias=False, base_size=(128, 128)):
        super(OverlapPatchTimePoseEmbed, self).__init__()
        
        self.base_size = base_size
        self.embed_dim = embed_dim
        
        # 使用正弦函数初始化位置编码
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(self.sinusoidal_position_encoding(base_size[0], base_size[1], embed_dim)) for _ in range(in_c)
        ])
        # 使用正弦函数初始化时间位置编码
        self.time_embeds = nn.ParameterList([
            nn.Parameter(self.sinusoidal_time_encoding(1+i, embed_dim)) for i in range(in_c)
        ])

        # 每个通道单独嵌入
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=5, stride=1, padding=2, bias=bias)
        
        # 将所有嵌入的通道合并并重建
        self.reconstruct = nn.Conv2d(in_c * embed_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def sinusoidal_position_encoding(self, H, W, embed_dim):
        """生成二维正弦位置编码"""
        position = torch.zeros(1, embed_dim, H, W)
        for i in range(embed_dim):
            div_term1 = 1000 ** (2 * (i // 2) / embed_dim)
            div_term2 = 10 ** (2 * (i // 2 + 1) / embed_dim)
            if i % 2 == 0:
                position[:, i, :, :] = (torch.sin(torch.linspace(0, H-1, H).unsqueeze(1) / div_term1) + torch.sin(torch.linspace(0, W-1, W).unsqueeze(0) / div_term2))*0.5
            else:
                position[:, i, :, :] = (torch.cos(torch.linspace(0, H-1, H).unsqueeze(1) / div_term2) + torch.cos(torch.linspace(0, W-1, W).unsqueeze(0) / div_term1))*0.5
        return position

    def sinusoidal_time_encoding(self, seq_len, embed_dim):
        """生成时间维度上的正弦位置编码"""
        time_encoding = torch.zeros(1, embed_dim)
        for i in range(embed_dim):
            div_term = 100 ** (2 * (i // 2) / embed_dim)
            if i % 2 == 0:
                time_encoding[0, i] = torch.sin(torch.tensor(seq_len - 1) / div_term)
            else:
                time_encoding[0, i] = torch.cos(torch.tensor(seq_len - 1) / div_term)
        return time_encoding
    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        embedded_channels = []

        pos_encodings = []  # 用于存储所有位置编码
        time_encodings = []  # 用于存储所有时间编码
        
        for idx in range(C):
            channel = x[:, idx:idx+1, :, :]  # 提取单个通道
            
            # 获取卷积后特征图的尺寸
            channel_embedded = self.proj(channel)
            H_emb, W_emb = channel_embedded.shape[2], channel_embedded.shape[3]
            
            # 动态调整位置编码的大小以适应卷积后的特征图
            pos_embed = F.interpolate(self.pos_embeds[idx], size=(H_emb, W_emb), mode='bilinear', align_corners=False)
            channel_embedded += pos_embed

            # 添加时间位置编码
            time_embed = self.time_embeds[idx]
            time_embed = time_embed.unsqueeze(-1).unsqueeze(-1).expand_as(channel_embedded)
            channel_embedded += time_embed
            
            
            embedded_channels.append(channel_embedded)
            # pos_encodings.append(pos_embed.detach().cpu().numpy())  # 保存位置编码
            # time_encodings.append(time_embed.detach().cpu().numpy())  # 保存时间编码
        # np.save('pos_encodings.npy', pos_encodings)
        # np.save('time_encodings.npy', time_encodings)
        # 将嵌入的通道拼接并重建
        x_embedded = torch.cat(embedded_channels, dim=1)
        x = self.reconstruct(x_embedded)
        
        return x

class BasicSpy(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(num_features=out_c),
            torch.nn.ReLU(inplace=False),
        )
        self.reconstruct = nn.Conv2d(in_channels=in_c+out_c, out_channels=out_c, kernel_size=1)
    def forward(self, x):
        x1 = self.netBasic(x)
        combined_x = torch.cat((x, x1), dim=1)
        x = self.reconstruct(combined_x)
        return x
    
class WDSpybottle(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(WDSpybottle,self).__init__()
        h_dim = int(in_dim)
        self.netBasic1 = BasicSpy(in_dim, h_dim)
        self.netBasic2 = BasicSpy(h_dim, h_dim)
        self.netBasic3 = BasicSpy(h_dim, out_dim)
    def forward(self, x):
        x = self.netBasic1(x)
        x = self.netBasic2(x)
        x = self.netBasic3(x)
        return x
    
class OverlapPatchTimePoseEmbedWD(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, out_dim=48, bias=False, base_size=(128, 128)):
        super(OverlapPatchTimePoseEmbedWD, self).__init__()
        
        self.base_size = base_size
        self.embed_dim = embed_dim
        

        # 使用正弦函数初始化位置编码
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(self.sinusoidal_position_encoding(base_size[0], base_size[1], embed_dim)) for _ in range(in_c)
        ])
        # 使用正弦函数初始化时间位置编码
        self.time_embeds = nn.ParameterList([
            nn.Parameter(self.sinusoidal_time_encoding(1+i, embed_dim)) for i in range(in_c)
        ])

        # 每个通道单独嵌入
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=5, stride=1, padding=2, bias=bias)
        
        # 将所有嵌入的通道合并并重建
        self.reconstruct = nn.Conv2d(in_channels=in_c * embed_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=bias)


    def sinusoidal_position_encoding(self, H, W, embed_dim):
        """生成二维正弦位置编码"""
        position = torch.zeros(1, embed_dim, H, W)
        for i in range(embed_dim):
            div_term = 100 ** (2 * (i // 2) / embed_dim)
            if i % 2 == 0:
                position[:, i, :, :] = torch.sin(torch.linspace(0, H-1, H).unsqueeze(1) / div_term) + torch.sin(torch.linspace(0, W-1, W).unsqueeze(0) / div_term)
            else:
                position[:, i, :, :] = torch.cos(torch.linspace(0, H-1, H).unsqueeze(1) / div_term) + torch.cos(torch.linspace(0, W-1, W).unsqueeze(0) / div_term)
        return position

    def sinusoidal_time_encoding(self, seq_len, embed_dim):
        """生成时间维度上的正弦位置编码"""
        time_encoding = torch.zeros(1, embed_dim)
        for i in range(embed_dim):
            div_term = 100 ** (2 * (i // 2) / embed_dim)
            if i % 2 == 0:
                time_encoding[0, i] = torch.sin(torch.tensor(seq_len - 1) / div_term)*2
            else:
                time_encoding[0, i] = torch.cos(torch.tensor(seq_len - 1) / div_term)*2
        return time_encoding
    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        embedded_channels = []

        pos_encodings = []  # 用于存储所有位置编码
        time_encodings = []  # 用于存储所有时间编码
        
        for idx in range(C):
            channel = x[:, idx:idx+1, :, :]  # 提取单个通道
            
            # 获取卷积后特征图的尺寸
            channel_embedded = self.proj(channel)
            H_emb, W_emb = channel_embedded.shape[2], channel_embedded.shape[3]
            
            # 动态调整位置编码的大小以适应卷积后的特征图
            pos_embed = F.interpolate(self.pos_embeds[idx], size=(H_emb, W_emb), mode='bilinear', align_corners=False)
            channel_embedded += pos_embed

            # 添加时间位置编码
            time_embed = self.time_embeds[idx]
            time_embed = time_embed.unsqueeze(-1).unsqueeze(-1).expand_as(channel_embedded)
            channel_embedded += time_embed
            
            
            embedded_channels.append(channel_embedded)
            # pos_encodings.append(pos_embed.detach().cpu().numpy())  # 保存位置编码
            # time_encodings.append(time_embed.detach().cpu().numpy())  # 保存时间编码
        # np.save('pos_encodings.npy', pos_encodings)
        # np.save('time_encodings.npy', time_encodings)
        # 将嵌入的通道拼接并重建
        x_embedded = torch.cat(embedded_channels, dim=1)

        x = self.reconstruct(x_embedded)

        return x
    
# class OverlapPatchTimePoseEmbed(nn.Module):
#     def __init__(self, in_c=3, embed_dim=48, bias=False, base_size=(16, 16)):
#         super(OverlapPatchTimePoseEmbed, self).__init__()
        
#         self.base_size = base_size
#         self.embed_dim = embed_dim
        
#         # 使用正弦函数初始化位置编码
#         self.pos_embeds = nn.ParameterList([
#             nn.Parameter(self.sinusoidal_position_encoding(base_size[0], base_size[1], embed_dim)) for _ in range(in_c)
#         ])
#         # # 使用正弦函数初始化时间位置编码
#         # self.time_embeds = nn.ParameterList([
#         #     nn.Parameter(self.sinusoidal_time_encoding(1, embed_dim)) for _ in range(in_c)
#         # ])
#         self.time_embeds = nn.Parameter(self.sinusoidal_time_encoding(in_c, in_c)) 
#         print(self.time_embeds.shape)
        
#         # 每个通道单独嵌入
#         self.proj = nn.Conv2d(1, embed_dim, kernel_size=5, stride=1, padding=2, bias=bias)
        
#         # 将所有嵌入的通道合并并重建
#         self.reconstruct = nn.Conv2d(in_c * embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

#     def sinusoidal_position_encoding(self, H, W, embed_dim):
#         """生成二维正弦位置编码"""
#         position = torch.zeros(1, embed_dim, H, W)
#         for i in range(embed_dim):
#             div_term = 10000 ** (2 * (i // 2) / embed_dim)
#             if i % 2 == 0:
#                 position[:, i, :, :] = torch.sin(torch.linspace(0, H-1, H).unsqueeze(1) / div_term) + torch.sin(torch.linspace(0, W-1, W).unsqueeze(0) / div_term)
#             else:
#                 position[:, i, :, :] = torch.cos(torch.linspace(0, H-1, H).unsqueeze(1) / div_term) + torch.cos(torch.linspace(0, W-1, W).unsqueeze(0) / div_term)
#         return position

    # def sinusoidal_time_encoding(self, seq_len, embed_dim):
    #     """生成时间维度上的正弦位置编码"""
    #     time_encoding = torch.zeros(1, embed_dim)
    #     for i in range(embed_dim):
    #         div_term = 10000 ** (2 * (i // 2) / embed_dim)
    #         if i % 2 == 0:
    #             time_encoding[0, i] = torch.sin(torch.tensor(seq_len - 1) / div_term)
    #         else:
    #             time_encoding[0, i] = torch.cos(torch.tensor(seq_len - 1) / div_term)
    #     return time_encoding
    
    # def forward(self, x):
    #     # print(x.shape)
    #     B, C, H, W = x.shape
    #     embedded_channels = []

    #     # pos_encodings = []  # 用于存储所有位置编码
    #     # time_encodings = []  # 用于存储所有时间编码
        
    #     for idx in range(C):
    #         channel = x[:, idx:idx+1, :, :]  # 提取单个通道
            
    #         channel_embedded = self.proj(channel)

    #         # 动态调整位置编码的大小以适应卷积后的特征图
    #         H_emb, W_emb = channel_embedded.shape[2], channel_embedded.shape[3]
    #         pos_embed = F.interpolate(self.pos_embeds[idx], size=(H_emb, W_emb), mode='bilinear', align_corners=False)
    #         channel_embedded += pos_embed

    #         # 添加时间编码
    #         time_embed = self.time_embeds[0][idx]
    #         channel_embedded += time_embed
            
            
    #         embedded_channels.append(channel_embedded)
    #         # pos_encodings.append(pos_embed.detach().cpu().numpy())  # 保存位置编码
    #         # time_encodings.append(time_embed.detach().cpu().numpy())  # 保存时间编码
    #     # np.save('pos_encodings.npy', pos_encodings)
    #     # np.save('time_encodings.npy', time_encodings)
    #     # 将嵌入的通道拼接并重建
    #     x_embedded = torch.cat(embedded_channels, dim=1)
    #     x = self.reconstruct(x_embedded)
        
    #     return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

##########################################################################
##---------- Restormer Super Resolution -----------------------
class RestormerSuperResolutionParam2(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        static = "train",
        params = 'cat'
    ):

        super(RestormerSuperResolutionParam2, self).__init__()
        self.params = params
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.patch_embed_param = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_param = nn.Conv2d(out_channels + 1, int(dim*2**1), kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)
        self.refinement_out = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output2 = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.static = static
        if self.static == "train":
            hc = dim*2**1
            self.cen = nn.Conv2d(int(out_channels), hc, kernel_size=3, stride=1, padding=1, bias=bias)
            self.upen = Upsample(int(hc))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
            self.enhance = nn.Sequential(*[TransformerBlock(dim=int(hc/2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            self.outputen = nn.Conv2d(int(hc/2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):

        inp_img = input["img"]
        denoise_rate = input["denoise_rate"]

        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            # print("inp_img shape:", inp_img.shape)
            # print("out_dec_level1 shape:", out_dec_level1.shape)
            # print("inp_img first 3 channels:", inp_img[:,0:3,:,:].shape)

            out_dec_level1 = self.output(out_dec_level1) 
            if self.params == 'cat':
                out_dec_level1 = torch.cat([out_dec_level1, denoise_rate], 1)
                out_dec_level1 = self.output_param(out_dec_level1)
                out_dec_level1 = self.refinement_out(out_dec_level1)
                out_dec_level1 = self.output2(out_dec_level1)
            
            out_hq = out_dec_level1 + inp_img
            

        if self.static == "train":
            # print()
            out_enhance = self.cen(out_hq)
            out_enhance = self.upen(out_enhance)
            out_enhance = self.enhance(out_enhance)
            out_enhance = self.outputen(out_enhance)

        else:
            out_enhance = None

        outdict = {"hq":out_hq, "sr":out_enhance}
        
        return outdict
    
#########################################################################
##---------- KDLAE_teacher -----------------------
class KDLAE_teacher(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        static = "train",
        params = 'cat'
    ):

        super(KDLAE_teacher, self).__init__()
        self.params = params
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.patch_embed_param = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_param = nn.Conv2d(out_channels + 1, int(dim*2**1), kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)
        self.refinement_out = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output2 = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.static = static
        if self.static == "train":
            hc = dim*2**1
            self.cen = nn.Conv2d(int(out_channels), hc, kernel_size=3, stride=1, padding=1, bias=bias)
            self.upen = Upsample(int(hc))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
            self.enhance = nn.Sequential(*[TransformerBlock(dim=int(hc/2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            self.outputen = nn.Conv2d(int(hc/2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):

        inp_img = input["img"]
        denoise_rate = input["denoise_rate"]

        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            # print("inp_img shape:", inp_img.shape)
            # print("out_dec_level1 shape:", out_dec_level1.shape)
            # print("inp_img first 3 channels:", inp_img[:,0:3,:,:].shape)

            out_dec_level1 = self.output(out_dec_level1) 
            if self.params == 'cat':
                out_dec_level1 = torch.cat([out_dec_level1, denoise_rate], 1)
                out_dec_level1 = self.output_param(out_dec_level1)
                out_dec_level1 = self.refinement_out(out_dec_level1)
                out_dec_level1 = self.output2(out_dec_level1)
            
            out_hq = out_dec_level1 + inp_img
            

        if self.static == "train":
            # print()
            out_enhance = self.cen(out_hq)
            out_enhance = self.upen(out_enhance)
            out_enhance = self.enhance(out_enhance)
            out_enhance = self.outputen(out_enhance)

        else:
            out_enhance = None

        outdict = {"hq":out_hq, "sr":out_enhance}
        
        return outdict

##########################################################################
##---------- KDLAE_student -----------------------
class KDLAE_student(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, residual=False, 
                 hidden_channels=[16, 32, 64], kernel_size=3):
        """
        多帧去噪模型 - 使用时空特征融合层替代瓶颈层
        
        参数:
            inp_channels: 输入通道数
            out_channels: 输出通道数
            residual: 是否使用残差连接
            hidden_channels: 隐藏层通道数列表
            kernel_size: 卷积核大小
            num_levels: 编码/解码层级数
        """
        super(KDLAE_student, self).__init__()
        self.residual = residual
        self.num_levels = len(hidden_channels) -1
        padding = kernel_size // 2
        
        self.encoders = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        
        in_channels = inp_channels
        for i in range(self.num_levels):
            out_channels_enc = hidden_channels[i]
            self.encoders.append(self._create_conv_block(in_channels, out_channels_enc, kernel_size, padding))
            self.pooling_layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2)))
            in_channels = out_channels_enc
        
        fusion_channels = hidden_channels[-1]
        self.st_fusion = self._create_conv_block(in_channels, fusion_channels, kernel_size, padding)
        
        self.upconv_layers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(self.num_levels-1, -1, -1):
            in_channels_up = hidden_channels[-1] if i == self.num_levels-1 else hidden_channels[i+1]
            out_channels_up = hidden_channels[i]
            self.upconv_layers.append(nn.ConvTranspose3d(in_channels_up, out_channels_up, 
                                                         kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            
            in_channels_dec = hidden_channels[i]
            self.decoders.append(self._create_conv_block(in_channels_dec, hidden_channels[i], kernel_size, padding))
        
        self.out_conv = nn.Conv3d(hidden_channels[0], out_channels, kernel_size=(1, 1, 1))
        
    def _create_conv_block(self, in_channels, out_channels, kernel_size, padding):
        """创建卷积块"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        """前向传播过程"""
        x = x.unsqueeze(1)  # 添加特征维度
        
        # 编码路径
        encoder_outputs = []
        current = x
        
        for i in range(self.num_levels):
            encoder_output = self.encoders[i](current)
            encoder_outputs.append(encoder_output)
            current = self.pooling_layers[i](encoder_output)
            # print(f"Encoder Level {i+1} Output Shape: {encoder_output.shape}")  # 输出每层形状
        
        # 特征融合
        # print(f"Before Fusion Shape: {current.shape}")
        current = self.st_fusion(current)
        # print(f"After Fusion Shape: {current.shape}")
        
        # 解码路径
        for i in range(self.num_levels):
            current = self.upconv_layers[i](current)
            current = current + encoder_outputs[self.num_levels - 1 - i]
            current = self.decoders[i](current)
            # print(f"Decoder Level {i+1} Output Shape: {current.shape}")
        
        # 输出层
        out = self.out_conv(current)
        
        # 残差连接
        if self.residual:
            out = out + x
            
        # 去除添加的深度维度
        out = out.squeeze(1)
        
        return out