from math import sqrt
from matplotlib import pyplot as plt

import numpy as np
import torch.nn as nn
import torch

from Modules.Loss import (LOSSDICT, JacobianDeterminantLoss, RBFBendingEnergyLossA)
from Modules.Interpolation import SpatialTransformer
from timm.models.layers import DropPath
import torch.nn.functional as F

from Networks.BaseNetwork import GenerativeRegistrationNetwork


def Sample(mu, log_var):
    eps = torch.randn(mu.size()).cuda()

    std = torch.exp(0.5 * log_var)

    z = mu + std * eps

    return z


class Gaussian(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super().__init__()

        self.mu = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.log_var = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


class RBFUnSharedEncoderGlobal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = Gaussian(8, 2, 3, 1, 1)
        self.fc_reverse = Gaussian(8, 2, 3, 1, 1)

    def forward(self, m2f, f2m):
        mu, log_var = self.fc(m2f)
        mu_reverse, log_var_reverse = self.fc_reverse(f2m)
        return (mu, log_var), (mu_reverse, log_var_reverse)


class RBFUnSharedEncoderLocal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = Gaussian(8, 2, 3, 1, 1)
        self.fc_reverse = Gaussian(8, 2, 3, 1, 1)

    def forward(self, m2f, f2m):
        mu, log_var = self.fc(m2f)
        mu_reverse, log_var_reverse = self.fc_reverse(f2m)
        return (mu, log_var), (mu_reverse, log_var_reverse)


class RBFEncoder(nn.Module):
    def __init__(self, global_dim, local_dim):
        super().__init__()
        self.unshared_encoder_global = RBFUnSharedEncoderGlobal(global_dim)
        self.unshared_encoder_local = RBFUnSharedEncoderLocal(local_dim)

    def forward(self, global_features, local_features):
        global_alpha, global_alpha_reverse = self.unshared_encoder_global(global_features[0], global_features[1])
        local_alpha, local_alpha_reverse = self.unshared_encoder_local(local_features[0], local_features[1])

        return global_alpha, global_alpha_reverse, local_alpha, local_alpha_reverse


class RBFDecoder(nn.Module):
    def __init__(self, img_size, int_steps=7):
        super().__init__()

        # location mesh of local control point
        lcp_loc_grid = [
            torch.linspace(s + (s - e) / 20, e - (s - e) / 20, 10)
            for s, e in ((2, 6), (2, 6))
        ]
        lcpoint_pos = torch.meshgrid(lcp_loc_grid)
        lcpoint_pos = torch.stack(lcpoint_pos, 2)[:, :, [1, 0]]
        lcpoint_pos = torch.flatten(lcpoint_pos, start_dim=0, end_dim=1).float()
        self.register_buffer('lcpoint_pos', lcpoint_pos)

        # location mesh of global control point
        cp_loc_grids = [torch.linspace(s + (e - s) / 16, e - (e - s) / 16, 8)
                        for s, e in ((0, 8), (0, 8))]
        cpoint_pos = torch.meshgrid(cp_loc_grids)
        cpoint_pos = torch.stack(cpoint_pos, 2)[:, :, [1, 0]]
        cpoint_pos = torch.flatten(cpoint_pos, start_dim=0, end_dim=1).float()
        self.register_buffer('cpoint_pos', cpoint_pos)


        cpoint_num_global = cpoint_pos.size()[0]
        cpoint_size_global = torch.max(cpoint_pos, 0)[0][[1, 0]]

        cpoint_num_local = lcpoint_pos.size()[0]
        cpoint_size_local = torch.max(lcpoint_pos, 0)[0][[1, 0]]

        # location mesh_global of output
        loc_vectors = [
            torch.linspace(0.0, 8, i_s)
            for (i_s, c_s) in zip(img_size, cpoint_size_global)
        ]
        loc = torch.meshgrid(loc_vectors)
        loc = torch.stack(loc, 2)
        loc = loc[:, :, [1, 0]].float().unsqueeze(2)
        loc_tile_global = loc.repeat(1, 1, cpoint_num_global, 1)
        self.register_buffer('loc_tile_global', loc_tile_global)
        loc_tile_local = loc.repeat(1, 1, cpoint_num_local, 1)
        self.register_buffer('loc_tile_local', loc_tile_local)

        self.img_size = img_size
        self.int_steps = int_steps

        cp_loc_global = self.cpoint_pos.unsqueeze(0).unsqueeze(0)
        cp_loc_tile_global = cp_loc_global.repeat(128, 128, 1, 1)

        # calculate r
        dist_global = torch.norm(self.loc_tile_global - cp_loc_tile_global, dim=3) / 2
        # add mask for r < 1
        mask_global = dist_global < 1
        weight_global = torch.pow(1 - dist_global, 4) * (4 * dist_global + 1)
        weight_global = weight_global * mask_global.float()
        weight_global = weight_global.unsqueeze(0).unsqueeze(4)
        self.register_buffer('w_g', weight_global)

        cp_loc_local = self.lcpoint_pos.unsqueeze(0).unsqueeze(0)
        cp_loc_tile_local = cp_loc_local.repeat(128, 128, 1, 1)

        # calculate r
        dist_local = torch.norm(self.loc_tile_local - cp_loc_tile_local, dim=3) / 1.5
        # add mask for r < 1
        mask_local = dist_local < 1
        weight_local = torch.pow(1 - dist_local, 4) * (4 * dist_local + 1)
        weight_local = weight_local * mask_local.float()
        weight_local = weight_local.unsqueeze(0).unsqueeze(4)
        self.register_buffer('w_l', weight_local)

        if self.int_steps:
            self.flow_transformer = SpatialTransformer([128, 128])

    def diffeomorphic(self, flow):
        v = flow / (2 ** self.int_steps)
        # v = flow
        v_list = []
        for _ in range(self.int_steps):
            v1 = self.flow_transformer(v, v)
            v = v + v1
            v_list.append(v)
        return v_list

    def forward(self, g_alpha, l_alpha):

        g_alpha = g_alpha.unsqueeze(1).unsqueeze(1)
        phi_global = torch.sum(self.w_g * g_alpha, 3)
        phi_global = phi_global.permute(0, 3, 1, 2)

        l_alpha = l_alpha.unsqueeze(1).unsqueeze(1)
        phi_local = torch.sum(self.w_l * l_alpha, 3)
        phi_local = phi_local.permute(0, 3, 1, 2)

        phi_global_list = []
        phi_local_list = []
        phi_list = []
        if self.int_steps:
            phi_global_list = self.diffeomorphic(phi_global)
            phi_local_list = self.diffeomorphic(phi_local)

        for i in range(len(phi_global_list)):
            phi_list.append(phi_global_list[i] + phi_local_list[i])

        return phi_list, self.cpoint_pos, self.lcpoint_pos


class RBFNetwork(nn.Module):
    def __init__(self, global_dim, local_dim, img_size=[128, 128], int_steps=None):
        super().__init__()

        self.encoder = RBFEncoder(global_dim, local_dim)
        self.decoder = RBFDecoder(img_size, int_steps)
        self.transformer = SpatialTransformer(img_size)
        self.transformer_reverse = SpatialTransformer(img_size)

    def forward(self, global_features, local_features, src, trg):
        global_alpha, global_alpha_reverse, local_alpha, local_alpha_reverse = self.encoder(global_features,
                                                                                            local_features)

        g_alpha = Sample(*global_alpha)
        g_alpha = torch.flatten(g_alpha, start_dim=2)
        g_alpha = g_alpha.permute(0, 2, 1)
        g_alpha_reverse = Sample(*global_alpha_reverse)
        g_alpha_reverse = torch.flatten(g_alpha_reverse, start_dim=2)
        g_alpha_reverse = g_alpha_reverse.permute(0, 2, 1)

        l_alpha = Sample(*local_alpha)
        l_alpha = torch.flatten(l_alpha, start_dim=2)
        l_alpha = l_alpha.permute(0, 2, 1)
        l_alpha_reverse = Sample(*local_alpha_reverse)
        l_alpha_reverse = torch.flatten(l_alpha_reverse, start_dim=2)
        l_alpha_reverse = l_alpha_reverse.permute(0, 2, 1)

        phi, cpoint_pos, lcpoint_pos = self.decoder(g_alpha, l_alpha)
        phi_reverse, cpoint_pos, lcpoint_pos = self.decoder(g_alpha_reverse, l_alpha_reverse)
        w_src = self.transformer(src, phi[-1])
        w_trg = self.transformer_reverse(trg, phi_reverse[-1])

        return phi, w_src, phi_reverse, w_trg, (global_alpha, local_alpha), (
            global_alpha_reverse, local_alpha_reverse), cpoint_pos, lcpoint_pos

    def test(self, global_features, local_features, src, trg):
        global_alpha, global_alpha_reverse, local_alpha, local_alpha_reverse = self.encoder(global_features,
                                                                                            local_features)

        g_alpha = global_alpha[0]
        g_alpha = torch.flatten(g_alpha, start_dim=2)
        g_alpha = g_alpha.permute(0, 2, 1)
        g_alpha_reverse = global_alpha_reverse[0]
        g_alpha_reverse = torch.flatten(g_alpha_reverse, start_dim=2)
        g_alpha_reverse = g_alpha_reverse.permute(0, 2, 1)

        l_alpha = local_alpha[0]
        l_alpha = torch.flatten(l_alpha, start_dim=2)
        l_alpha = l_alpha.permute(0, 2, 1)
        l_alpha_reverse = local_alpha_reverse[0]
        l_alpha_reverse = torch.flatten(l_alpha_reverse, start_dim=2)
        l_alpha_reverse = l_alpha_reverse.permute(0, 2, 1)

        phi_list, _, _ = self.decoder(g_alpha, l_alpha)
        phi_reverse_list, _, _ = self.decoder(g_alpha_reverse, l_alpha_reverse)
        w_src = self.transformer(src, phi_list[-1])
        w_trg = self.transformer_reverse(trg, phi_reverse_list[-1])

        return phi_list, w_src, phi_reverse_list, w_trg


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),  # 卷积3*3，步长为1，padding为1
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):  # 池化（下采样）
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(  # 序列构造器
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),  # 这里不采用最大池化，最大池化特征丢失太多，所以采用步长为2
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):  # 上采样
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)  # 1*1卷积，降低通道，无需特征提取，只是降通道数

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='bilinear')  # 最邻近插值法
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(2, 16)
        self.d1 = DownSample(16)
        self.c2 = Conv_Block(16, 32)
        self.d2 = DownSample(32)
        self.c3 = Conv_Block(32, 64)
        self.d3 = DownSample(64)
        self.c4 = Conv_Block(64, 128)
        self.d4 = DownSample(128)
        self.c5 = Conv_Block(128, 256)
        self.u1 = UpSample(256)
        self.c6 = Conv_Block(256, 128)
        self.u2 = UpSample(128)
        self.c7 = Conv_Block(128, 64)
        self.u3 = UpSample(64)
        self.c8 = Conv_Block(64, 32)
        self.u4 = UpSample(32)
        self.c9 = Conv_Block(32, 16)
        self.out = nn.Conv2d(16, 2, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, flag_global, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )

        self.res = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        b, hw, dim = x.shape
        new_h = int(sqrt(hw))
        x = self.mlp(x) + x
        x_res = x.permute(0, 2, 1).reshape(b, dim, new_h, new_h)
        x = self.net(x_res) + self.res(x_res)
        x = x.permute(0, 2, 3, 1).reshape(b, new_h * new_h, self.hidden_dim)
        return x


class Attention_1(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x  # because the original x has different size with current x, use v to do skip connection

        return x


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_1(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PreUnfoldGlobal(nn.Module):
    def __init__(self, input_dim):
        super(PreUnfoldGlobal, self).__init__()
        self.soft1 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.soft2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.soft3 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.att1 = Token_transformer(dim=input_dim * 7 * 7, in_dim=32, num_heads=1, mlp_ratio=1.0)
        self.att2 = Token_transformer(dim=32 * 3 * 3, in_dim=32, num_heads=1, mlp_ratio=1.0)
        self.linear = nn.Linear(32 * 3 * 3, 256)

    def forward(self, src):
        x0 = self.soft1(src).transpose(1, 2)
        x1 = self.att1(x0)
        B, new_Hw, C = x1.shape
        x1 = x1.transpose(1, 2).reshape(B, C, int(np.sqrt(new_Hw)), int(np.sqrt(new_Hw)))
        x1 = self.soft2(x1).transpose(1, 2)
        x2 = self.att2(x1)
        B, new_Hw, C = x2.shape
        x2 = x2.transpose(1, 2).reshape(B, C, int(np.sqrt(new_Hw)), int(np.sqrt(new_Hw)))
        x3 = self.soft3(x2).transpose(1, 2)
        output = self.linear(x3)
        return output


class PreUnfoldLocal(nn.Module):
    def __init__(self, input_dim):
        super(PreUnfoldLocal, self).__init__()
        self.soft1 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.soft2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.soft3 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.soft4 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        self.att1 = Token_transformer(dim=input_dim * 7 * 7, in_dim=32, num_heads=1, mlp_ratio=1.0)
        self.att2 = Token_transformer(dim=32 * 3 * 3, in_dim=32, num_heads=1, mlp_ratio=1.0)
        self.att3 = Token_transformer(dim=32 * 3 * 3, in_dim=32, num_heads=1, mlp_ratio=1.0)
        self.linear = nn.Linear(32 * 3 * 3, 256)

    def forward(self, src):
        x0 = self.soft1(src).transpose(1, 2)
        x1 = self.att1(x0)
        B, new_Hw, C = x1.shape
        x1 = x1.transpose(1, 2).reshape(B, C, int(np.sqrt(new_Hw)), int(np.sqrt(new_Hw)))
        x1 = self.soft2(x1).transpose(1, 2)
        x2 = self.att2(x1)
        B, new_Hw, C = x2.shape
        x2 = x2.transpose(1, 2).reshape(B, C, int(np.sqrt(new_Hw)), int(np.sqrt(new_Hw)))
        x2 = self.soft3(x2).transpose(1, 2)
        x3 = self.att3(x2)
        B, new_Hw, C = x3.shape
        x3 = x3.transpose(1, 2).reshape(B, C, int(np.sqrt(new_Hw)), int(np.sqrt(new_Hw)))
        x4 = self.soft4(x3).transpose(1, 2)
        output = self.linear(x4)
        return output


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, depth, num_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.Norm = nn.LayerNorm(hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        self.o = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.q(self.Norm(query))
        K = self.k(self.Norm(key))
        V = self.v(self.Norm(value))

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        matmul_data = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            matmul_data = matmul_data.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(matmul_data, dim=-1)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hidden_dim)

        x = self.o(x)

        return x


class Encoder(nn.Module):
    def __init__(self, image_size, patch_num, dim, depth, channels, num_heads, mlp_dim, flag_global, dropout=0.1,
                 embedding_dropout=0.1, ratio=[1, 2, 4, 8, 16, 32]):
        super().__init__()

        # 位置编码
        self.pos_embedding_f2m = nn.Parameter(torch.randn(1, patch_num, dim))
        self.pos_embedding_m2f = nn.Parameter(torch.randn(1, patch_num, dim))
        self.dropout = nn.Dropout(embedding_dropout)
        self.num_heads = num_heads
        self.flag = flag_global
        self.ratio = ratio
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttentionLayer(dim // ratio[i], depth, num_heads[i]),
                mlp(dim // ratio[i], dim // ratio[i + 1], flag_global, dropout=dropout),
                MultiHeadAttentionLayer(dim // ratio[i], depth, num_heads[i]),
                mlp(dim // ratio[i], dim // ratio[i + 1], flag_global, dropout=dropout),
            ]))

    def forward(self, src, trg, mask=None):

        b = src.shape[0]

        src_pos = src + self.pos_embedding_m2f
        src_pos = self.dropout(src_pos)

        trg_pos = trg + self.pos_embedding_f2m
        trg_pos = self.dropout(trg_pos)

        for attn_m2f, mlp_m2f, attn_f2m, mlp_f2m in self.layers:
            src_pos_feature = attn_m2f(src_pos, trg_pos, trg_pos, mask) + src_pos
            trg_pos_feature = attn_f2m(trg_pos, src_pos, src_pos, mask) + trg_pos
            src_pos_feature = mlp_m2f(src_pos_feature)
            trg_pos_feature = mlp_f2m(trg_pos_feature)

            src_pos = src_pos_feature
            trg_pos = trg_pos_feature

        final_m2f = src_pos
        final_f2m = trg_pos

        if self.flag is False:
            final_m2f = final_m2f.permute(0, 2, 1)
            final_m2f = final_m2f.contiguous().view(b, 8, 10, 10)
            final_f2m = final_f2m.permute(0, 2, 1)
            final_f2m = final_f2m.contiguous().view(b, 8, 10, 10)
        else:
            final_m2f = final_m2f.permute(0, 2, 1)
            final_m2f = final_m2f.contiguous().view(b, 8, 8, 8)
            final_f2m = final_f2m.permute(0, 2, 1)
            final_f2m = final_f2m.contiguous().view(b, 8, 8, 8)

        return final_m2f, final_f2m


class Transformer(nn.Module):
    def __init__(self, image_size, patch_num, dim, depth_encoder, channels, num_heads, mlp_dim,
                 flag_global):
        super().__init__()

        self.encoder = Encoder(image_size, patch_num, dim, depth_encoder, channels, num_heads, mlp_dim, flag_global)

    def forward(self, src, trg):
        output = self.encoder(src, trg)
        return output


class TransRBF(GenerativeRegistrationNetwork):
    def __init__(self,
                 i_size=[128, 128],
                 c_list=[1.5, 2, 2.5],
                 factor_list=[130000, 50],
                 similarity_loss='WLCC',
                 similarity_loss_param={},
                 int_steps=7):
        super(GenerativeRegistrationNetwork, self).__init__(i_size)

        # 图像T2T预处理
        self.global_fixed_unfold = PreUnfoldGlobal(input_dim=1)
        self.local_fixed_unfold = PreUnfoldLocal(input_dim=1)

        # 预处理后的数据进入Transformer
        self.transformer1 = Transformer(image_size=128, patch_num=64, dim=256,
                                        depth_encoder=5, channels=1, num_heads=[16, 8, 4, 2, 1],
                                        mlp_dim=512, flag_global=True)
        self.transformer2 = Transformer(image_size=64, patch_num=100, dim=256,
                                        depth_encoder=5, channels=1, num_heads=[16, 8, 4, 2, 1],
                                        mlp_dim=512, flag_global=False)
        # 逆一致性网络
        self.unet = UNet()

        # RBF径向基函数，在VAE中充当概率解码器
        self.RBF = RBFNetwork(global_dim=16, local_dim=16, int_steps=int_steps)

        # 计算扭曲能量
        self.bending_energy_cal = RBFBendingEnergyLossA()
        self.bending_energy_cal_reverse = RBFBendingEnergyLossA()

        # LCC相似性度量
        self.similarity_loss = LOSSDICT[similarity_loss](
            **similarity_loss_param)
        self.similarity_loss_reverse = LOSSDICT[similarity_loss](
            **similarity_loss_param)

        self.bidir_sim = LOSSDICT[similarity_loss]([9, 9])

        # 雅可比形变约束
        self.jacobian_loss = JacobianDeterminantLoss()
        self.jacobian_loss_reverse = JacobianDeterminantLoss()

        self.factor_list = factor_list
        self.c_list = c_list
        self.int_steps = 7

        # STN
        self.forward_transformer = SpatialTransformer([128, 128])
        self.reverse_transformer = SpatialTransformer([128, 128])
        self.flow_transformer = SpatialTransformer([128, 128])

        # generate a name 
        name = str(similarity_loss) + '--'
        for k in similarity_loss_param:
            name += '-' + str(similarity_loss_param[k])
        name += '--'
        for i in self.factor_list:
            name += str(i)
        self.name = name
        if int_steps:
            self.name += '-diff'
            self.name += str(int_steps)
        
        

    def load_flattened_weights(self, flattened_weights,m,v,toprate):


        def compute_snr_exp_mean_abs(m,v,lambda_exp=0.1):
            def norm_cdf(x):
                """使用PyTorch计算标准正态分布的累积分布函数（CDF）"""
                return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0,device='cuda'))))
            """
            输入模型参数，计算SNR_exp_mean_abs指标
            :param model_params: 张量，张量形状为 (l,)
            :param lambda_exp: 指数变换的超参数λ
            :return: SNR_exp_mean_abs值，形状 (l,)
            """

            # 计算均值和标准差
            w = m
            #方差变标准差
            sigma = torch.sqrt(v)

            # 处理sigma=0的情况（避免除以零）
            mask = (sigma == 0)
            safe_sigma = torch.where(mask, torch.tensor(1e-8,device='cuda'), sigma)

            # 计算 E_exp_mean_abs = E[e^{λ|X|}]
            term1 = torch.exp(0.5 * (lambda_exp ** 2) * safe_sigma ** 2 + lambda_exp * w) * norm_cdf(
                w / safe_sigma + lambda_exp * safe_sigma)
            term2 = torch.exp(0.5 * (lambda_exp ** 2) * safe_sigma ** 2 - lambda_exp * w) * norm_cdf(
                -w / safe_sigma + lambda_exp * safe_sigma)
            E_exp_mean_abs = term1 + term2

            # 当sigma=0时，直接计算 e^{λ|w|}
            E_exp_mean_abs = torch.where(mask, torch.exp(lambda_exp * torch.abs(w)), E_exp_mean_abs)

            # 计算 E_exp_mean_abs_2 = E[e^{2λ|X|}]
            lam = 2 * lambda_exp
            term1_2 = torch.exp(0.5 * (lam ** 2) * safe_sigma ** 2 + lam * w) * norm_cdf(
                w / safe_sigma + lam * safe_sigma)
            term2_2 = torch.exp(0.5 * (lam ** 2) * safe_sigma ** 2 - lam * w) * norm_cdf(
                -w / safe_sigma + lam * safe_sigma)
            E_exp_mean_abs_2 = term1_2 + term2_2

            # 当sigma=0时，直接计算 e^{2λ|w|}
            E_exp_mean_abs_2 = torch.where(mask, torch.exp(2 * lambda_exp * torch.abs(w)), E_exp_mean_abs_2)

            # 计算SNR：均值/标准差
            variance = E_exp_mean_abs_2 - E_exp_mean_abs ** 2
            variance = torch.clamp(variance, min=1e-8)  # 防止负值
            snr = E_exp_mean_abs / torch.sqrt(variance)

            # 处理可能的NaN
            snr = torch.nan_to_num(snr, nan=0.0)

            return snr
        snr_values = compute_snr_exp_mean_abs(m, v)
        
            
        threshold = torch.quantile(snr_values, 1 - toprate / 100.0)
        important_indices = snr_values >= threshold

        idx = 0
        with torch.no_grad():
            for param in self.parameters():
                num_w = param.numel()
                vec = flattened_weights[idx:idx + num_w]
                mask = important_indices[idx:idx + num_w]
                
                param.copy_(torch.where(mask.view_as(param), vec.view_as(param), param))

                idx += num_w

    def get_weight_vector(self):
        return torch.cat([params.view(-1) for params in self.parameters()])

    def diffeomorphic(self, flow):
        v = flow / (2 ** self.int_steps)
        v_list = []
        for _ in range(self.int_steps):
            v1 = self.flow_transformer(v, v)
            v = v + v1
            v_list.append(v)
        return v_list

    def forward(self, src, trg):
        src1 = self.global_fixed_unfold(src)
        trg1 = self.global_fixed_unfold(trg)
        global_features = self.transformer1(src1, trg1)
        src_64, trg_64 = src[:, :, 32:96, 32:96], trg[:, :, 32:96, 32:96]
        src_642 = self.local_fixed_unfold(src_64)
        trg_642 = self.local_fixed_unfold(trg_64)
        local_features = self.transformer2(src_642, trg_642)
        phi_list_ed2es, w_src, phi_list_es2ed, w_trg, (global_alpha, local_alpha), (
            global_alpha_reverse, local_alpha_reverse), cpoint_pos, lcpoint_pos = self.RBF(global_features,
                                                                                           local_features,
                                                                                           src, trg)
        return phi_list_ed2es, w_src, phi_list_es2ed, w_trg, (global_alpha, local_alpha), (
            global_alpha_reverse, local_alpha_reverse), cpoint_pos, lcpoint_pos

    def test(self, src, trg):
        src1 = self.global_fixed_unfold(src)
        trg1 = self.global_fixed_unfold(trg)
        global_features = self.transformer1(src1, trg1)
        src_64, trg_64 = src[:, :, 32:96, 32:96], trg[:, :, 32:96, 32:96]
        src_642 = self.local_fixed_unfold(src_64)
        trg_642 = self.local_fixed_unfold(trg_64)
        local_features = self.transformer2(src_642, trg_642)
        phi_list_ed2es, w_src, phi_list_es2ed, w_trg = self.RBF.test(global_features, local_features, src, trg)

        return phi_list_ed2es, w_src, phi_list_es2ed, w_trg


    def test_unet(self, src, trg):
        phi_ed2es, w_src, phi_es2ed, w_trg = self.test(src, trg)
        phi_ed2es_reverse = self.unet(phi_ed2es[-1])
        phi_list_ed2es_reverse = self.diffeomorphic(phi_ed2es_reverse)
        phi_es2ed_reverse = self.unet(phi_es2ed[-1])
        phi_list_es2ed_reverse = self.diffeomorphic(phi_es2ed_reverse)
        return phi_list_ed2es_reverse[-1], phi_list_es2ed_reverse[-1]


    def objective(self, src, trg):
        phi_list_ed2es, w_src, phi_list_es2ed, w_trg, (global_alpha, local_alpha), (
            global_alpha_reverse, local_alpha_reverse), cpoint_pos, lcpoint_pos = self(src, trg)
        # sigma terms
        sigmas = torch.flatten(global_alpha[1], start_dim=2).permute(0, 2, 1)
        sigmad = torch.flatten(local_alpha[1], start_dim=2).permute(0, 2, 1)
        sigmas_reverse = torch.flatten(global_alpha_reverse[1], start_dim=2).permute(0, 2, 1)
        sigmad_reverse = torch.flatten(local_alpha_reverse[1], start_dim=2).permute(0, 2, 1)

        mus = torch.flatten(global_alpha[0], start_dim=2).permute(0, 2, 1)
        mud = torch.flatten(local_alpha[0], start_dim=2).permute(0, 2, 1)
        mus_reverse = torch.flatten(global_alpha_reverse[0], start_dim=2).permute(0, 2, 1)
        mud_reverse = torch.flatten(local_alpha_reverse[0], start_dim=2).permute(0, 2, 1)

        sigma_term_global = torch.sum(torch.exp(sigmas) - sigmas, dim=[1, 2])
        sigma_term_local = torch.sum(torch.exp(sigmad) - sigmad, dim=[1, 2])
        sigma_term_global_reverse = torch.sum(torch.exp(sigmas_reverse) - sigmas_reverse, dim=[1, 2])
        sigma_term_local_reverse = torch.sum(torch.exp(sigmad_reverse) - sigmad_reverse, dim=[1, 2])

        smooth_term_global = self.bending_energy_cal(mus, cpoint_pos, 2)
        smooth_term_local = self.bending_energy_cal(mud, lcpoint_pos, 1.5)
        smooth_term_global_reverse = self.bending_energy_cal_reverse(mus_reverse, cpoint_pos, 2)
        smooth_term_local_reverse = self.bending_energy_cal_reverse(mud_reverse, lcpoint_pos, 1.5)

        kl_loss_global = sigma_term_global * 0.5 + smooth_term_global
        kl_loss_local = sigma_term_local * 0.5 + smooth_term_local
        kl_loss_global_reverse = sigma_term_global_reverse * 0.5 + smooth_term_global_reverse
        kl_loss_local_reverse = sigma_term_local_reverse * 0.5 + smooth_term_local_reverse

        kl_loss = kl_loss_global + kl_loss_local
        kl_loss_reverse = kl_loss_global_reverse + kl_loss_local_reverse

        jacobian_loss = self.jacobian_loss(phi_list_ed2es[-1])
        jacobian_loss_reverse = self.jacobian_loss_reverse(phi_list_es2ed[-1])

        similarity_loss = self.similarity_loss(w_src, trg)
        similarity_loss_reverse = self.similarity_loss_reverse(w_trg, src)

        return {
            'similarity_loss':
                (similarity_loss + similarity_loss_reverse) / 2,
            'sigma_global':
                (sigma_term_global + sigma_term_global) / 2,
            'sigma_local':
                (sigma_term_local + sigma_term_local) / 2,
            'smooth_global':
                (smooth_term_global + smooth_term_global) / 2,
            'smooth_local':
                (smooth_term_local + smooth_term_local) / 2,
            'KL_loss_global':
                (kl_loss_global + kl_loss_global_reverse) / 2,
            'KL_loss_local':
                (kl_loss_local + kl_loss_local_reverse) / 2,
            'KL_loss':
                (kl_loss + kl_loss_reverse) / 2,
            'jacobian_loss':
                (jacobian_loss + jacobian_loss_reverse) / 2,
            'loss':
                self.factor_list[0] * ((similarity_loss + similarity_loss_reverse) / 2) + (
                        kl_loss + kl_loss_reverse) / 2
        }


    def objective_unet(self, src, trg):
        phi_list_ed2es, w_src, phi_list_es2ed, w_trg = self.test(src, trg)
        phi_ed2es_reverse = self.unet(phi_list_ed2es[-1])
        phi_list_ed2es_reverse = self.diffeomorphic(phi_ed2es_reverse)
        phi_es2ed_reverse = self.unet(phi_list_es2ed[-1])
        phi_list_es2ed_reverse = self.diffeomorphic(phi_es2ed_reverse)

        Id_loss_list = []
        for i in range(len(phi_list_ed2es)):
            Id_loss_forward = self.similarity_loss(self.forward_transformer(src, phi_list_ed2es[i]),
                                                   self.reverse_transformer(trg, phi_list_ed2es_reverse[
                                                       len(phi_list_ed2es) - i - 1]))
            Id_loss_reverse = self.similarity_loss(self.reverse_transformer(trg, phi_list_es2ed[i]),
                                                   self.forward_transformer(src, phi_list_es2ed_reverse[
                                                       len(phi_list_ed2es) - i - 1]))
            Id_loss_list.append((Id_loss_forward + Id_loss_reverse) / 2)

        Id_loss = torch.mean(torch.stack(Id_loss_list), dim=0)

        return {
            'Id_loss':
                Id_loss
        }


    def objective_bidir(self, src, trg):
        phi_list_ed2es, w_src, phi_list_es2ed, w_trg, (global_alpha, local_alpha), (
            global_alpha_reverse, local_alpha_reverse), cpoint_pos, lcpoint_pos = self(src, trg)

        phi_ed2es_reverse = self.unet(phi_list_ed2es[-1])
        phi_list_ed2es_reverse = self.diffeomorphic(phi_ed2es_reverse)
        phi_es2ed_reverse = self.unet(phi_list_es2ed[-1])
        phi_list_es2ed_reverse = self.diffeomorphic(phi_es2ed_reverse)

        Id_loss_list = []
        for i in range(len(phi_list_ed2es)):
            Id_forward = torch.mean(torch.abs(phi_list_ed2es[i] - phi_list_es2ed_reverse[i].detach()), dim=[1, 2, 3])
            Id_reverse = torch.mean(torch.abs(phi_list_es2ed[i] - phi_list_ed2es_reverse[i].detach()), dim=[1, 2, 3])
            Id_loss_list.append((Id_forward + Id_reverse) / 2)

        bidir_loss = torch.mean(torch.stack(Id_loss_list), dim=0)

        return {
            'bidir_loss':
                bidir_loss
        }
