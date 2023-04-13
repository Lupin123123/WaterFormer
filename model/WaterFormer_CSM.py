import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (
            W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Downsample(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)
        self.conv = nn.Conv2d(
            in_channels=dim, out_channels=2*dim, stride=2, kernel_size=2)

    def forward(self, x, H, W):
        short_cut = x
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"

        x = x.view(B, C, H, W)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        H = x.shape[-2]
        W = x.shape[-1]
        x = self.conv(x).view(B, int((H*W)/4), -1)

        return x, short_cut


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super(Upsample, self).__init__()
        out_ch = int(in_ch/2)
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch *
                      2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch *
                      2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"

        x = x.view(B, C, H, W)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        H = x.shape[-2]
        W = x.shape[-1]
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out).view(B, int((H*W)*4), -1)

        return x_out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] = relative_coords[:, :, 0] + self.window_size[0] - 1
        relative_coords[:, :, 1] = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords[:, :, 0] = relative_coords[:, :, 0]*(2*self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(
                self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(
            shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        # [nW*B, Mh*Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # [nW*B, Mh, Mw, C]
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H*W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt = cnt + 1
        mask_windows = window_partition(
            img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        # [nW, Mh*Mw]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        tmp = self.to_qkv(x)
        # print(tmp.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # print(qkv[0].shape)
        # print(qkv[1].shape)
        # print(qkv[2].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape)
        # print(k.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # print(dots.shape)
        attn = self.attend(dots)
        # print(attn.shape)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # print(out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(out.shape)
        return self.to_out(out)


class CSM_Attention(nn.Module):
    def __init__(self) -> None:
        super(CSM_Attention, self).__init__()
        self.Attention = Attention(dim=96)

    def forward(self, x):
        y = self.Attention(x)
        return y


class CSM(nn.Module):
    def __init__(self) -> None:
        super(CSM, self).__init__()
        self.patch_emb1 = PatchEmbed(in_c=96)
        self.patch_emb2 = PatchEmbed(in_c=192)
        self.patch_emb3 = PatchEmbed(in_c=384)
        self.CSM = CSM_Attention()

    def forward(self, x1, x2):

        b1, h1w1, d1 = x1.shape
        b2, h2w2, d2 = x2.shape
        # b3, h3w3, d3 = x3.shape

        h1 = w1 = int(np.sqrt(h1w1))
        h2 = w2 = int(np.sqrt(h2w2))
        # h3 = w3 = int(np.sqrt(h3w3))

        xx1 = x1.reshape(b1, d1, h1, w1)
        xx2 = x2.reshape(b2, d2, h2, w2)
        # xx3 = x3.reshape(b3, d3, h3, w3)

        xx1, _, _ = self.patch_emb1(xx1)
        xx2, _, _ = self.patch_emb2(xx2)
        # xx3, _, _= self.patch_emb3(xx3)

        tmp = torch.cat([xx1, xx2], dim=1)
        out_csm = self.CSM(tmp)

        out1 = nn.Linear(
            in_features=out_csm.shape[-1], out_features=d1, bias=True)(out_csm)
        out2 = nn.Linear(
            in_features=out_csm.shape[-1], out_features=d2, bias=True)(out_csm)
        # out3 = nn.Linear(in_features=out_csm.shape[-1], out_features=d3, bias=True)(out_csm)

        out1 = nn.Linear(in_features=out_csm.shape[-2], out_features=h1w1, bias=True)(
            out1.transpose(-1, -2)).transpose(-1, -2)
        out2 = nn.Linear(in_features=out_csm.shape[-2], out_features=h2w2, bias=True)(
            out2.transpose(-1, -2)).transpose(-1, -2)

        return out1, out2


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3,
                 embed_dim=96, depths=(4, 4, 4, 4), num_heads=(8, 8, 8, 8),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim*2**i_layer) if i_layer < self.num_layers/2 else int(embed_dim*2**(self.num_layers-i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(
                                    depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=Downsample if (
                                    i_layer < self.num_layers-1) else None,
                                use_checkpoint=use_checkpoint
                                )
            self.layers.append(layers)

        self.CSM = CSM()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        B_ori, C_ori, H_ori, W_ori, = x.shape
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        short_cut_lst = []
        half = int(len(self.layers)/2)
        for layer_id in range(half):
            layer = self.layers[layer_id]
            x, H, W = layer(x, H, W)
            process = Downsample(dim=int(self.embed_dim*2**layer_id))
            x, short_cut = process(x, H, W)
            short_cut_lst.append(short_cut)
            H = int(H/2)
            W = int(W/2)

        cnt = 0
        CSM_out = []
        out1, out2 = self.CSM(short_cut_lst[0], short_cut_lst[1])
        CSM_out.append(out2)
        CSM_out.append(out1)
        for layer_id in range(half, len(self.layers)):
            layer = self.layers[layer_id]
            x, H, W = layer(x, H, W)
            process = Upsample(in_ch=int(self.embed_dim*2 **
                               (len(self.layers)-layer_id)))
            csm_out = CSM_out[cnt]
            x = process(x, H, W)
            x = x+ csm_out
            H = int(H*2)
            W = int(W*2)
            cnt = cnt + 1

        x = x.view(B_ori, self.embed_dim, int(
            H_ori/self.patch_size), int(W_ori/self.patch_size))
        x = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=3,
                               kernel_size=3, stride=2, padding=1, output_padding=1)(x)
        x = nn.ConvTranspose2d(in_channels=3, out_channels=3,
                               kernel_size=3, stride=2, padding=1, output_padding=1)(x)

        return x
