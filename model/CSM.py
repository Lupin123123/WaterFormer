import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0], 0, 0))

        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

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
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
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

    def forward(self, x1, x2, x3):

        b1, h1w1, d1 = x1.shape
        b2, h2w2, d2 = x2.shape
        b3, h3w3, d3 = x3.shape

        h1 = w1 = int(np.sqrt(h1w1))
        h2 = w2 = int(np.sqrt(h2w2))
        h3 = w3 = int(np.sqrt(h3w3))   

        xx1 = x1.reshape(b1, d1, h1, w1)
        xx2 = x2.reshape(b2, d2, h2, w2)
        xx3 = x3.reshape(b3, d3, h3, w3)

        xx1, _, _ = self.patch_emb1(xx1)
        xx2, _, _= self.patch_emb2(xx2)
        xx3, _, _= self.patch_emb3(xx3)


        tmp = torch.cat([xx1, xx2, xx3], dim=1)
        out_csm = self.CSM(tmp)
 
        out1 = nn.Linear(in_features=out_csm.shape[-1], out_features=d1, bias=True)(out_csm)
        out2 = nn.Linear(in_features=out_csm.shape[-1], out_features=d2, bias=True)(out_csm)
        out3 = nn.Linear(in_features=out_csm.shape[-1], out_features=d3, bias=True)(out_csm)       

        out1 = nn.Linear(in_features=out_csm.shape[-2], out_features=h1w1, bias=True)(out1.transpose(-1, -2)).transpose(-1, -2)
        out2 = nn.Linear(in_features=out_csm.shape[-2], out_features=h2w2, bias=True)(out2.transpose(-1, -2)).transpose(-1, -2)
        out3 = nn.Linear(in_features=out_csm.shape[-2], out_features=h3w3, bias=True)(out3.transpose(-1, -2)).transpose(-1, -2)

        print(out1.shape)
        print(out2.shape)
        print(out3.shape)
        
        return out1, out2, out3