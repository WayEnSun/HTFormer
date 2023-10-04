"""
Various positional encodings for the transformer.
"""

import math

import torch
from torch import nn
from typing import List, Optional, Tuple



class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('ERROR: normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x,mask):
        # (b,d,h,w)
        # (b,h,w)

        assert mask is not None
        not_mask = ~mask  # (b,h,w)
        # 1 1 1 1... 2 2 2 2... 3 3 3 3...
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # Cumulative sum along axis 1 (h axis) --> (b,h,w)
        # 1 2 3 4... 1 2 3 4... 1 2 3 4...
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # Cumulative sum along axis 2 (w axis) --> (b,h,w)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 2pi * (y / sigma(y))
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 2pi * (x / sigma(x))

        # num_pos_feats = d/2
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # (0,1,2,...,d/2)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d)
        return pos  # (b,d,h,w)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        # x_emb: (w, d/2)
        # y_emb: (h, d/2)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos  # (H, W, C) --> (C, H, W) --> (1, C, H, W) --> (B, C, H, W)


class PositionEmbeddingNone(nn.Module):
    """
    No positional encoding.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.n_dim = num_pos_feats * 2

    def forward(self, x):
        b, _, h, w = x.size()
        return torch.zeros((b, self.n_dim, h, w), device=x.device)  # (B, C, H, W)


def build_position_encoding(cfg):
    N_steps = cfg.MODEL.HIDDEN_DIM // 2
    if cfg.MODEL.POSITION_EMBEDDING in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif cfg.MODEL.POSITION_EMBEDDING in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif cfg.MODEL.POSITION_EMBEDDING in ('none'):
        print('not using positional encoding')
        position_embedding = PositionEmbeddingNone(N_steps)
    else:
        raise ValueError(f'ERROR: not supported {cfg.MODEL.POSITION_EMBEDDING}')

    inner_position_embedding = PositionEmbeddingSine(cfg.MODEL.AIA.MATCH_DIM // 2, normalize=True)

    return position_embedding, inner_position_embedding


# if __name__== "__main__" :
#     import torch
#     import torch.nn as nn
#     from torchviz import make_dot
#     from typing import Dict, List
#     import numpy as np
#     img=torch.randn(1,1024,20,20)
#     mask=torch.from_numpy(np.array(range(400)).reshape(1,20,20).astype(np.bool_))
#     x=NestedTensor(img, mask)
#     N_steps = 256 // 2
#     p_e = PositionEmbeddingSine(N_steps, normalize=True)
#     i_p_e = PositionEmbeddingSine(64 // 2, normalize=True)

#     out: List[NestedTensor] = []
#     pos: List[NestedTensor] = []
#     inr: List[NestedTensor] = []


#     out.append(x)
#     # Position encoding
#     pos.append(p_e(x).to(x.tensors.dtype))
#     inr.append(i_p_e(x).to(x.tensors.dtype))

#     print("out 0 : ",out[0].tensors.shape)
#     print("mask : ",out[0].mask.shape)
#     print("pos : ",pos[0].shape)
#     print("inr : ",inr[0].shape)
#     # out 0 :  torch.Size([1, 1024, 20, 20])
#     # mask :  torch.Size([1, 20, 20])
#     # pos :  torch.Size([1, 256, 20, 20])
#     # inr :  torch.Size([1, 64, 20, 20])

#     feat=torch.cat([out[0].tensors],dim=0)
#     mask=torch.cat([out[0].mask],dim=1)
#     pos=torch.cat([pos[0]],dim=0)
#     inr=torch.cat([inr[0]],dim=0)
#     print("feat : ",feat.shape)
#     print("mask : ",mask.shape)
#     print("pos : ",pos.shape)
#     print("inr : ",inr.shape)