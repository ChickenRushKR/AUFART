import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from inspect import isfunction
from .dec_layers import *

class FLAMETransformerDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    """

    def __init__(self):
        super().__init__()
        transformer_args = dict(
            num_tokens=1,
            # token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            token_dim=1,
            dim=512,
            depth= 6,
            heads= 8,
            mlp_dim= 512,
            dim_head= 64,
            dropout= 0.0,
            emb_dropout= 0.0,
            norm= 'layer',
            context_dim=512 # from vitpose-H
        )
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decparams = nn.Linear(dim, 53)
        nn.init.xavier_uniform_(self.decparams.weight, gain=0.01)


    def forward(self, x, **kwargs):

        batch_size = x.shape[0]

        token = torch.zeros(batch_size, 1, 1).to(x.device)

        # Pass through transformer
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1) # (B, C)

        # Readout from token_out
        pred_exp_pose = self.decparams(token_out)

        return pred_exp_pose
    