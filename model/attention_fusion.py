from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        # 判断是否存在'context'，若不存在，则说明是self-attention，若存在，说明是cross-attention
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        temp = self.to_kv(context)
        k,v = temp.chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        depth,                                  # Process的层数，可以设置为0
        dim,                                    # Decoder输出特征的维度
        latent_dim = 512,                       # Encoder输入Query的分辨率
        cross_heads = 1,                        # Encoder的头大小
        latent_heads = 8,                       # Process的头大小
        cross_dim_head = 64,                    # Encoder和Decoder中计算过程中的维度
        latent_dim_head = 64,                   # Process计算过程中的维度
        weight_tie_layers = False,
    ):
        super().__init__()

        # Encoder部分，不一定是一层
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        # Process部分，多层
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
        # # Decoder部分，只能为一层
        # self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        # self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        # # 将最后结果进行重塑为指定想要的形状
        # self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,                           # Image特征
        mask = None,
        queries_encoder = None,         # PointCloud特征
        # queries_decoder = None          # PointCloud特征
    ):
        b, *_, device = *data.shape, data.device
        # PointCloud特征
        x = queries_encoder

        # ---- Encoder过程 ----
        cross_attn, cross_ff = self.cross_attend_blocks
        # 经过Attention得到Query与原Query相加，维度不变
        x = cross_attn(x, context = data, mask = mask) + x
        # 其目的是让特征较重要的部分，更加重要
        x = cross_ff(x) + x
        # ---- Encoder过程 ----


        #  ---- Process过程 ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        # ---- Process过程 ----

        return x

        # # ---- Decoder过程 ----
        # if not exists(queries_decoder):
        #     return x
        # # cross attend from decoder queries to latents
        # latents = self.decoder_cross_attn(queries_decoder, context = x)
        # # optional decoder feedforward
        # if exists(self.decoder_ff):
        #     latents = latents + self.decoder_ff(latents)
        # # final linear out
        # return self.to_logits(latents)
        # # ---- Decoder过程 ----
