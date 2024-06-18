import math
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from network_arch.net.mlp import MLP


def get_downsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(in_dim * 4, hidden_dim, 1))
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)

def get_upsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.Conv2d(in_dim, hidden_dim, 3, padding=1))
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
    
def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - 1.0)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([emb.sin(), emb.cos()], dim=-1)

def get_attn_layer(in_dim, use_full_attn):
    if use_full_attn:
        return Attention(in_dim)
    else:
        return nn.Identity()


class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1]**0.5)
    
    
class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        self.norm = RMSNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        temb_channels, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        groups=8
    ):
        super().__init__()
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(), torch.nn.Linear(temb_channels, out_channels))

        self.residual_conv = nn.Conv2d(
            in_channels, out_channels=out_channels,
            kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.nonlinearity = nn.SiLU()
        
    def forward(self, x, temb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        temb = self.time_emb_proj(self.nonlinearity(temb))
        x += temb[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)

        return x + residual


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 time_dim, 
                 hidden_dims=[64, 128, 256, 512],
                 image_size=64, 
                 out_channels=None):
        super(UNet, self).__init__()

        self.sample_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_dims = hidden_dims
        self.time_dim = time_dim

        time_embed_dim = max(hidden_dims)

        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim))

        self.init_conv = nn.Conv2d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        down_blocks = []
        in_dim = hidden_dims[0]
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)
            is_first = idx == 0
            down_blocks.append(
                nn.ModuleList([
                    ResidualBlock(in_dim, in_dim, time_embed_dim),
                    ResidualBlock(in_dim, in_dim, time_embed_dim),
                    get_attn_layer(in_dim, not is_first),
                    get_downsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim
        self.down_blocks = nn.ModuleList(down_blocks)
        
        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
        
        up_blocks = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)
            up_blocks.append(
                nn.ModuleList([
                    ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                    ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                    get_attn_layer(in_dim, not is_last),
                    get_upsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim
        self.up_blocks = nn.ModuleList(up_blocks)
        
        self.out_block = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0],
                                       time_embed_dim)
        self.conv_out = nn.Conv2d(hidden_dims[0], out_channels=self.out_channels, kernel_size=1)
        
    def forward(self, sample, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=sample.device)

        timesteps = torch.flatten(timesteps)
        timesteps = timesteps.broadcast_to(sample.shape[0])

        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_emb = self.time_embedding(t_emb)

        x = self.init_conv(sample)
        r = x.clone()

        skips = []
        for block1, block2, attn, downsample in self.down_blocks:
            x = block1(x, t_emb)
            skips.append(x)

            x = block2(x, t_emb)
            x = attn(x)
            skips.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block1, block2, attn, upsample in self.up_blocks:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t_emb)

            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, t_emb)
            x = attn(x)

            x = upsample(x)

        x = self.out_block(torch.cat((x, r), dim=1), t_emb)
        out = self.conv_out(x)
        return out

        
class Phi(nn.Module):
    def __init__(
        self, 
        latent_size: int, 
        latent_channels: int, 
        action_dim: int, 
        bn_dim: int, 
        feature_dim: int, 
        hidden_dim: int, 
        hidden_depth: int
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(3*latent_channels*latent_size*latent_size, bn_dim), 
            nn.LayerNorm(bn_dim), 
            nn.Tanh()
        )
        self.model = MLP(
            input_dim=bn_dim+action_dim, 
            output_dim=feature_dim, 
            hidden_dims=[hidden_dim]*hidden_depth
        )
    
    def forward(self, latent, action):
        x = self.bottleneck(latent.view(latent.shape[0], -1))
        x = torch.cat([x, action], axis=-1)
        return self.model(x)


class NablaMuWithTimeEmbed(nn.Module):
    def __init__(
        self, 
        latent_size: int, 
        latent_channels: int, 
        feature_dims: int, 
        time_dim: int, 
        hidden_dims=[8, 16, 64]
    ):
        super().__init__()
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.feature_dims = feature_dims
        self.unet = UNet(
            in_channels=latent_channels, 
            time_dim=time_dim, 
            hidden_dims=hidden_dims, 
            image_size=latent_size, 
            out_channels=feature_dims*latent_channels
        )
        
    def forward(self, perturbed_next_latent, timestep):
        out =  self.unet(perturbed_next_latent, timestep)
        out = out.reshape(-1, self.feature_dims, self.latent_channels, self.latent_size, self.latent_size)
        return out