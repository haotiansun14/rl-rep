import torch
import torch.nn as nn
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar).sum(-1)
            else:
                return 0.5 * (
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar).sum(-1)

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


OUT_DIM = {2: 39, 4: 35, 6: 31}

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, latent_dim=50, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.latent_dim = latent_dim
        self.num_layers = num_layers

        convs = [nn.Conv2d(3, num_filters, 3, stride=2), nn.ReLU()]
        for i in range(num_layers-1):
            convs.extend([
                nn.Conv2d(num_filters, num_filters, 3, stride=1), 
                nn.ReLU()
            ])
        self.convs = nn.Sequential(*convs)

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(latent_dim, 2*latent_dim)

    def forward_conv(self, obs):
        B = obs.shape[0]
        obs = obs / 255. - 0.5
        obs = self.convs(obs)
        obs = obs.view(B, -1)
        return obs

    def forward(self, obs):
        h = self.forward_conv(obs)
        h = self.fc(h)
        h = self.ln(h)
        h = nonlinearity(h)
        h = self.out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, obs_shape, latent_dim, num_layers=4, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            latent_dim, num_filters * self.out_dim * self.out_dim
        )
        
        deconvs = []
        for i in range(self.num_layers-1):
            deconvs.extend([
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1), 
                nn.ReLU()
            ])
        deconvs.extend([nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=1), nn.ReLU()])
        deconvs.append(nn.Conv2d(num_filters, 3, 3, stride=1, padding=1))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        deconv = self.deconvs(deconv)
        return deconv


class VAE(nn.Module):
    def __init__(
        self, 
        obs_shape, 
        latent_dim: int=1024, 
        ae_num_layers: int=4, 
        ae_num_filters: int=32, 
    ) -> None:
        super().__init__()
        self.encoder = Encoder(obs_shape, latent_dim, ae_num_layers, ae_num_filters)
        self.decoder = Decoder(obs_shape, latent_dim, ae_num_layers, ae_num_filters)
        
    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
    def decode(self, z):
        z = self.decoder(z)
        return z
    
    def forward(self, x, sample_posterior=True, forward_decoder=False):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if forward_decoder:
            dec = self.decode(z)
            return z, posterior, dec
        else:
            return z, posterior
        
class Scaler(nn.Module):
    def __init__(self, activate=False):
        super().__init__()
        self.activate = activate
        self.initialized = False
        self.scale_factor = 1.0

    def init(self, batch):
        self.initialized = True
        if not self.activate:
            return
        self.scale_factor = batch.flatten().std()
        print(f"Scaler: using scale factor: {self.scale_factor}")

    def forward(self, x, reverse=False):
        if not self.activate: 
            return x
        if reverse:
            return x / self.scale_factor
        else:
            return x * self.scale_factor
        
