from __future__ import annotations

import math
from math import sqrt
from typing import Literal
from functools import partial

import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from x_transformers import Decoder

from vit import DenoiseViT

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

# sinusoidal embedding

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class LearnedSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# simple mlp

class MLP(Module):
    def __init__(
        self,
        dim_cond,
        dim_input,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        super().__init__()
        layers = ModuleList([])

        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim_cond),
        )

        for _ in range(depth):

            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )

            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):
        assert noised.ndim == 2

        time_emb = self.to_time_emb(times)
        cond = F.silu(time_emb + cond)

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual

        return denoised

class ArSpElucidatedDiffusion(Module):
    def __init__(
        self,
        dim: int,
        net: MLP,
        *,
        sample_steps = 99,     # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        clamp_during_sampling = True
    ):
        super().__init__()

        self.net = net
        self.dim = dim

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.sample_steps = sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        self.clamp_during_sampling = clamp_during_sampling

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_seq, sigma, *, cond, clamp = None):
        clamp = default(clamp, self.clamp_during_sampling)

        batch, device = noised_seq.shape[0], noised_seq.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = right_pad_dims_to(noised_seq, sigma)

        net_out = self.net(
            self.c_in(padded_sigma) * noised_seq,
            times = self.c_noise(sigma),
            cond = cond
        )

        out = self.c_skip(padded_sigma) * noised_seq +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, sample_steps = None):
        sample_steps = default(sample_steps, self.sample_steps)

        N = sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return torch.flip(sigmas, [0])
    
    def sample_spatial_schedule(self, sample_steps = None):
        sigmas = self.sample_schedule()
        return torch.flip(sigmas, [0]).unsqueeze(0).repeat(3,1)
    
    def sample_spatial(self, t):
        idx = repeat(torch.tensor([self.sample_steps]), '1 -> n', n=self.sample_steps)
        s = torch.zeros(self.sample_steps)
        b = torch.tensor(list(range(min(self.sample_steps, t), 0, -1)))
        s[:min(self.sample_steps, t)] = b
        return (idx - s).int()

    @torch.no_grad()
    def sample(
        self,
        denoised_pred,
        t,
        cond, 
        sample_steps = None, 
        clamp = True
    ):
        clamp = default(clamp, self.clamp_during_sampling)
        sample_steps = default(sample_steps, self.sample_steps)

        shape = denoised_pred.shape

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        # Double check this is correct ---
        sigmas = self.sample_schedule(sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / sample_steps, sqrt(2) - 1),
            0.
        )

        spatial = self.sample_spatial(t)
        sigma = repeat(sigmas[spatial], "d -> b d 1", b = shape[0])
        gamma = repeat(gammas[spatial], "d -> b d 1", b = shape[0])
        sigma_next = sigma - 1

        eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling
        sigma_hat = sigma + gamma * sigma
        seq_hat = denoised_pred + torch.sqrt(sigma_hat ** 2 - sigma ** 2) * eps
        # ---

        model_output = self.preconditioned_network_forward(seq_hat, sigma_hat, cond = cond, clamp = clamp)
        denoised_over_sigma = (seq_hat - model_output) / sigma_hat
        seq = seq_hat + (sigma_next - sigma_hat) * denoised_over_sigma

        pred = seq[:,0,:]

        # second order correction, if not the last timestep
        model_output_next = self.preconditioned_network_forward(seq, sigma_next, cond = cond, clamp = clamp)
        denoised_prime_over_sigma = (seq - model_output_next) / sigma_next
        seq = seq_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

        if clamp:
            seq = seq.clamp(-1., 1.)

        return pred, seq

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, seq, cond):
        batch_size, seq_len, dim, device = *seq.shape, self.device

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = right_pad_dims_to(seq, sigmas)
        noise = torch.randn_like(seq)
        
        noised_seq = seq + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_seq, sigmas, cond = cond)

        losses = F.mse_loss(denoised, seq, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()

class ArSpDiffusion(Module):
    def __init__(
        self,
        dim,
        num_classes,
        window_size,
        sample_steps,
        sample_size,
        depth = 8,
        dim_head = 64,
        heads = 8,
        dim_input = None,
        decoder_kwargs: dict = dict(),
        mlp_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict(
            clamp_during_sampling = True
        )
    ):
        super().__init__()

        self.start_token = nn.Parameter(torch.zeros(dim))
        self.sample_size = sample_size
        self.sample_steps = sample_steps
        self.window_size = window_size

        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim)

        dim_input = default(dim_input, dim)
        self.dim = dim
        self.dim_input = dim_input
        self.proj_in = nn.Linear(dim_input, dim)

        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        # Using specialized ViT instead of MLP - to attend over future as well
        self.denoiser = DenoiseViT(
            dim_cond = dim,
            dim_input = dim_input,
            dim = 256,
            depth = 6,
            heads = 12,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1,
            **mlp_kwargs
        )

        self.diffusion = ArSpElucidatedDiffusion(
            dim_input,
            self.denoiser,
            sample_steps=sample_steps,
            **diffusion_kwargs
        )

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        label = 1
    ):
        self.eval()

        start_tokens = repeat(self.start_token, 'd -> b p d', b = batch_size, p=self.sample_steps)

        
        out = torch.empty((batch_size, 0, self.dim_input), device = self.device, dtype = torch.float32)
        cond = repeat(self.label_embedding(label), 'd -> b 1 d', b = batch_size)

        cache = None
        
        sigma_init = self.diffusion.sample_schedule(self.sample_steps)[0]
        denoised_seq = sigma_init * torch.randn((batch_size, self.sample_steps, self.dim_input), device = self.device)

        for t in tqdm(range(self.sample_steps + self.sample_size), desc = 'tokens'):
            
            cond = torch.cat((start_tokens, cond), dim = 1)[:,-self.sample_steps:,:]

            cond, cache = self.transformer(cond, cache = cache, return_hiddens = True)

            pred, denoised_seq = self.diffusion.sample(denoised_seq, t, cond = cond[:,-min(self.window_size, cond.shape[1]):, :])
            
            # first t steps are warmup
            if t > self.sample_steps:
                # add denoised center to out
                pred = repeat(pred, 'b d -> b 1 d')
                out = torch.cat((out, pred), dim = 1)
                
                # compute the next cond token
                cond = self.proj_in(out)

                # Add new rand sample to end
                rand_sample = sigma_init * torch.randn((batch_size, 1, self.dim_input), device = self.device)
                denoised_seq = torch.cat((denoised_seq[:,1:,:], rand_sample), dim = 1)

        return out

    def forward(
        self,
        seq,
        label,
        offset
    ):
        b, seq_len, dim = seq.shape

        # append start tokens
        cond = seq[:, (offset-self.sample_steps+2):offset, :]
        target = seq[:, offset:(offset+self.sample_steps) , :]

        cond = self.proj_in(cond)
        start_token = repeat(self.start_token, 'd -> b 1 d', b = b)
        label_token  = repeat(self.label_embedding(label), '1 d -> b 1 d', b = b)

        cond = torch.cat((start_token, label_token, cond), dim = 1)

        cond = self.transformer(cond)

        # only look at previous window_size for past condition
        diffusion_loss = self.diffusion(target, cond = cond)

        return diffusion_loss

def normalize_to_neg_one_to_one(patches):
    return (patches / 255) * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5 

class ArSpImageDiffusion(Module):
    def __init__(
        self,
        *,
        patch_size,
        num_classes,
        window_size = 256,
        sample_steps = 99,
        sample_size = 500,
        channels = 3,
        model: dict = dict(),
    ):
        super().__init__()

        dim_in = channels * patch_size ** 2

        self.pad_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.start_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.end_token = nn.Parameter(torch.randn(1, 1, dim_in))

        self.model = ArSpDiffusion(
            **model,
            window_size = window_size,
            sample_steps = sample_steps,
            dim_input = dim_in,
            num_classes = num_classes,
            sample_size = sample_size+1
        )

        self.to_tokens = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.to_image = Rearrange('b (h w) (p1 p2 c) -> b (h p1) (w p2) c', p1 = patch_size, p2 = patch_size, h=int(sqrt(sample_size)))


    def sample(
        self, 
        batch_size = 1,
        label = None,
    ):
        tokens = self.model.sample(label = label, batch_size = batch_size)
        images = self.to_image(tokens)
        return unnormalize_to_zero_to_one(images)

    def forward(
        self, 
        img,
        mask,
        offset,
        label
    ):
        patches = self.to_tokens(img)
        patches = normalize_to_neg_one_to_one(patches)

        patches_lookup = torch.concat([
            self.pad_token,
            self.start_token,
            self.end_token,
            patches
        ], dim=1).squeeze()

        tokens = patches_lookup[mask]
        return self.model(tokens, label, offset)
