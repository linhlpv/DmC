from typing import Optional
from typing import Union
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from .basic import DiffusionModel


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class EDMArchetecture(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,
            default_x_shape: Sequence[int] = [42],

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.dot_scale_s = None
        self.dot_sigma_s = None
        self.scale_s = None
        self.t_s = None
        self.sigma_s = None
        self.x_weight_s, self.D_weight_s = None, None
        self.sample_steps = None
        self.default_x_shape = default_x_shape
        print(f"default_x_shape: {default_x_shape}")

    def set_sample_steps(self, N: int):
        raise NotImplementedError

    def c_skip(self, sigma):
        raise NotImplementedError

    def c_out(self, sigma):
        raise NotImplementedError

    def c_in(self, sigma):
        raise NotImplementedError

    def c_noise(self, sigma):
        raise NotImplementedError

    def loss_weighting(self, sigma):
        raise NotImplementedError

    def sample_noise_distribution(self, N):
        raise NotImplementedError

    def sample_scale_distribution(self, N):
        raise NotImplementedError

    def D(self, x, sigma, condition=None, use_ema=False):
        """ Prepositioning in EDM """
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.model_ema["diffusion"] if use_ema else self.model["diffusion"]
        c_noise = at_least_ndim(c_noise.squeeze(), 1)
        return c_skip * x + c_out * F(c_in * x, c_noise, condition)

    
    def D_duc(self, x, sigma, condition=None, use_ema=False):
        batch, device = x.shape[0], x.device
        # print(f'batch size: {batch}, device: {device}')

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)
        
        # print(f'sigma shape: {sigma.shape}')
        
        padded_sigma = sigma.view(batch, *([1] * len(self.default_x_shape)))
        
        """ Prepositioning in EDM """
        c_skip, c_out, c_in, c_noise = self.c_skip_duc(padded_sigma), self.c_out_duc(padded_sigma), self.c_in_duc(padded_sigma), self.c_noise_duc(sigma)
        F = self.model_ema["diffusion"] if use_ema else self.model["diffusion"]
        c_noise = at_least_ndim(c_noise.squeeze(), 1)
        F_out = F(c_in * x, c_noise, condition)
        # print(f'c_skip shape: {c_skip.shape}, x shape: {x.shape}, c_out shape: {c_out.shape}, F_out shape: {F_out.shape}, c_in shape: {c_in.shape}, c_noise shape: {c_noise.shape}')
        return c_skip * x + c_out * F_out
        # return c_skip * x + c_out * F(c_in * x, c_noise, condition)

    # ---------------------------------------------------------------------------
    # Training

    def loss(self, x0, condition=None):
        sigma = self.sample_noise_distribution(x0.shape[0])
        sigma = at_least_ndim(sigma, x0.dim())
        eps = torch.randn_like(x0) * sigma * (1. - self.fix_mask)
        condition = self.model["condition"](condition) if condition is not None else None
        loss = (self.loss_weighting(sigma) * (self.D(x0 + eps, sigma, condition) - x0) ** 2)
        return (loss * self.loss_weight).mean()

    def update(self, x0, condition=None, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(x0, condition)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    def update_classifier(self, x0, condition):
        sigma = self.sample_noise_distribution(x0.shape[0])
        sigma = at_least_ndim(sigma, x0.dim())
        noise = self.c_noise(sigma)
        eps = torch.randn_like(x0) * sigma * (1. - self.fix_mask)
        log = self.classifier.update(x0 + eps, at_least_ndim(noise.squeeze(), 1), condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def dot_x(
            self, x, i, use_ema=False,
            # ----------------- CFG ----------------- #
            condition_vec_cfg=None,
            w_cfg: float = 0.0,
            # ----------------- CG ----------------- #
            condition_vec_cg=None,
            w_cg: float = 1.0,
    ):
        b = x.shape[0]
        sigma = at_least_ndim(self.sigma_s[i].repeat(b), x.dim())
        noise = self.c_noise(sigma)
        unscale = 1. / self.scale_s[i] * (1. - self.fix_mask) + self.fix_mask
        # ----------------- CFG ----------------- #
        with torch.no_grad():
            if w_cfg != 0.0 and w_cfg != 1.0:
                repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
                condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                D = self.D(
                    (x * unscale).repeat(*repeat_dim),
                    sigma.repeat(*repeat_dim),
                    condition_vec_cfg, use_ema)
                D = w_cfg * D[:b] + (1. - w_cfg) * D[b:]
            elif w_cfg == 0.0:
                D = self.D(x * unscale, sigma, None, use_ema)
            else:
                D = self.D(x * unscale, sigma, condition_vec_cfg, use_ema)
        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x * unscale,
                                                    at_least_ndim(noise.squeeze(), 1), condition_vec_cg)
            D = D + w_cg * self.scale_s[i] * (sigma ** 2) * grad
        else:
            log_p = None

        # do not change the fixed portion
        dot_x = self.x_weight_s[i] * x - self.D_weight_s[i] * D
        dot_x = dot_x * (1. - self.fix_mask)

        return dot_x, {"log_p": log_p}
    
    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            cond=None,
            use_ema=False,
            
            # ----------------- CFG ----------------- #
            w_cfg: float = 0.0,
    ):
        denoised = None
        # ----------------- CFG ----------------- #
        with torch.no_grad():
            if w_cfg != 0.0 and w_cfg != 1.0:
                repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
                cond = torch.cat([cond, torch.zeros_like(cond)], 0)
                denoised = self.D_duc(x.repeat(*repeat_dim),
                                sigma,
                                cond, use_ema)
                denoised = w_cfg * denoised[:x.shape[0]] + (1. - w_cfg) * denoised[x.shape[0]:]
            elif w_cfg == 0.0:
                denoised = self.D_duc(x, sigma, condition=None, use_ema=use_ema)
            elif w_cfg == 1.0:
                denoised = self.D_duc(x, sigma, condition=cond, use_ema=use_ema)

            denoised_over_sigma = (x - denoised) / sigma
        return denoised_over_sigma, {'log_p': None}

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            solver: str = "euler",
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
            # default_x_shape: Sequence[int] = [42],
            **kwargs
    ):
        """
        Sample from the diffusion model.
        ---
        Input:
            - prior: Optional[torch.Tensor] = None
                The known fixed portion of the input data. Should be in the shape of `(n_samples, *x_shape)`.
                Leave the unknown part as `0`. If `None`, which means `fix_mask` is `None`, the model takes no prior.

            - sample_steps: int = 5
                Number of sampling steps.

            - use_ema: bool = True
                Whether to use the EMA model. If `False`, you should `eval` the model.

            - solver: str = "euler"
                The solver to use. Can be either "euler" or "heun".

            - condition_cfg: Optional[torch.Tensor] = None
                The condition for the CFG. Should be in the shape of `(n_samples, *shape_of_nn_condition_input)`.
                If `None`, the model takes no condition.

            - mask_cfg: Optional[torch.Tensor] = None
                The mask for the CFG. Should be in the shape of `(n_samples, *shape_of_nn_condition_mask)`.
                Model will ignore the `mask_cfg==0` parts in `condition_cfg`. If `None`, the model takes no mask.

            - w_cfg: float = 0.0
                The weight for the CFG. If `0.0`, the model takes no CFG.

            - condition_cg: Optional[torch.Tensor] = None
                The condition for the CG. Should be in the shape of `(n_samples, 1)`.
                If `None`, the model takes no condition.

            - w_cg: float = 0.0
                The weight for the CG. If `0.0`, the model takes no CG.

            - preserve_history: bool = False
                Whether to preserve the history of the sampling process. If `True`, the model will return the history.

        Output:
            - xt: torch.Tensor
                The sampled data. Should be in the shape of `(n_samples, *x_shape)`.

            - log: dict
                The log of the sampling process. Contains the following keys:
                    - "sample_history": np.ndarray
                        The history of the sampling process. Should be in the shape of `(n_samples, N + 1, *x_shape)`.
                        If `preserve_history` is `False`, this key will not exist.
                    - "log_p": torch.Tensor
                        The log probability of the sampled data estimated by CG.
                        Should be in the shape of `(n_samples,)`.
        """

        # self.default_x_shape = default_x_shape
        

        model = self.model_ema if use_ema else self.model
        log, x_history = {}, None

        # print(f"condition_cfg shape {condition_cfg.shape}")
        condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None


        if sample_steps != self.sample_steps:
            self.set_sample_steps(sample_steps)

        N = self.sample_steps
        gammas = torch.where(
            (self.sigma_s >= self.S_tmin) & (self.sigma_s <= self.S_tmax),
            min(self.S_churn / self.sample_steps, math.sqrt(2) - 1),
            0.
        )
        
        sigmas_and_gammas = list(zip(self.sigma_s[:-1], self.sigma_s[1:], gammas[:-1]))
        # print(sigmas_and_gammas)
        
        # inputs are noise at the beginning
        init_sigma = self.sigma_s[0]
        shape = (n_samples, *self.default_x_shape)
        # print(f'shape: {shape}')

        if prior is None:
            xt = torch.randn(shape, device=self.device) * init_sigma
        else:
            xt = torch.randn_like(prior, device=self.device) * init_sigma
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, N + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()
        
        for i, (sigma, sigma_next, gamma) in enumerate(sigmas_and_gammas):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            # print(sigma, sigma_next, gamma)
                
            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            xt_hat = xt + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            denoised_over_sigma, log = self.score_fn(xt_hat, sigma_hat, cond=condition_vec_cfg, use_ema=use_ema, w_cfg=w_cfg)
            xt_next = xt_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            # print(f"xt shape: {xt.shape}, xt_next shape: {xt_next.shape}, xt_hat shape: {xt_hat.shape}, denoised_over_sigma shape: {denoised_over_sigma.shape}")


            
            if prior is not None:
                xt_next = xt_next * (1. - self.fix_mask) + prior * self.fix_mask
            
            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma, log = self.score_fn(xt_next, sigma_next, cond=condition_vec_cfg, use_ema=use_ema, w_cfg=w_cfg)
                xt_next = xt_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)
                if prior is not None:
                    xt_next = xt_next * (1. - self.fix_mask) + prior * self.fix_mask
                
            xt = xt_next
            if preserve_history:
                x_history[:, i + 1] = xt.cpu().numpy()
            
            
        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(
                    xt, at_least_ndim(self.c_noise(self.sigma_s[-1]).squeeze(), 1), condition_cg)
            log["log_p"] = logp

        return xt, log

    def sample_x(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            extra_sample_steps: int = 8,
            use_ema: bool = True,
            solver: str = "euler",
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
    ):
        raise NotImplementedError


class EDM(EDMArchetecture):
    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- EDM Params ------------------- #
            sigma_data: float = 1.0,
            sigma_min: float = 0.002,
            sigma_max: float = 80.,
            rho: float = 7.,
            P_mean: float = -1.2,
            P_std: float = 1.2,
            default_x_shape: Sequence[int] = [42],
            
            # ------------------- Gamma Params ------------------- #
            S_churn: float = 80,
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, default_x_shape, device)

        self.sigma_data = sigma_data
        self.sigma_min, self.sigma_max, self.rho = sigma_min, sigma_max, rho
        self.P_mean, self.P_std = P_mean, P_std
        self.S_churn, self.S_tmin, self.S_tmax, self.S_noise = S_churn, S_tmin, S_tmax, S_noise

    def set_sample_steps(self, N: int):
        self.sample_steps = N
        self.sigma_s = (self.sigma_max ** (1 / self.rho) + torch.arange(N, device=self.device) / (N-1) *
                        (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        self.sigma_s = F.pad(self.sigma_s, (0, 1), value=0.)  # last step is sigma value of 0.

    def c_skip(self, sigma): return self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)

    def c_out(self, sigma): 
        return sigma * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_in(self, sigma): return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma): return 0.25 * sigma.log()


    # derived preconditioning params - Table 1
    def c_skip_duc(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out_duc(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in_duc(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise_duc(self, sigma):
        return log(sigma) * 0.25

    def loss_weighting(self, sigma): return (self.sigma_data ** 2 + sigma ** 2) / ((sigma * self.sigma_data) ** 2)

    def sample_noise_distribution(self, N):
        log_sigma = torch.randn(N, device=self.device) * self.P_std + self.P_mean
        return log_sigma.exp()

    def sample_scale_distribution(self, N):
        return torch.ones(N, device=self.device)