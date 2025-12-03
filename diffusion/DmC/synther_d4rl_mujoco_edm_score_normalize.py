import sys
# sys.path.append("/home/s223719687/python_project/DiffusionRL/CleanDiffuser/")

import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('<folder-path>/CleanDiffuser')

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion.edm import EDM
from cleandiffuser.nn_diffusion.resmlpdenoiser import ResidualMLPDenoiser
from cleandiffuser.utils import report_parameters, FreezeModules
from utils import set_seed

from cleandiffuser.nn_condition import BaseNNCondition, get_mask

import h5py
from tqdm import tqdm

from typing import Dict
from typing import Union, Dict, Callable, List

import torch.nn
from torch.utils.data import Dataset


class ScoreNNCondition(BaseNNCondition):
    """ Simple MLP NNCondition for value conditioning.
    
    value (bs, 1) -> ValueNNCondition -> embedding (bs, emb_dim)

    Args:
        emb_dim (int): Embedding dimension.
        dropout (float): Label dropout rate.
    
    Example:
        >>> value = torch.rand(32, 1)
        >>> condition = ValueNNCondition(emb_dim=64, dropout=0.25)
        >>> # If condition.training, embedding will be masked to be dummy condition 
        >>> # with label dropout rate 0.25.
        >>> embedding = condition(value) 
        >>> embedding.shape
        torch.Size([32, 64])
    """
    def __init__(self, emb_dim: int, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(1, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, emb_dim))
    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = get_mask(
            mask, (condition.shape[0],), self.dropout, self.training, condition.device)
        mask = at_least_ndim(mask, condition.dim())
        return self.mlp(condition) * mask


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys



def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")

class EmptyNormalizer:
    """ Empty Normalizer

    Does nothing to the input data.
    """

    def normalize(self, x: np.ndarray):
        return x

    def unnormalize(self, x: np.ndarray):
        return x


class GaussianNormalizer(EmptyNormalizer):
    """ Gaussian Normalizer

    Normalizes data to have zero mean and unit variance.
    For those dimensions with zero variance, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> normalizer = GaussianNormalizer(x_dataset, 1)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> normalizer = GaussianNormalizer(x_dataset, 2)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(self, X: np.ndarray, start_dim: int = -1):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.mean = np.mean(X, axis=axes)
        self.std = np.std(X, axis=axes)
        self.std[self.std == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        return (x - at_least_ndim(self.mean, ndim, 1)) / at_least_ndim(self.std, ndim, 1)

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        return x * at_least_ndim(self.std, ndim, 1) + at_least_ndim(self.mean, ndim, 1)


class BaseNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class Normalizer(BaseNormalizer):
    def __init__(
            self,
            dataset: torch.Tensor,
            eps: float = 1e-5,
            skip_dims: List[int] = [],
            target_std: float = 1.0,
            device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.register_buffer('mean', dataset.mean(dim=0).to(device))
        self.register_buffer('std', (dataset.std(dim=0) + eps).to(device))
        self.skip_dims = skip_dims
        if skip_dims:
            self.mean[skip_dims] = 0.0
            self.std[skip_dims] = 1.0
        self.target_std = target_std
        print('Means:', self.mean)
        print('Stds:', self.std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std * self.target_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.target_std * self.std + self.mean

    def reset(self, dataset: torch.Tensor, eps: float = 1e-5):
        self.mean = dataset.mean(dim=0)
        self.std = dataset.std(dim=0) + eps
        if self.skip_dims:
            self.mean[self.skip_dims] = 0.0
            self.std[self.skip_dims] = 1.0
        print('Means:', self.mean)
        print('Stds:', self.std)


class BaseDataset(Dataset):
    def get_normalizer(self, **kwargs):
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key1: T, Do1  # default key: state
                key2: T, Do2
            action: T, Da
            reward: T, 1
            info: 
        """
        raise NotImplementedError()
    
def make_inputs(
        dataset, modelled_terminals: bool = True,
) -> np.ndarray:
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs
    

class D4RLMuJoCoTDDataset(BaseDataset):
    """ **D4RL-MuJoCo Transition Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into transitions.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observation of shape (batch_size, o_dim)
    - batch["next_obs"]["state"], next observation of shape (batch_size, o_dim)
    - batch["act"], action of shape (batch_size, a_dim)
    - batch["rew"], reward of shape (batch_size, 1)
    - batch["tml"], terminal of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo TD dataset. Obtained by calling `d4rl.qlearning_dataset(env)`.
        normalize_reward: bool,
            Normalize the reward. Default is False.
        
        score_path: str,
            Path to the score file.
        score_type: str,
            Type of the score. Default is 'max_min_normalize_score' = exp(-max_min_normalize_score). 
            'weighted_shifted_log_sas_ratio': exp(-shifted_log_sas_ratio) 
            'w_shifted_log_sas_ratio': 1 / (1 + shifted_log_sas_ratio)

    Examples:
        >>> env = gym.make("halfcheetah-medium-expert-v2")
        >>> dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 17)
        >>> act = batch["act"]           # (32, 6)
        >>> rew = batch["rew"]           # (32, 1)
        >>> tml = batch["tml"]           # (32, 1)
        >>> next_obs = batch["next_obs"]["state"]  # (32, 17)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(self, dataset: Dict[str, np.ndarray], disable_terminal_norm: bool = True, skip_dims: List[int] = []):
        super().__init__()

        observations, actions, next_observations, rewards, terminals, scores = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["terminals"].astype(np.float32),
            dataset["scores"].astype(np.float32))

        print(f'Observations: {observations.shape}, Actions: {actions.shape}, Next Observations: {next_observations.shape}, Rewards: {rewards.shape}, Terminals: {terminals.shape}, Scores: {scores.shape}')

    
        
        inputs = make_inputs(dataset)
        
        event_dims = inputs.shape[1]
        if disable_terminal_norm:
            terminal_dim = event_dims - 1
            if terminal_dim not in skip_dims:
                skip_dims.append(terminal_dim)

        if skip_dims:
            print('Skip normalization for dims:', skip_dims)

        self.normalizers = Normalizer(torch.tensor(inputs), skip_dims=skip_dims)

        self.obs = torch.tensor(observations, dtype=torch.float32)
        self.act = torch.tensor(actions, dtype=torch.float32)
        self.rew = torch.tensor(rewards, dtype=torch.float32)[:, None]
        self.tml = torch.tensor(terminals, dtype=torch.float32)[:, None]
        self.next_obs = torch.tensor(next_observations, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.float32).view(-1, 1)
        # self.normalizers = {
        #     "state": GaussianNormalizer(observations)}
        # normed_observations = self.normalizers["state"].normalize(observations)
        # normed_next_observations = self.normalizers["state"].normalize(next_observations)

        # self.obs = torch.tensor(normed_observations, dtype=torch.float32)
        # self.act = torch.tensor(actions, dtype=torch.float32)
        # self.rew = torch.tensor(rewards, dtype=torch.float32)[:, None]
        # self.tml = torch.tensor(terminals, dtype=torch.float32)[:, None]
        # self.next_obs = torch.tensor(normed_next_observations, dtype=torch.float32)

        self.size = self.obs.shape[0]
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        data = {
            'obs': {
                'state': self.obs[idx], },
            'next_obs': {
                'state': self.next_obs[idx], },
            'act': self.act[idx],
            'rew': self.rew[idx],
            'tml': self.tml[idx], 
            'score': self.scores[idx],}

        return data


class SynthERD4RLMuJoCoTDDataset(D4RLMuJoCoTDDataset):
    def __init__(self, save_path, dataset, disable_terminal_norm: bool = True, skip_dims: List[int] = [], score_path: str = None, score_type: str = 'max_min_normalize_score'):
        super().__init__(dataset, disable_terminal_norm, skip_dims, score_path, score_type)

        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32)[:, None],
            dataset["terminals"].astype(np.float32)[:, None])

        extra_transitions = np.load(save_path + "extra_transitions.npy")
        extra_observations = extra_transitions[:, :self.o_dim]
        extra_actions = extra_transitions[:, self.o_dim:self.o_dim + self.a_dim].clip(-1., 1.)
        extra_rewards = extra_transitions[:, self.o_dim + self.a_dim]
        extra_next_observations = extra_transitions[:, self.o_dim + self.a_dim + 1:self.o_dim * 2 + self.a_dim + 1]
        extra_terminals = (extra_transitions[:, -1] > 0.5).astype(np.float32)

        actions = np.concatenate([actions, extra_actions], 0)
        rewards = np.concatenate([rewards, extra_rewards[:, None]], 0)
        terminals = np.concatenate([terminals, extra_terminals[:, None]], 0)

        # # Since synthER generates normalized observations, we do not need to normalize them.
        # normed_observations = np.concatenate([
        #     self.normalizers["state"].normalize(observations), extra_observations], 0)
        # normed_next_observations = np.concatenate([
        #     self.normalizers["state"].normalize(next_observations), extra_next_observations], 0)

        ## Cat the extra transitions to the original dataset, note that we do not normalize the input and the extra transitions
        ## are not normalized as well - they are unnormalized samples from the diffusion model
        observations = np.concatenate([observations, extra_observations], 0)
        next_observations = np.concatenate([next_observations, extra_next_observations], 0)

        self.obs = torch.tensor(observations)
        self.act = torch.tensor(actions)
        self.rew = torch.tensor(rewards)
        self.tml = torch.tensor(terminals)
        self.next_obs = torch.tensor(next_observations)

        self.size = self.obs.shape[0]


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, act_dim), nn.Tanh(), )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
        obs = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
        return self(obs).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1), )
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1), )

    def both(self, obs, act):
        q1, q2 = self.Q1(torch.cat([obs, act], -1)), self.Q2(torch.cat([obs, act], -1))
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class TD3BC:
    def __init__(
            self, obs_dim: int, act_dim: int,
            policy_noise: float = 0.2, noise_clip: float = 0.5,
            policy_freq: int = 2, alpha: float = 2.5,
            device: str = "cpu"):

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor_target = deepcopy(self.actor).requires_grad_(False).eval().to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False).eval().to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optim, T_max=1000_000)
        self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optim, T_max=1000_000)

        self.policy_noise, self.noise_clip, self.policy_freq, self.alpha = (
            policy_noise, noise_clip, policy_freq, alpha)

        self.device = device

    def ema_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * 0.995 + param.data * 0.005)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * 0.995 + param.data * 0.005)

    def update(self, obs, act, rew, next_obs, tml, update_actor: bool = False):

        log = {}

        with torch.no_grad():

            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs) + noise).clamp(-1., 1.)

            target_q = rew + (1. - tml) * 0.99 * self.critic_target(next_obs, next_act)

        current_q1, current_q2 = self.critic.both(obs, act)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.critic_lr_scheduler.step()

        log["max_target_q"] = target_q.max().item()
        log["min_target_q"] = target_q.min().item()
        log["mean_target_q"] = target_q.mean().item()
        log["critic_loss"] = critic_loss.item()

        if update_actor:

            with FreezeModules([self.critic, ]):
                pred_act = self.actor(obs)
                q = self.critic(obs, pred_act)
                lmbda = self.alpha / q.abs().mean().detach()

            policy_loss = -lmbda * q.mean()
            bc_loss = F.mse_loss(pred_act, act)

            actor_loss = policy_loss + bc_loss

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.actor_lr_scheduler.step()

            log["policy_loss"] = policy_loss.item()
            log["policy_q"] = q.mean().item()
            log["bc_loss"] = bc_loss.item()

            self.ema_update()

        else:
            log["policy_loss"] = 0.
            log["policy_q"] = 0.
            log["bc_loss"] = 0.

        return log

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])


@hydra.main(config_path="<path>/configs/synther/mujoco_edm_score_uniform_score", config_name="mujoco", version_base=None)
def pipeline(args):
    set_seed(args.seed)

    save_path = f'./{args.sub_folder}/{args.pipeline_name}_{args.score_type}_{args.cond_score}/{args.task.env_name}_to_{args.tar_env_name}/'

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Load scores ----------------------
    score_name = f'{args.task.env_name}_to_{args.tar_env_name}_score.npz'
    score_path = os.path.join(args.score_dir, score_name)
    print(f'Loading scores from {score_path}')
    scores_data = np.load(score_path)
    scores_data = dict(scores_data)
    score_data = scores_data[args.score_type]
    print(f'Loaded scores: {score_data}, with type: {args.score_type}')
    print(f'max score: {np.max(score_data)}, min score: {np.min(score_data)}, mean score: {np.mean(score_data)}, std score: {np.std(score_data)}')
    cond_score = np.percentile(score_data, 95)
    print(f'cond_score: {cond_score}, max score: {np.max(score_data)}, min score: {np.min(score_data)}')

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset['scores'] = score_data
    dataset = D4RLMuJoCoTDDataset(dataset, args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    default_x_dim = [obs_dim * 2 + act_dim + 2]

    # --------------- Network Architecture -----------------
    nn_diffusion = ResidualMLPDenoiser(d_in = obs_dim * 2 + act_dim + 2,
                                        mlp_width = 2048,
                                        num_layers = 6,
                                        learned_sinusoidal_cond = False,
                                        random_fourier_features = True,
                                        learned_sinusoidal_dim = 16,
                                        activation = "relu",
                                        layer_norm = False,
                                        cond_dim = None,
                                        emb_dim=128,
                                        timestep_emb_type="positional")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    nn_condition = ScoreNNCondition(emb_dim=256, dropout=0.25)
    

    # --------------- Diffusion Model Actor --------------------
    synther = EDM(
        nn_diffusion, nn_condition=nn_condition, optim_params={"lr": args.diffusion_learning_rate},
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device, default_x_shape=default_x_dim)

    # ---------------------- Diffusion Training ----------------------
    if args.mode == "train_diffusion":
        print(f'normalize :{dataset.normalizers}')
        print(f'normalize :mean  {dataset.normalizers.mean}, std {dataset.normalizers.std}, skip_dims {dataset.normalizers.skip_dims}')


        lr_scheduler = CosineAnnealingLR(synther.optimizer, T_max=args.diffusion_gradient_steps)

        synther.train()

        n_gradient_step = 0
        log = {"avg_diffusion_loss": 0.}


        ## comment if you only want to sample extra transitions
        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)
            score = batch["score"].to(args.device)

            x = torch.cat([obs, act, rew, next_obs, tml], -1)
            x = dataset.normalizers.normalize(x) # normalize the input

            log["avg_diffusion_loss"] += synther.update(x, condition=score)["loss"]
            lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_diffusion_loss"] /= args.log_interval
                print(log)
                log = {"avg_diffusion_loss": 0., "gradient_steps": 0}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                synther.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                synther.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

                
        # generate extra transitions with the size equal with the dataset size
        synther.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        synther.eval()
        default_x_dim = [obs_dim * 2 + act_dim + 2]

        ori_size = dataset.obs.shape[0]
        # syn_size = 5000000 - ori_size
        syn_size = ori_size
        syn_size = max(ori_size, 1000000)
        print('Generate ', syn_size)
        # cond_score = np.percentile(score_data, 95)
        # cond_score = np.percentile(score_data, 90)
        # cond_score = np.percentile(score_data, 80)
        
        cond_score = args.cond_score
        print(f'cond_score: {cond_score}, max score: {np.max(score_data)}, min score: {np.min(score_data)}')
        w_cfg = 1.2

        extra_transitions = []
        cond_scores_store = []
        prior = torch.zeros((100000, 2 * obs_dim + act_dim + 2)).to(args.device)
        # condition = torch.ones((100000, 1)).to(args.device) * cond_score
        for _ in tqdm(range(syn_size // 100000)):

            cond_percentiles = np.random.randint(cond_score, 100, size=100000)
            print(cond_percentiles, cond_percentiles.max(), cond_percentiles.min(), cond_percentiles.mean())
            cond_scores = np.percentile(score_data, cond_percentiles)
            condition = torch.tensor(cond_scores, dtype=torch.float).to(args.device).view(-1, 1)
            print(condition.shape, condition)

            syn_transitions, _ = synther.sample(
                prior, solver=args.solver, n_samples=100000, sample_steps=args.sampling_steps, use_ema=args.use_ema, condition_cfg=condition, w_cfg=w_cfg, temperature=1.0 )
            syn_transitions = dataset.normalizers.unnormalize(syn_transitions) # unnormalize the output
            extra_transitions.append(syn_transitions.cpu().numpy())
            cond_scores_store.append(cond_scores)
        
        cond_percentiles = np.random.randint(cond_score, 100, size=syn_size % 100000)
        cond_scores = np.percentile(score_data, cond_percentiles)
        condition = torch.tensor(cond_scores, dtype=torch.float).to(args.device).view(-1, 1)

        syn_transitions, _ = synther.sample(
            torch.zeros((syn_size % 100000, 2 * obs_dim + act_dim + 2)).to(args.device),
            n_samples=syn_size % 100000, sample_steps=args.sampling_steps, use_ema=args.use_ema, solver=args.solver,  condition_cfg=condition, w_cfg=w_cfg, temperature=1.0 )
        syn_transitions = dataset.normalizers.unnormalize(syn_transitions) # unnormalize the output
        extra_transitions.append(syn_transitions.cpu().numpy())
        cond_scores_store.append(cond_scores)
        extra_transitions = np.concatenate(extra_transitions, 0)
        cond_scores_store = np.concatenate(cond_scores_store, 0)
        # extra_transitions = np.concatenate([extra_transitions, extra_transitions.reshape(-1,1)], 1)

        np.save(save_path + "extra_transitions.npy", extra_transitions)
        np.save(save_path + "cond_scores.npy", cond_scores_store)
        print(f'Finish.')

    # ---------------------- Dataset Upsampling ----------------------------
    elif args.mode == "dataset_upsampling":
        default_x_dim = [obs_dim * 2 + act_dim + 2]
        synther.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        synther.eval()

        ori_size = dataset.obs.shape[0]
        # syn_size = 5000000 - ori_size
        syn_size = ori_size

        extra_transitions = []
        prior = torch.zeros((100000, 2 * obs_dim + act_dim + 2)).to(args.device)
        for _ in tqdm(range(syn_size // 100000)):
            syn_transitions, _ = synther.sample(
                prior, solver=args.solver, n_samples=100000, sample_steps=args.sampling_steps, use_ema=args.use_ema )
            syn_transitions = dataset.normalizers.unnormalize(syn_transitions) # unnormalize the output
            extra_transitions.append(syn_transitions.cpu().numpy())

        syn_transitions, _ = synther.sample(
            torch.zeros((syn_size % 100000, 2 * obs_dim + act_dim + 2)).to(args.device),
            n_samples=syn_size % 100000, sample_steps=args.sampling_steps, use_ema=args.use_ema, solver=args.solver )
        syn_transitions = dataset.normalizers.unnormalize(syn_transitions) # unnormalize the output
        extra_transitions.append(syn_transitions.cpu().numpy())
        extra_transitions = np.concatenate(extra_transitions, 0)

        np.save(save_path + "extra_transitions.npy", extra_transitions)
        print(f'Finish.')

    # --------------------- Train RL ------------------------
    elif args.mode == "train_rl":

        dataset = SynthERD4RLMuJoCoTDDataset(save_path, d4rl.qlearning_dataset(env), args.normalize_reward)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        td3bc = TD3BC(obs_dim, act_dim, device=args.device)

        n_gradient_step = 0
        log = {"avg_policy_loss": 0., "avg_bc_loss": 0., "avg_policy_q": 0., "avg_critic_loss": 0., "target_q": 0.}

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            _log = td3bc.update(obs, act, rew, next_obs, tml, bool(n_gradient_step % td3bc.policy_freq))

            log["avg_policy_loss"] += _log["policy_loss"]
            log["avg_bc_loss"] += _log["bc_loss"]
            log["avg_policy_q"] += _log["policy_q"]
            log["avg_critic_loss"] += _log["critic_loss"]
            log["target_q"] += _log["mean_target_q"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_policy_loss"] /= args.log_interval
                log["avg_bc_loss"] /= args.log_interval
                log["avg_policy_q"] /= args.log_interval
                log["avg_critic_loss"] /= args.log_interval
                log["target_q"] /= args.log_interval
                print(log)
                log = {"avg_policy_loss": 0., "avg_bc_loss": 0., "avg_policy_q": 0.,
                       "avg_critic_loss": 0., "target_q": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                td3bc.save(save_path + f"td3bc_ckpt_{n_gradient_step + 1}.pt")
                td3bc.save(save_path + f"td3bc_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step > args.rl_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        td3bc = TD3BC(obs_dim, act_dim, device=args.device)
        td3bc.load(save_path + f"td3bc_ckpt_{args.ckpt}.pt")
        td3bc.actor.eval()
        normalizer = dataset.normalizers["state"]

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        episode_rewards = []

        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # sample actions
                with torch.no_grad():
                    act = td3bc.actor(obs).cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()