import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

from torch.nn.modules.dropout import Dropout
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from tensorboardX                         import SummaryWriter
from torchrl.record import CSVLogger
# from torchtnt.utils.loggers import CSVLogger as csvlogger



class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter('{}/tb'.format(log_dir))
        self.logger = CSVLogger(log_dir='{}/csv'.format(log_dir), exp_name='log_csv')
        # self.csv_logger = csvlogger(path='{}/csv_logger'.format(log_dir))


    def add_dict(self, log_dict, global_step):
        for key, value in log_dict.items():
            self.writer.add_scalar(key, value, global_step)
            self.logger.log_scalar(key, value, global_step)
            # self.csv_logger.log(key, value, step)

    def add_scalar(self, key, value, global_step):
        self.writer.add_scalar(key, value, global_step)
        self.logger.log_scalar(key, value, global_step)
        # self.csv_logger.log(key, value, step)

    def close(self):
        self.writer.close()
        self.logger.close()



# Modify to add the mc_return to the replay buffer. That is used in Cal-QL algorithm 
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state_dim =    state_dim
        self.action_dim =   action_dim

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.mc_return = np.zeros((max_size, 1))
        self.modified_reward = np.zeros((max_size, 1))

        self.scores = {}

        self.device = device

    def add(self, state, action, next_state, reward, done, modified_reward=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.mc_return[self.ptr] = 0.0 # set return to goal for online sample is 0 as original implementation
        if modified_reward is not None:
            self.modified_reward[self.ptr] = modified_reward # modified reward is used for decouple policy methods

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    # used for method using RND bonus
    def get_moments(self):
        state_mean, state_std = self.state.mean(0), self.state.std(0)
        action_mean, action_std = self.action.mean(0), self.action.std(0)

        return (torch.from_numpy(state_mean), torch.from_numpy(state_std)), (torch.from_numpy(action_mean), torch.from_numpy(action_std))


    def sample(self, batch_size, modified_reward=False, mc_return=False, score_weight_type=None):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        if mc_return :
            if modified_reward:
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    torch.FloatTensor(self.mc_return[ind]).to(self.device),
                    torch.FloatTensor(self.modified_reward[ind]).to(self.device)
                )
            else:
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    torch.FloatTensor(self.mc_return[ind]).to(self.device)
                )
        else:
            if modified_reward:
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    torch.FloatTensor(self.modified_reward[ind]).to(self.device)
                )
            else:
                if score_weight_type is not None:
                    return (
                        torch.FloatTensor(self.state[ind]).to(self.device),
                        torch.FloatTensor(self.action[ind]).to(self.device),
                        torch.FloatTensor(self.next_state[ind]).to(self.device),
                        torch.FloatTensor(self.reward[ind]).to(self.device),
                        torch.FloatTensor(self.not_done[ind]).to(self.device),
                        torch.FloatTensor(self.scores[score_weight_type][ind]).to(self.device)
                    )
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device)
                )

    def get_data(self, batch_size, ind, modified_reward=False, mc_return=False, score_weight_type=None):
        if mc_return :
            if modified_reward:
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    torch.FloatTensor(self.mc_return[ind]).to(self.device),
                    torch.FloatTensor(self.modified_reward[ind]).to(self.device)
                )
            else:
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    torch.FloatTensor(self.mc_return[ind]).to(self.device)
                )
        else:
            if modified_reward:
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    torch.FloatTensor(self.modified_reward[ind]).to(self.device)
                )
            else:
                if score_weight_type is not None:
                    return (
                        torch.FloatTensor(self.state[ind]).to(self.device),
                        torch.FloatTensor(self.action[ind]).to(self.device),
                        torch.FloatTensor(self.next_state[ind]).to(self.device),
                        torch.FloatTensor(self.reward[ind]).to(self.device),
                        torch.FloatTensor(self.not_done[ind]).to(self.device),
                        torch.FloatTensor(self.scores[score_weight_type][ind]).to(self.device)
                    )
                return (
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device)
                )
    
    def convert_D4RL(self, dataset, mc_return=False, score_path=None, scores=None, size=None):
        if size is not None:
            self.state = dataset['observations'][:size]
            self.action = dataset['actions'][:size]
            self.next_state = dataset['next_observations'][:size]
            self.reward = dataset['rewards'][:size].reshape(-1,1)
            self.not_done = 1. - dataset['terminals'][:size].reshape(-1,1)
        else:
            self.state = dataset['observations']
            self.action = dataset['actions']
            self.next_state = dataset['next_observations']
            self.reward = dataset['rewards'].reshape(-1,1)
            self.not_done = 1. - dataset['terminals'].reshape(-1,1)

        print(f"state shape: {self.state.shape}, action shape: {self.action.shape}, reward shape: {self.reward.shape}, next_state shape: {self.next_state.shape}, done shape: {self.not_done.shape}")
        self.size = self.state.shape[0]
        self.ptr = (self.ptr + dataset['observations'].shape[0]) % self.max_size
        if mc_return:
            self.mc_return = dataset['mc_returns'].reshape(-1,1)
        
        if score_path is not None:
            scores = np.load(score_path)
            for key in scores.keys():
                self.scores[key] = scores[key].reshape(-1,1)
                if size is not None:
                    self.scores[key] = self.scores[key][:size]
                
        if scores is not None:
            for key in scores.keys():
                self.scores[key] = scores[key].reshape(-1,1)
                if size is not None:
                    self.scores[key] = self.scores[key][:size]

    
    def load_extra_dataset(self, savepath, cond_score_in_extra_data=False, generated_scores=None, score_key=None):
        extra_transitions = np.load(savepath)
        add_size = extra_transitions.shape[0]
        print(f'current replay buffer size: {self.size}')
        print(f'current pointer: {self.ptr}')
        print(f"Adding {add_size} extra transitions to the replay buffer")

        if cond_score_in_extra_data:
            assert extra_transitions.shape[1] == 2*self.state_dim + self.action_dim + 3 or extra_transitions.shape[1] == 2*self.state_dim + self.action_dim + 4, "Extra data should have the score as the last column or two last columns"

        extra_state = extra_transitions[:, :self.state_dim]
        extra_action = extra_transitions[:, self.state_dim:self.state_dim+self.action_dim].clip(-1., 1.)
        extra_reward = extra_transitions[:, self.state_dim + self.action_dim]
        extra_next_state = extra_transitions[:, self.state_dim + self.action_dim + 1:2*self.state_dim + self.action_dim + 1]
        extra_done = (extra_transitions[:, 2*self.state_dim + self.action_dim + 1: 2*self.state_dim + self.action_dim + 2] > 0.5).astype(np.float32)
        extra_not_done = 1. - extra_done

        if cond_score_in_extra_data:
            scores = extra_transitions[:, 2*self.state_dim + self.action_dim + 2]
            print(f'loaded scores shape: {scores.shape}, scores: {scores}')
        else:
            if not isinstance(generated_scores, np.ndarray) and generated_scores is not None:
                scores = np.ones_like(extra_reward) * generated_scores
        
        self.state = np.concatenate([self.state, extra_state], axis=0)
        self.action = np.concatenate([self.action, extra_action], axis=0)
        self.reward = np.concatenate([self.reward, extra_reward.reshape(-1,1)], axis=0)
        self.next_state = np.concatenate([self.next_state, extra_next_state], axis=0)
        self.not_done = np.concatenate([self.not_done, extra_not_done.reshape(-1,1)], axis=0)
        
        if score_key is not None:
            print(f"scores shape {scores.shape}")
            self.scores[score_key] = np.concatenate([self.scores[score_key], scores.reshape(-1,1)], axis=0) ## only add the score w.r.t the key to the scores dict

            print(f'after concat with extra data, scores shape: {self.scores[score_key].shape}, state shape: {self.state.shape}, action shape: {self.action.shape}, reward shape: {self.reward.shape}, next_state shape: {self.next_state.shape}, done shape: {self.not_done.shape}')

        # self.ptr = (self.ptr + add_size) % self.max_size
        self.size = min(self.size + add_size, self.max_size)
        print(f'current replay buffer size: {self.size}')

        

class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affines.append(nn.Linear(hidden_dim, out_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)
            self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
                    # x = self.norm_layer(x)
        return x

def identity(x):
    return x

def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def orthogonal_init(tensor, gain=0.01):
    torch.nn.init.orthogonal_(tensor, gain=gain)

class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            final_init_scale=None,
            dropout_rate=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]
        
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            # add dropout
            if self.dropout_rate:
                h = self.dropout(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output
    
    def sample(self, *inputs):
        preds = self.forward(*inputs)

        sample_idxs = np.random.choice(self.ensemble_size, 2, replace=False)
        preds_sample = preds[sample_idxs]
        
        return torch.min(preds_sample, dim=0)[0], sample_idxs



def get_return_to_go(dataset, env, config):
    print(env)
    returns = []
    ep_ret, ep_len = 0.0, 0
    cur_rewards = []
    terminals = []
    N = len(dataset["rewards"])
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len += 1
        is_last_step = (
            (t == N - 1)
            or (
                np.linalg.norm(
                    dataset["observations"][t + 1] - dataset["next_observations"][t]
                )
                > 1e-6
            )
            or ep_len == env._max_episode_steps
        )

        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0
            if (
                config['is_sparse_reward']
                and r
                == env.ref_min_score * config['reward_scale'] + config['reward_bias']
            ):
                discounted_returns = [r / (1 - config['discount'])] * ep_len
            else:
                for i in reversed(range(ep_len)):
                    discounted_returns[i] = cur_rewards[
                        i
                    ] + config['discount'] * prev_return * (1 - terminals[i])
                    prev_return = discounted_returns[i]
            returns += discounted_returns
            ep_ret, ep_len = 0.0, 0
            cur_rewards = []
            terminals = []
    return returns