import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils
# import faiss 

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter
from algo.utils                           import Logger


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs/")
    parser.add_argument("--policy", default="SAC", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    parser.add_argument('--score_dir', default="<path-to-the-folder>/datasets/mujoco/scores/4", help='path to the score directory')
    parser.add_argument('--weight_type', default="w_shifted_log_sas_ratio")
    parser.add_argument('--cut_off', default=25, type=float)
    parser.add_argument('--extra_src_dir', type=str, default='<path-to-the-folder>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90', help='path to the extra source directory')
    parser.add_argument('--knn', default=4, type=int)
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=12, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--comment', default=None, help='Comment about the methods, using when debugging or tunning hyperparams')
    parser.add_argument("--reg_weight", default=0.5, type=float)

    args = parser.parse_args()

    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments
    if domain == 'antmaze':
        src_env_name = src_env_name.split('-')[0]
        src_env_name += '-' + args.srctype + '-v0'
    elif domain == 'adroit':
        src_env_name = src_env_name.split('-')[0]
        src_env_name += '-' + args.srctype + '-v0'
    else:
        src_env_name += '-' + args.srctype + '-v2'
    src_env = None
    src_eval_env = gym.make(src_env_name)
    src_eval_env.seed(args.seed)
    

    tar_env = None
    tar_env_config = {
        'env_name': tar_env_name,
        'shift_level': args.shift_level,
    }
    tar_eval_env = call_env[domain](tar_env_config)
    tar_eval_env.seed(args.seed + 100)

    
    if args.mode not in [0,1,2,3, 9, 10, 12]:
        raise NotImplementedError # cannot support other modes
    
    policy_config_name = args.policy.lower()

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    # log path, we use logging with tensorboard
    policy_name = args.policy
    extra_src_dir_postfix = args.extra_src_dir.split('/')[-1]
    outdir = args.dir + f'/{args.cut_off}' + f'/reg_weight_{args.reg_weight}' + f'/knn_{str(args.knn)}' + '/' + f'extra_src_{extra_src_dir_postfix}' + '/' + policy_name + "_filter_score_" + f"{args.weight_type}" + '_cutoff_' + str(args.cut_off) + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + f'_reg_weight_{str(args.reg_weight)}' + '/r' + str(args.seed)

    # writer = SummaryWriter('{}/tb'.format(outdir))
    writer = Logger('{}'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))
    
    print(f"outdir: {outdir}")

    # seed all
    src_env.action_space.seed(args.seed) if src_env is not None else None
    tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard', 'ot']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
        'weight_type': args.weight_type, #update the weight type from args,
        'reg_weight': args.reg_weight, # for cvae regularization
        'cut_off': args.cut_off,
    })


    policy = call_algo(args.policy, config, args.mode, device)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))
        
        for item in vars(args).items():
            f.write('\n args params {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size=5000000)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # in case that the domain is offline, we directly load its offline data
    score_file_name = f'{src_env_name}_to_{args.env.replace("-", "_")}_{str(args.shift_level)}_{args.tartype}_score.npz'
    score_path = os.path.join(args.score_dir, score_file_name)
    print(score_path)
    scores = np.load(score_path)
    scores = dict(scores)
    
    # load extra source data 
    extra_src_data_name = src_env_name + '_to_' + tar_env_name.replace('-', '_') + '_' + str(args.shift_level) + '_' + args.tartype.replace('-', "_") + 'extra_transitions.npy'
    extra_src_data_path = os.path.join(args.extra_src_dir, extra_src_data_name)
    print(f"extra_src_data_path: {extra_src_data_path}")

    # load d4rl dataset for src and do filtering based on scores
    src_dataset = d4rl.qlearning_dataset(src_eval_env)

    src_replay_buffer.convert_D4RL(src_dataset, scores=scores)
    src_replay_buffer.load_extra_dataset(extra_src_data_path, cond_score_in_extra_data=True, score_key=args.weight_type)
    

    tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
    tar_replay_buffer.convert_D4RL(tar_dataset)
    print(f"target dataset size {len(tar_dataset['rewards'])}")

    eval_cnt = 0

    if args.policy.lower() == 'iql_score_weight_cvae' and args.policy.lower() == 'iql_score_weight_cvae_filter_inside':
        if config['policy_regu']:
            policy.train_vae(tar_replay_buffer, config['batch_size'], writer)
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    # offline-offline learning
    for t in range(int(config['max_step'])):
        policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

        if (t + 1) % config['eval_freq'] == 0:
            src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
            writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
            eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
            print(f'normalized score: {eval_normalized_score}')
            writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)
            
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
    writer.close()