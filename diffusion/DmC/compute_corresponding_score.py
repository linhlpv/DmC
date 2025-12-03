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
import faiss 

from pathlib                              import Path
import h5py
from tqdm                                import tqdm
from tensorboardX                         import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def compute_score(sas_src, sas_tar, sa_src, sa_tar, args):
    assert sas_src.shape[0] == sa_src.shape[0]
    res = faiss.StandardGpuResources()
    d = sas_src.shape[1]  
    sas_tar_index_flat = faiss.IndexFlatL2(d)
    sas_src_index_flat = faiss.IndexFlatL2(d)
    sa_tar_index_flat = faiss.IndexFlatL2(sa_src.shape[1])
    sa_src_index_flat = faiss.IndexFlatL2(sa_src.shape[1])
    # make it into a gpu index
    sas_tar_gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, sas_tar_index_flat)
    sas_src_gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, sas_src_index_flat)
    sa_tar_gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, sa_tar_index_flat)
    sa_src_gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, sa_src_index_flat)
    start_time = time.time()
    sas_tar_gpu_index_flat.add(sas_tar)
    print('Time taken to add sas tar:', time.time()-start_time)
    start_time = time.time()
    sas_src_gpu_index_flat.add(sas_src)
    print('Time taken to add sas src:', time.time()-start_time)
    start_time = time.time()
    sa_tar_gpu_index_flat.add(sa_tar)
    print('Time to add sa tar:', time.time()-start_time)
    start_time = time.time()
    sa_src_gpu_index_flat.add(sa_src)
    print('Time to add sa src:', time.time()-start_time)

    k = args.num_nearest_neighbors
    print('Searching for the nearest neighbors of sas_src in sas_tar')
    start_time = time.time()
    sas_src_tar_D, sas_src_tar_I = sas_tar_gpu_index_flat.search(sas_src, k)
    print('Time taken to search:', time.time()-start_time)
    
    print('Searching for the nearest neighbors of sas_src in sas_src')
    start_time = time.time()
    sas_src_src_D, sas_src_src_I = sas_src_gpu_index_flat.search(sas_src, k)
    print(sas_src_src_D[:5])
    print('Time taken to search:', time.time()-start_time)



    print('Searching for the nearest neighbors of sa_src in sas_tar')
    start_time = time.time()
    sa_src_tar_D, sa_src_tar_I = sa_tar_gpu_index_flat.search(sa_src, k)
    print('Time taken to seach:', time.time()-start_time)

    print('Searching for the nearest neighbors of sa_src in sa_src')
    start_time = time.time()
    sa_src_src_D, sa_src_src_I = sa_src_gpu_index_flat.search(sa_src, k)



    sas_src_tar_D_k = sas_src_tar_D[:, args.kth_nearest_neighbors] # get the distance to the k-th nearest neighbor
    sas_src_src_D_k = sas_src_src_D[:, args.kth_nearest_neighbors] # get the distance to the k-th nearest

    sa_src_tar_D_k = sa_src_tar_D[:, args.kth_nearest_neighbors]
    sa_src_src_D_k = sa_src_src_D[:, args.kth_nearest_neighbors]



    # compute the log ratio of the k-th nearest
    if args.kth_nearest_neighbors == 0:
        log_sas_ratio = np.log(sas_src_tar_D_k)
        log_sa_ratio = np.log(sa_src_tar_D_k)
    else:
        log_sas_ratio = np.log(sas_src_tar_D_k) - np.log(sas_src_src_D_k)
        log_sa_ratio = np.log(sa_src_tar_D_k) - np.log(sa_src_src_D_k)

    if args.use_sa == "True":
        log_sas_ratio = log_sas_ratio - log_sa_ratio

    shifted_log_sas_ratio = log_sas_ratio - min(log_sas_ratio) # just shift the log_sas_ratio to 0, larger value means the further to the target dataset
    w_shifted_log_sas_ratio = 1 / (1 + shifted_log_sas_ratio) # the larger the value, the closer to the target dataset, norm to [0, 1]

    return  w_shifted_log_sas_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--tar_dir', type=str, default='<path-to-the-folder-dataset>/dataset/datasets')
    parser.add_argument('--src_env', type=str, default='halfcheetah-medium-v2', help='source environment')
    parser.add_argument('--tar_env', type=str, default='halfcheetah_gravity_0.5_expert', help='target environment')
    parser.add_argument('--num_nearest_neighbors', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--save_dir', type=str, default='./score_sas', help='save directory') # score_sas_and_sa
    parser.add_argument('--use_sa', default="False")
    
    parser.add_argument('--kth_nearest_neighbors', type=int, default=5, help='k-th nearest neighbors')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # save name
    score_save_name = f'{args.src_env}_to_{args.tar_env}_score.npz'

    # load the source and target datasets
    src_dataset = d4rl.qlearning_dataset(gym.make(args.src_env))
    tar_dataset_path = os.path.join(args.tar_dir, args.tar_env + '.hdf5')
    print(f'tar_dataset_path: {tar_dataset_path}')
    data_dict = {}
    with h5py.File(tar_dataset_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  
                data_dict[k] = dataset_file[k][:]
            except ValueError as e: 
                data_dict[k] = dataset_file[k][()]

    tar_dataset = data_dict
    print(f'src_dataset: {src_dataset.keys()}')
    print(f'tar_dataset: {tar_dataset.keys()}')
    src_sas = np.concatenate([src_dataset['observations'], src_dataset['actions'], src_dataset['next_observations']], axis=1)
    tar_sas = np.concatenate([tar_dataset['observations'], tar_dataset['actions'], tar_dataset['next_observations']], axis=1)
    src_sa  = np.concatenate([src_dataset['observations'], src_dataset['actions']], axis=1)
    tar_sa  = np.concatenate([tar_dataset['observations'], tar_dataset['actions']], axis=1)
    w_shifted_log_sas_ratio = compute_score(src_sas, tar_sas, src_sa, tar_sa, args)

    # Save the scores, then later we can load it during training the diffusion model
    save_path = os.path.join(f'{args.save_dir}/{args.kth_nearest_neighbors}', score_save_name)
    sa_save_path = os.path.join(f'{args.save_dir}/sa/{args.kth_nearest_neighbors}', score_save_name)
    if not os.path.exists(f'{args.save_dir}/{args.kth_nearest_neighbors}'):
        os.makedirs(f'{args.save_dir}/{args.kth_nearest_neighbors}')
    if not os.path.exists(f'{args.save_dir}/sa/{args.kth_nearest_neighbors}'):
        os.makedirs(f'{args.save_dir}/sa/{args.kth_nearest_neighbors}')
    scores_dict = {
        'w_shifted_log_sas_ratio': w_shifted_log_sas_ratio,
    }
    np.savez(save_path, **scores_dict)
