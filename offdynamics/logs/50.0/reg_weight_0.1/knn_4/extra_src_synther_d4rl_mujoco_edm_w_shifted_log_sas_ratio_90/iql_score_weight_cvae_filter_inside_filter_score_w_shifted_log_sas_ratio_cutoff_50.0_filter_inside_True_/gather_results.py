import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm


# reference scores for all dynamics shift tasks

REF_MIN_SCORE = {
    'halfcheetah-kinematic' : -280.178953 ,
    'halfcheetah-morph' : -280.178953 ,
    'halfcheetah-gravity' : -280.178953 ,
    'hopper-kinematic' : -26.3360015397715 ,
    'hopper-morph' : -26.3360015397715 ,
    'hopper-gravity' : -26.3360015397715 ,
    'walker2d-kinematic' : 10.079455055289959 ,
    'walker2d-morph' : 10.079455055289959 ,
    'walker2d-gravity' : 10.079455055289959 ,
    'ant-kinematic' : -325.6 ,
    'ant-morph' : -325.6 ,
    'ant-gravity' : -325.6 ,
}

REF_MAX_SCORE = {
    'halfcheetah-kinematic' : 7065.03 ,
    'halfcheetah-morph' : 9713.59 ,
    'halfcheetah-gravity' : 9509.15 ,
    'hopper-kinematic' : 2842.73 ,
    'hopper-morph' : 3152.75 ,
    'hopper-gravity' : 3234.3 ,
    'walker2d-kinematic' : 3257.51 ,
    'walker2d-morph' : 4398.43 ,
    'walker2d-gravity' : 5194.713 ,
    'ant-kinematic' : 5122.57 ,
    'ant-morph' : 5722.01 ,
    'ant-gravity' : 4317.065 ,
}

def get_normalized_score(env_name, score):
    # print(f'env_name: {env_name}, score: {score}')
    ref_min_score = REF_MIN_SCORE[env_name]
    ref_max_score = REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score) * 100

root_dir = './'
sub_path = 'csv/log_csv/scalars/test/'

for folder_name in tqdm(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    target_folder_data = []
    src_folder_data = []
    target_folder_normalized = []
    logging_normalize_score = []
    training_normalized_score = []
    skip = False
    print(f'Processing folder {folder_name}')
    if not os.path.isdir(folder_path):
        continue
    for random_seed in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, random_seed)):
            continue
        random_seed_path = os.path.join(folder_path, random_seed)
        target_return_path = os.path.join(random_seed_path, f'{sub_path}target return.csv')
        src_return_path = os.path.join(random_seed_path, f'{sub_path}source return.csv')
        normalized_score_path = os.path.join(random_seed_path, f'{sub_path}target normalized score.csv')
        if os.path.exists(target_return_path) and os.path.exists(src_return_path) and os.path.exists(normalized_score_path):
            # try:
            target_return = pd.read_csv(target_return_path, index_col=0)
            src_return = pd.read_csv(src_return_path, index_col=0)
            normalized_score = pd.read_csv(normalized_score_path, index_col=0)
            target_values = target_return.values[:,0]
            # print(target_values)
            env_name = folder_name.split('-srcdatatype')[0]
            # print(env_name)
            target_values_normalized = get_normalized_score(env_name, target_values)
            training_normalized_score.append(target_values_normalized)
            # print(target_values_normalized)
            # print(normalized_score.values[:,0])
            # print("---")
            # exit(0)
            src_values = src_return.values[:,0]
            target_folder_data.append(max(target_values))
            target_folder_normalized.append(max(target_values_normalized))
            src_folder_data.append(max(src_values))
            logging_normalize_score.append(max(normalized_score.values[:,0]))
            # except:
            #     print(f'Error in {folder_name}')
            #     continue
    
        else:
            print('File not found: ', target_return_path, src_return_path)
            print(f'We skip this folder {folder_name} due to the lack of information with random seed {random_seed}')
            skip = False
            continue
    # if skip:
    #     continue
    # print(logging_normalize_score, target_folder_normalized, target_folder_data, src_folder_data)


    # try:
    src_folder_data = np.array(src_folder_data)
    target_folder_data = np.array(target_folder_data)
    target_folder_normalized = np.array(target_folder_normalized)
    logging_normalize_score = np.array(logging_normalize_score)
    training_normalized_score = np.array(training_normalized_score)
    print(logging_normalize_score, target_folder_normalized, target_folder_data, src_folder_data)
    target_folder_mean = np.mean(target_folder_data, axis=0)
    target_folder_std = np.std(target_folder_data, axis=0) 
    src_folder_mean = np.mean(src_folder_data, axis=0)
    src_folder_std = np.std(src_folder_data, axis=0)
    target_folder_normalized_mean = np.mean(target_folder_normalized, axis=0)
    target_folder_normalized_std = np.std(target_folder_normalized, axis=0)
    logging_normalize_score_mean = np.mean(logging_normalize_score, axis=0)
    logging_normalize_score_std = np.std(logging_normalize_score, axis=0)
    # except:
    #     print(f'Error in {folder_name}')
    #     continue
    # print(f'src_folder_data.shape: {src_folder_data.shape}, {src_folder_data}')
    # print(f'target_folder_data.shape: {target_folder_data.shape}, {target_folder_data}')
    # print(f'target_folder_mean.shape: {target_folder_mean.shape}, {target_folder_mean}')
    # print(f'target_folder_std.shape: {target_folder_std.shape}, {target_folder_std}')
    # print(f'src_folder_mean.shape: {src_folder_mean.shape}, {src_folder_mean}')
    # print(f'src_folder_std.shape: {src_folder_std.shape}, {src_folder_std}')
    # print(logging_normalize_score_mean - target_folder_normalized_mean) 

    # df = pd.DataFrame({'target_normalized_score': target_folder_normalized_mean, 'target_normalized_score_std': target_folder_normalized_std, 'target_mean': target_folder_mean, 'target_std': target_folder_std, 'src_mean': src_folder_mean, 'src_std': src_folder_std})
    # max_normalized_score_mean_idx = np.argmax(logging_normalize_score_mean)
    # max_normalized_score = logging_normalize_score_mean[max_normalized_score_mean_idx]
    # max_normalized_score_std = logging_normalize_score_std[max_normalized_score_mean_idx]
    max_score_file_name = os.path.join(folder_path, 'max_normalized_score.txt')
    with open(max_score_file_name, 'w') as f:
        f.write(f'target folder normalized score {target_folder_normalized_mean} +- {target_folder_normalized_std}\n')
        f.write(f'logging normalized score {logging_normalize_score_mean} +- {logging_normalize_score_std}\n')
        f.write(f'seeds {len(target_folder_normalized)}\n')
    
    training_normalized_score_mean = np.mean(training_normalized_score, axis=0)
    training_normalized_score_std = np.std(training_normalized_score, axis=0)
    df = pd.DataFrame({'training_normalized_score_mean': training_normalized_score_mean, 'training_normalized_score_std': training_normalized_score_std})
    df.to_csv(os.path.join(folder_path, 'training_normalized_score.csv'))
    # exit(0)
    # df.to_csv(os.path.join(folder_path, 'results.csv'))
