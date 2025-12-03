import os
import sys
import numpy as np
import sys

root_dir = './'
empty_folders = []

for folder_name in os.listdir(root_dir):
    if not os.path.isdir(os.path.join(root_dir, folder_name)):
        continue


    folder_path = os.path.join(root_dir, folder_name)
    data_path = os.path.join(folder_path, 'extra_transitions.npy')
    cond_score_path = os.path.join(folder_path, 'cond_scores.npy')
    if not os.path.exists(data_path) or not os.path.exists(cond_score_path):
        print(f'No data or cond_score for {folder_name}')
        empty_folders.append(folder_name)
        continue
    data = np.load(data_path)
    cond_scores = np.load(cond_score_path)
    if len(cond_scores.shape) == 1:
        cond_scores = cond_scores.reshape(-1, 1)
    combined_data = np.concatenate([data, cond_scores], axis=1)
    save_name = f'{folder_name}extra_transitions.npy'
    save_path = os.path.join(folder_path, save_name)
    np.save(save_path, combined_data)
    print(f'Saved to {save_path}')

# with open('empty_folders.txt', 'w') as f:
#     for folder in empty_folders:
#         f.write(folder + '\n')