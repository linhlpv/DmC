#!/bin/bash

src_envs=("ant-medium-v2" "ant-medium-expert-v2" "ant-medium-replay-v2")
tar_envs=("ant_gravity_0.5_expert" "ant_gravity_0.5_medium" "ant_gravity_0.5_medium_expert")

# 4 is 5 due to zero indexing
for src_env in "${src_envs[@]}"; do
  for tar_env in "${tar_envs[@]}"; do
    echo "Running compute_corresponding_score.py with src_env=$src_env and tar_env=$tar_env" 
    python compute_corresponding_score.py --src_env "$src_env" --tar_env "$tar_env" --kth_nearest_neighbors 4 --save_dir ./score_sas
  done
done

