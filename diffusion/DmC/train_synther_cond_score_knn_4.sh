#!/bin/bash

# HalfCheetah
python synther_d4rl_mujoco_edm_score_normalize.py mode=train_diffusion score_dir=<path>/score_sas_icml/4 sub_folder=results/uniform_score/cond_only_sas/knn_4  score_type=w_shifted_log_sas_ratio cond_score=0 task=halfcheetah-medium-replay-v2 batch_size=256 tar_env_name=halfcheetah_gravity_0.5_medium_expert

