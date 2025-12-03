######################### MUJOCO gravity
############ medium target   

## Ant 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 

## hopper 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## walker2d 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## halfcheetah 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




############ expert target   

## Ant 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## hopper 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## walker2d 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## halfcheetah 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



############ medium-expert target   

## Ant 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## hopper 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## walker2d 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## halfcheetah 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-gravity  --shift_level 0.5 --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 





######################### MUJOCO kinematic
############ medium target   

## Ant 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## hopper 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## walker2d 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 


## halfcheetah 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



############ expert target   

## Ant 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## hopper 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## walker2d 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 





## halfcheetah 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




############ medium-expert target   

## Ant 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env ant-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 




## hopper 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env hopper-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 



## walker2d 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env walker2d-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 





## halfcheetah 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 1  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90   
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 2  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 3  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 4  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90  
# python train_DmC.py --policy  iql_score_weight_cvae_filter_inside --cut_off 50 --env halfcheetah-kinematic  --shift_level ot --srctype medium-expert --tartype medium_expert --seed 5  --mode 12 --knn 4 --dir ./logs/full_exps --reg_weight 0.1 --extra_src_dir <path-to-project>/datasets/mujoco/extra_transitions/knn_4/synther_d4rl_mujoco_edm_w_shifted_log_sas_ratio_90 





