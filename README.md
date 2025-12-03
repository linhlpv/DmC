# DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning


To reproduce the results, please do the following steps:
- Download the target datasets from the OTDF paper, add the "ot" at the end of each dataset files. Put the dataset in both <diffusion/dataset/datasets> and <offdynamics/datasets/>
- Generate the domain gap scores, run the script <diffusion/DmC/compute_score.sh>
- Train the diffusion model and generate the dataset, run the script <iffusion/DmC/train_synther_cond_score_knn_4.sh>
- Then process the generated dataset, run the python file gather_gen_data_and_score.py under the results folder.
- Finally, train the offline cross domain RL with augmented dataset, run the script <offdynamics/train_DmC.sh>


**This code is heavily inspired by the implementation of CleanDiffuser and ODRL. We thank the authors of two repos so much for their amazing works.**