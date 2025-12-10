# DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning

This is the source code for replicating the results from our papers accepted to ECAI 2025 (and GenPlan workshop at AAAI 2025), title [DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning](https://arxiv.org/abs/2507.20499). **This code is heavily inspired by the implementation of CleanDiffuser and ODRL. We thank the authors of two repos so much for their amazing work.**

Thank you for your interest!

## Setup
Before training, please install the following packages and libraries by running the following command
```
conda create --name dmc
conda activate dmc
pip install -r requirements.txt
```

## Training

For training, please do the following steps:
- Download the target datasets from the OTDF paper, and add the "ot" at the end of each dataset file. Put the dataset in both <diffusion/dataset/datasets> and <offdynamics/datasets/>
- Then, generate the domain gap scores, run the script <diffusion/DmC/compute_score.sh>
- Next, train the diffusion model and generate the dataset, run the script <diffusion/DmC/train_synther_cond_score_knn_4.sh>
- Process the generated dataset, run the Python file gather_gen_data_and_score.py under the results folder.
- Finally, train the offline cross-domain RL with an augmented dataset, run the script <offdynamics/train_DmC.sh>

## Citation
If you find our code helpful or utilise our proposed method as comparison baselines in your experiments, please cite our paper. Again, thank you for your interest!

```
@misc{van2025dmcnearestneighborguidance,
      title={DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning}, 
      author={Linh Le Pham Van and Minh Hoang Nguyen and Duc Kieu and Hung Le and Hung The Tran and Sunil Gupta},
      year={2025},
      eprint={2507.20499},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.20499}, 
}
```



