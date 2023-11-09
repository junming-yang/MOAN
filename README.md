# Overview
This is an official implementation of the offline model-based RL algorithm [MOAN](https://arxiv.org/abs/2309.02157) all by pytorch. 

# Dependencies
- MuJoCo 2.0
- Gym 0.22.0
- D4RL
- PyTorch 1.8+

# Usage
## Train
```
# for hopper-medium-replay-v2 task
python3 train.py --task "hopper-medium-replay-v2" --rollout-length 5 --seed 50 --d-coeff 0.1
# for halfcheetah-medium-replay-v2 task
python3 train.py --task "hopper-medium-replay-v2" --rollout-length 5 --seed 50 --d-coeff 0.1
# for walker2d-medium-replay-v2 task
python3 train.py --task "hopper-medium-replay-v2" --rollout-length 5 --seed 50 --d-coeff 0.1
```

# Reference
If you find this code useful, please reference in our paper:
```
@article{yang2023model,
  title={Model-based offline policy optimization with adversarial network},
  author={Yang, Junming and Chen, Xingguo and Wang, Shengyuan and Zhang, Bolei},
  journal={arXiv preprint arXiv:2309.02157},
  year={2023}
}
```