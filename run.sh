#!/bin/bash

if (($1==1)); then
  echo "train hopper env"
  python3 train.py --task "halfcheetah-medium-replay-v2" --rollout-length 5 --reward-penalty-coef 1 --seed 30 --load-model True
elif (($1 == 2)); then
  echo  "train halfcheetah env"
  python train.py --task "halfcheetah-medium-expert-v2" --rollout-length 5 --reward-penalty-coef 1
elif (($1 == 3)); then
  echo "train walker2d env"
  python train.py --task "walker2d-medium-replay-v2" --rollout-length 1 --reward-penalty-coef 1
fi