#!/bin/bash

if (($1==1)); then
  echo "train hopper env"
  python3 train.py --task "hopper-random-v2" --rollout-length 5 --reward-penalty-coef 1 --seed 50 --d-coeff 0.1;

elif (($1 == 2)); then
  echo  "train halfcheetah env"
  python train.py --task "halfcheetah-random-v2" --rollout-length 5 --reward-penalty-coef 0.5 --seed 50 --d-coeff 0.05;

elif (($1 == 3)); then
  echo "train walker2d env"
  python train.py --task "walker2d-random-v2" --rollout-length 5 --reward-penalty-coef 1 --seed 50 --load-model True --d-coeff 0.05;
fi