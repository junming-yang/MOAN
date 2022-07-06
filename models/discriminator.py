# -*- coding: UTF-8 -*-
import random
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from d3rlpy.models.torch import ProbabilisticEnsembleDynamicsModel


class Discriminator(nn.Module):
    def __init__(self, obs_shape, act_shape, logger, interval=5, lr=0.0001):
        super(Discriminator, self).__init__()
        self.observation_shape = obs_shape
        self.action_size = act_shape
        self.interval = interval
        self._learning_rate = lr
        self.logger = logger
        self.model = nn.Sequential(
            nn.Linear(26, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self._optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._criterion = torch.nn.BCELoss()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.cnt = 0

    def forward(self, next_state) -> torch.Tensor:
        return self.model(next_state)

    def rollout_horizon(self) -> List:
        """
        rollout horizon from dataset

        return: rollout list [obs_t, act_t,obs_tp1, rew_tp1]
        """
        rand_epi = random.randint(0, len(self.dataset.episodes) - 1)
        rand_state = random.randint(0, len(self.dataset.episodes[rand_epi].__dict__["_observations"]) - 2)
        obs_t = torch.tensor(self.dataset.episodes[rand_epi].__dict__["_observations"][rand_state], dtype=torch.float32,
                             device=self.device)
        act_t = torch.tensor(self.dataset.episodes[rand_epi].__dict__["_actions"][rand_state], dtype=torch.float32,
                             device=self.device)
        obs_tp1 = torch.tensor(self.dataset.episodes[rand_epi].__dict__["_observations"][rand_state + 1],
                               dtype=torch.float32, device=self.device)
        rew_tp1 = torch.tensor([self.dataset.episodes[rand_epi].__dict__["_rewards"][rand_state]], dtype=torch.float32,
                               device=self.device)
        return [obs_t, act_t, obs_tp1, rew_tp1]

    def compute_error(self,
                      dynamics: Optional[ProbabilisticEnsembleDynamicsModel],
                      obs_t: torch.Tensor,
                      act_t: torch.Tensor,
                      rew_tp1: torch.Tensor,
                      obs_tp1: torch.Tensor,
                      ) -> torch.Tensor:
        pre_obs, pre_rew, var = dynamics.predict_with_variance(obs_t, act_t)
        # print(pre_obs.shape, pre_rew.shape, var.shape)  # 多了个5
        cat_shape = 2 * obs_t.shape[1] + act_t.shape[1] + 1
        expert = torch.cat(self.rollout_horizon())
        for i in range(1, obs_t.shape[0]):
            rollout = torch.cat(self.rollout_horizon())
            expert = torch.cat((expert, rollout), 0)
        expert = expert.reshape(obs_t.shape[0], cat_shape)
        loss_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
        loss_gen_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
        for i in range(pre_obs.shape[1]):
            learner = torch.cat([obs_t, act_t, pre_obs[:, i:i + 1, :].squeeze(), torch.squeeze(pre_rew[:, i:i + 1], 1)],
                                dim=1)
            real_loss = self._criterion(self.model(expert), torch.ones((100, 1), device=obs_t.device))
            fake_loss = self._criterion(self.model(learner.detach()), torch.zeros((100, 1), device=obs_t.device))
            g_loss = self._criterion(self.model(learner), torch.ones((100, 1), device=obs_t.device))
            if i % 3 == 0 and i > 0 and self.save_var:
                self.logger.record("var/var", var.detach().mean(), self.cnt, printed=False)
                self.logger.record("var/model_expert", self.model(expert).detach().mean(), self.cnt, printed=False)
                self.logger.record("var/model_learner", self.model(learner).detach().mean(), self.cnt, printed=False)
                self.cnt += 1
            discriminator_loss = fake_loss.mean() + real_loss.mean()
            generate_loss = g_loss.mean()
            loss_sum += discriminator_loss
            loss_gen_sum += generate_loss
        return loss_sum, loss_gen_sum

    def update(self, loss):
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def compute_penalty(self,
                        observations: np.ndarray,
                        actions: np.ndarray,
                        next_obs: np.ndarray,
                        rewards: np.ndarray,
                        ) -> np.ndarray:
        """
        Penalty for mopo_algo

        args:
                    obs_t: np.ndarray,
                    act_t: np.ndarray,
                    rew_tp1: np.ndarray,
                    obs_tp1: np.ndarray,

        return: penalty factor d_penalty
        """
        obs_t = torch.tensor(observations)
        act_t = torch.tensor(actions)
        obs_tp1 = torch.tensor(next_obs)
        rew_tp1 = torch.tensor(rewards)
        rew_p = torch.cat([obs_t, act_t, obs_tp1, rew_tp1], dim=1).to('cpu')
        d_penalty = self.model(rew_p)
        d_penalty = 1 - d_penalty
        d_penalty = d_penalty.detach().cpu().numpy()
        return d_penalty

    @property
    def get_interval(self):
        return self.interval
