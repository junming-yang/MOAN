# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, obs_shape, act_shape, logger, offline_buffer, interval=20, lr=0.0005):
        super(Discriminator, self).__init__()
        self.observation_shape = obs_shape[0]
        self.action_size = act_shape
        self.z = 2 * self.observation_shape + self.action_size + 1
        self.interval = interval
        self.offline_buffer = offline_buffer
        self._learning_rate = lr
        self.logger = logger
        self.model = nn.Sequential(
            nn.Linear(self.z, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self._optim = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        self._criterion = torch.nn.BCELoss()
        self.cnt = 0

    def forward(self, next_state) -> torch.Tensor:
        return torch.clip(self.model(next_state), 0.1, 0.9)

    def rollout_offline_buffer(self,
                               batch_size: torch.Tensor
                               ) -> torch.Tensor:
        """
        return: Tensor shape[batch_size, cat_vector]
        """
        rollout_data = self.offline_buffer.sample(batch_size)
        obs = torch.Tensor(rollout_data["observations"])
        act = torch.Tensor(rollout_data["actions"])
        next_obs = torch.Tensor(rollout_data["next_observations"])
        rew = torch.Tensor(rollout_data["rewards"])
        delta_obs = next_obs - obs
        rollout_data = torch.cat([obs, act, delta_obs, rew], dim=1)
        return rollout_data

    def compute_loss(self,
                     model_input: torch.Tensor,
                     predictions: torch.Tensor,
                     groundtruths: torch.Tensor
                     ):
        """
        model_input: [obs_t, act_t]
        predictions shape: [ensemble_num, batch_size, next_obs + rew]
        """
        pre_mean, pre_var = predictions
        batch_size = model_input.shape[0]
        # expert = torch.cat([model_input, groundtruths], dim=1)
        expert = self.rollout_offline_buffer(batch_size)
        loss_sum = torch.tensor(0.0, dtype=torch.float32)
        loss_gen_sum = torch.tensor(0.0, dtype=torch.float32)
        for i in range(pre_mean.shape[0]):
            learner = torch.cat([model_input, pre_mean[i]], dim=1)
            real_loss = self._criterion(self.model(expert), Variable(torch.ones(batch_size, 1)))
            fake_loss = self._criterion(self.model(learner.detach()), Variable(torch.zeros(batch_size, 1)))
            g_loss = self._criterion(self.model(learner), Variable(torch.ones(batch_size, 1)))

            # record expert and learner var
            if self.cnt % 5 == 0:
                self.logger.record("var/model_expert", self.model(expert).detach().mean(), self.cnt, printed=False)
                self.logger.record("var/model_learner", self.model(learner).detach().mean(), self.cnt, printed=False)
            self.cnt += 1

            discriminator_loss = (fake_loss.mean() + real_loss.mean()) / 2
            generate_loss = g_loss.mean()
            loss_sum += discriminator_loss
            loss_gen_sum += generate_loss
        self.logger.record("loss/d_loss", loss_sum.mean(), self.cnt, printed=False)
        self.logger.record("loss/g_loss", loss_gen_sum.mean(), self.cnt, printed=False)
        return loss_sum, loss_gen_sum

    def update(self, loss):
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    @torch.no_grad()
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
        obs_t = torch.tensor(observations, dtype=torch.float32)
        act_t = torch.tensor(actions, dtype=torch.float32)
        obs_tp1 = torch.tensor(next_obs, dtype=torch.float32)
        rew_tp1 = torch.tensor(np.reshape(rewards, (-1, 1)), dtype=torch.float32)
        rew_p = torch.cat([obs_t, act_t, obs_tp1, rew_tp1], dim=1)
        d_penalty = self.forward(rew_p)
        d_penalty = 1 - d_penalty
        return d_penalty.detach().cpu().numpy()

    @property
    def get_interval(self):
        return self.interval
