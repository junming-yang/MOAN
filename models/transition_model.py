import numpy as np
import torch
import os
import sys
sys.path.append("..")
from common import util, functional

from models.ensemble_dynamics import EnsembleModel
from operator import itemgetter
from common.normalizer import StandardNormalizer
from copy import deepcopy

#from matplotlib import pyplot as plt
#from sklearn import manifold

#import util from common

class TransitionModel:
    def __init__(self,
                 obs_space,
                 action_space,
                 static_fns,
                 lr,
                 discriminator,
                 d_coeff=0,
                 d_penalty=True,
                 holdout_ratio=0.1,
                 inc_var_loss=False,
                 use_weight_decay=False,
                 **kwargs):

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.device = util.device
        self.model = EnsembleModel(obs_dim=obs_dim, action_dim=action_dim, device=util.device, **kwargs['model'])
        self.static_fns = static_fns
        self.lr = lr

        self.discriminator = discriminator
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.networks = {
            "model": self.model
        }
        self.d_penalty = d_penalty
        self.d_coeff = d_coeff
        self.obs_space = obs_space
        self.holdout_ratio = holdout_ratio
        self.inc_var_loss = inc_var_loss
        self.use_weight_decay = use_weight_decay
        self.obs_normalizer = StandardNormalizer()
        self.act_normalizer = StandardNormalizer()
        self.model_train_timesteps = 0
        self.update_count = 0
        self.ad_update_count = 0
        self.coeff = 0.9

    @torch.no_grad()
    def eval_data(self, data, update_elite_models=False):
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data)
        obs_list = torch.Tensor(obs_list)
        action_list = torch.Tensor(action_list)
        next_obs_list = torch.Tensor(next_obs_list)
        reward_list = torch.Tensor(reward_list)
        delta_obs_list = next_obs_list - obs_list
        obs_list, action_list = self.transform_obs_action(obs_list, action_list)
        model_input = torch.cat([obs_list, action_list], dim=-1).to(util.device)
        predictions = functional.minibatch_inference(args=[model_input], rollout_fn=self.model.predict,
                                                     batch_size=10000,
                                                     cat_dim=1)  # the inference size grows as model buffer increases
        groundtruths = torch.cat((delta_obs_list, reward_list), dim=1).to(util.device)
        eval_mse_losses, _ = self.model_loss(predictions, groundtruths, mse_only=True)
        if update_elite_models:
            elite_idx = np.argsort(eval_mse_losses.cpu().numpy())
            self.model.elite_model_idxes = elite_idx[:self.model.num_elite]
        return eval_mse_losses.detach().cpu().numpy(), None

    def reset_normalizers(self):
        self.obs_normalizer.reset()
        self.act_normalizer.reset()

    def update_normalizer(self, obs, action):
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(action)

    def transform_obs_action(self, obs, action):
        obs = self.obs_normalizer.transform(obs)
        action = self.act_normalizer.transform(action)
        return obs, action

    def update(self, data_batch):
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data_batch)
        obs_batch = torch.Tensor(obs_batch)
        action_batch = torch.Tensor(action_batch)
        next_obs_batch = torch.Tensor(next_obs_batch)
        reward_batch = torch.Tensor(reward_batch)

        delta_obs_batch = next_obs_batch - obs_batch
        obs_batch, action_batch = self.transform_obs_action(obs_batch, action_batch)

        # predict with model
        model_input = torch.cat([obs_batch, action_batch], dim=-1).to(util.device)
        predictions = self.model.predict(model_input)

        # compute training loss
        groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1).to(util.device)
        train_mse_losses, train_var_losses = self.model_loss(predictions, groundtruths)
        train_mse_loss = torch.sum(train_mse_losses)
        train_var_loss = torch.sum(train_var_losses)

        if self.update_count == 0:
            # adversarial
            self.discriminator.get_transform_obs_action(self.transform_obs_action)

        train_d_loss, train_g_loss = self.discriminator.compute_loss(model_input, predictions, groundtruths)
        # debug
        if self.update_count == 0:
            print("mse_loss:{}, var_loss:{}, d_loss:{}".format(train_mse_loss, train_var_loss, train_d_loss))

        train_transition_loss = train_mse_loss + train_var_loss + 1 * train_g_loss
        train_transition_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(
            self.model.min_logvar)  # why
        if self.use_weight_decay:
            decay_loss = self.model.get_decay_loss()
            train_transition_loss += decay_loss
        else:
            decay_loss = None

        # update transition model and discriminator
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()

        if 0 < self.update_count < 100000 and self.update_count % self.discriminator.get_interval == 0:
            self.discriminator.update(train_d_loss)
        #if self.update_count == 80000:
        #    self.coeff = 0.98

        self.model_optimizer.step()
        self.update_count += 1

        # compute test loss for elite model
        return {
            "loss/train_model_loss_mse": train_mse_loss.item(),
            "loss/train_model_loss_var": train_var_loss.item(),
            "loss/train_model_loss": train_var_loss.item(),
            "loss/decay_loss": decay_loss.item() if decay_loss is not None else 0,
            "misc/max_std": self.model.max_logvar.mean().item(),
            "misc/min_std": self.model.min_logvar.mean().item()
        }

    def adversarial_learning(self, data_batch, model_batch):
        # data_batch
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data_batch)
        obs_batch = torch.Tensor(obs_batch)
        action_batch = torch.Tensor(action_batch)
        next_obs_batch = torch.Tensor(next_obs_batch)
        reward_batch = torch.Tensor(reward_batch)

        delta_obs_batch = next_obs_batch - obs_batch
        obs_batch, action_batch = self.transform_obs_action(obs_batch, action_batch)

        # model_batch
        model_obs_batch, model_action_batch, model_next_obs_batch, model_reward_batch = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(model_batch)
        model_obs_batch = torch.Tensor(model_obs_batch)
        model_action_batch = torch.Tensor(model_action_batch)
        model_next_obs_batch = torch.Tensor(model_next_obs_batch)
        model_reward_batch = torch.Tensor(model_reward_batch)

        model_delta_obs_batch = model_next_obs_batch - model_obs_batch
        model_obs_batch, model_action_batch = self.transform_obs_action(model_obs_batch, model_action_batch)

        # predict with model
        data_input = torch.cat([obs_batch, action_batch], dim=-1)
        model_input = torch.cat([model_obs_batch, model_action_batch], dim=-1)
        # predictions = self.model.predict(model_input)
        # groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1)

        if self.ad_update_count == 0:
            for params in self.model_optimizer.param_groups:
                params['lr'] *= 0.1

        data_next_obs, data_rew = self.model.predict(data_input)
        model_next_obs, model_rew = self.model.predict(model_input)
        # data_output = torch.cat((data_next_obs, data_rew), dim=-1)
        # model_output = torch.cat((model_next_obs, model_rew), dim=-1)

        # print("shape:{}".format(data_next_obs.shape))  # [7, 256, 13]
        # train_d_loss, train_g_loss = self.discriminator.compute_loss(model_input, predictions, groundtruths)
        train_d_loss, train_g_loss = self.discriminator.compute_loss(data_next_obs, model_next_obs)

        self.model_optimizer.zero_grad()
        train_g_loss.backward(retain_graph=True)

        # update transition model and discriminator
        self.discriminator.update(train_d_loss)

        if self.ad_update_count % self.discriminator.get_interval == 0:
            self.model_optimizer.step()

        self.ad_update_count += 1

        return {
            "loss/train_g_loss": train_g_loss,
            "loss/train_d_loss": train_d_loss
        }

    def _compile_feature(self, data_input, model_input):
        for i in range(4):
            data_output = self.model.feature[i](data_input)
            model_output = self.model.feature[i](model_input)

            return data_output, model_output

    def model_loss(self, predictions, groundtruths, mse_only=False):
        pred_means, pred_logvars = predictions
        if self.inc_var_loss and not mse_only:
            # Average over batch and dim, sum over ensembles.
            inv_var = torch.exp(-pred_logvars)
            mse_losses = torch.mean(torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        elif mse_only:
            mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None
        else:
            assert 0
        return mse_losses, var_losses

    @torch.no_grad()
    def predict(self, obs, act, deterministic=False, penalty_coeff=0):
        """
        predict next_obs and rew
        return: next_obs, rewards, terminals, info
        """
        if len(obs.shape) == 1:
            obs = obs[None,]
            act = act[None,]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(util.device)

        scaled_obs, scaled_act = self.transform_obs_action(obs, act)

        model_input = torch.cat([scaled_obs, scaled_act], dim=-1).to(util.device)
        pred_diff_means, pred_diff_logvars = self.model.predict(model_input)
        pred_diff_means = pred_diff_means.detach().cpu().numpy()
        # add curr obs for next obs
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach().cpu().numpy()

        if deterministic:
            pred_diff_means = pred_diff_means
        else:
            pred_diff_means = pred_diff_means + np.random.normal(size=pred_diff_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = pred_diff_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

        next_obs, rewards = pred_diff_samples[:, :-1] + obs, pred_diff_samples[:, -1]
        terminals = self.static_fns.termination_fn(obs, act, next_obs)

        # penalty rewards
        penalty_learned_var = True
        if penalty_coeff != 0:
            if not penalty_learned_var:
                ensemble_means_obs = pred_diff_means[:, :, 1:]
                mean_obs_means = np.mean(ensemble_means_obs, axis=0)  # average predictions over models
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0, :obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
                penalty = np.max(dists, axis=0)  # max distances over models
            else:
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
            d_penalty = 0
            if self.d_penalty:
                d_penalty = np.squeeze(self.discriminator.compute_penalty(scaled_obs, scaled_act, next_obs, rewards))
            penalized_rewards = rewards - penalty_coeff * penalty - self.d_coeff * (d_penalty - 0.5)
            penalty = penalty_coeff * penalty + self.d_coeff * d_penalty
        else:
            penalty = 0
            penalized_rewards = rewards

        assert (type(next_obs) == np.ndarray)
        info = {'penalty': penalty, 'penalized_rewards': penalized_rewards}
        penalized_rewards = penalized_rewards[:, None]
        terminals = terminals[:, None]
        return next_obs, penalized_rewards, penalty, terminals, info

    def update_best_snapshots(self, val_losses):
        updated = False
        for i in range(len(val_losses)):
            current_loss = val_losses[i]
            best_loss = self.best_snapshot_losses[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self.best_snapshot_losses[i] = current_loss
                self.save_model_snapshot(i)
                updated = True
                improvement = (best_loss - current_loss) / best_loss
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))
        return updated

    def reset_best_snapshots(self):
        self.model_best_snapshots = [deepcopy(self.model.ensemble_models[idx].state_dict()) for idx in
                                     range(self.model.ensemble_size)]
        self.best_snapshot_losses = [1e10 for _ in range(self.model.ensemble_size)]

    def save_model_snapshot(self, idx):
        self.model_best_snapshots[idx] = deepcopy(self.model.ensemble_models[idx].state_dict())

    def load_best_snapshots(self):
        self.model.load_state_dicts(self.model_best_snapshots)

    def save_model(self, logger, info):
        save_dir = os.path.join(logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(info))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)

    def load_model(self, env_name):
        file_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.dirname(file_path)
        model_load_dir = os.path.join(root_path, 'dymodel')
        for network_name, network in self.networks.items():
            load_path = os.path.join(model_load_dir, "model.pt")
            self.model = torch.load(load_path)

"""
def TSNE(model):
    #X是特征，不包含target; X_tsne是已经降维之后的特征
    X1 = f['observations'][1000:2200]
    A1 = f['actions'][1000:2200]
    obs_normalizer = StandardNormalizer()
    act_normalizer = StandardNormalizer()
    X1 = obs_normalizer.transform(X1)
    A1 = act_normalizer.transform(A1)
    input = torch.cat((torch.tensor(X1), torch.tensor(A1)), -1)
    output, logvar = model.predict(input)
    #output = output - 0.03 * logvar
    output = output[:5, :, :]
    X1 = output.mean(0).detach().numpy()[:, :-1]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X1)
    # print("Org data dimension is {}.Embedded data dimension is {}".format(X1.shape[-1], X_tsne.shape[-1]))
    X2 = f['next_observations'][1000:2200]
    X_tsne2 = tsne.fit_transform(X2)
    X3 = np.concatenate((X2[:, 10:], f['rewards'][1000:2200][:, np.newaxis]), axis=1)
    X_tsne3 = tsne.fit_transform(X3)
    X4 = np.concatenate((X2[:, 10:], output.mean(0).detach().numpy()[:, 11:]), axis=1)
    X_tsne4 = tsne.fit_transform(X4)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    x2_min, x2_max = X_tsne2.min(0), X_tsne2.max(0)
    X2_norm = (X_tsne2 - x2_min) / (x2_max - x2_min)
    x3_min, x3_max = X_tsne3.min(0), X_tsne3.max(0)
    X3_norm = (X_tsne3 - x3_min) / (x3_max - x3_min)
    x4_min, x4_max = X_tsne4.min(0), X_tsne4.max(0)
    X4_norm = (X_tsne4 - x4_min) / (x4_max - x4_min)
    plt.figure(figsize=(8, 8))
    Y_norm = X_norm[:, 1]
    X_norm = X_norm[:, 0]
    Y2_norm = X2_norm[:, 1]
    X2_norm = X2_norm[:, 0]
    Y3_norm = X3_norm[:, 1]
    X3_norm = X3_norm[:, 0]
    Y4_norm = X4_norm[:, 1]
    X4_norm = X4_norm[:, 0]
    plt.scatter(X2_norm, Y2_norm, color='#FF5C68', label='true state', edgecolor='#FF030E', alpha=2 / 3)
    plt.scatter(X_norm, Y_norm, color='#6B63FF', label='dynamics state', edgecolor='#2F18FF', alpha=2 / 3)
    plt.scatter(X3_norm, Y3_norm, color='#FFC65E', label='true reward', edgecolor='#FFA300', alpha=2 / 3)
    plt.scatter(X4_norm, Y4_norm, color='#5AB865', label='dynamics reward', edgecolor='#008300', alpha=2 / 3)
    plt.legend(loc='upper right')
    plt.title('t-SNE embedding of MOAN')
    #plt.show()
    plt.savefig('./scatter2.svg', dpi=1000, format='svg', transparent=True)


if __name__ == '__main__':
    file_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.dirname(file_path)
    model_path = model_load_dir = os.path.join(root_path, 'dymodel', 'hopper_medium_replay_v2.pt')
    model = torch.load(model_path)
    f = h5py.File('./hopper_medium_replay-v2.hdf5', 'r')
    TSNE(model)
"""