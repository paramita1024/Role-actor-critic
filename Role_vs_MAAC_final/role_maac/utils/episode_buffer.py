import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

class EpisodeReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, episode_length, num_threads, num_agents, obs_dims, ac_dims, hidden_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.max_eps = max_steps // episode_length
        self.episode_length = episode_length
        self.num_threads = num_threads
        self.num_agents = num_agents
        self.obs_buffs = []
        self.hidden_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim, hdim in zip(obs_dims, ac_dims, hidden_dims):
            self.obs_buffs.append(np.zeros((self.max_eps, episode_length, odim), dtype=np.float32))
            self.hidden_buffs.append(np.zeros((self.max_eps, episode_length, hdim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((self.max_eps, episode_length, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros((self.max_eps, episode_length), dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((self.max_eps, episode_length, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros((self.max_eps, episode_length), dtype=np.uint8))

        self.ep_obs = []
        self.ep_hidden = []
        self.ep_ac = []
        self.ep_rew = []
        self.ep_next_obs = []
        self.ep_done = []
        for odim, adim, hdim in zip(obs_dims, ac_dims, hidden_dims):
            self.ep_obs.append(np.zeros((episode_length, num_threads, odim), dtype=np.float32))
            self.ep_hidden.append(np.zeros((episode_length, num_threads, hdim), dtype=np.float32))
            self.ep_ac.append(np.zeros((episode_length, num_threads, adim), dtype=np.float32))
            self.ep_rew.append(np.zeros((episode_length, num_threads), dtype=np.float32))
            self.ep_next_obs.append(np.zeros((episode_length, num_threads, odim), dtype=np.float32))
            self.ep_done.append(np.zeros((episode_length, num_threads), dtype=np.float32))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)
        self.ep_i = 0

    def __len__(self):
        return self.filled_i

    def init_episode(self):
        self.ep_i = 0

    def push_step(self, observations, hidden, actions, rewards, next_observations, dones):
        for i in range(self.num_agents):
            self.ep_obs[i][self.ep_i] = observations[:, i]
            self.ep_hidden[i][self.ep_i] = hidden[:, i]
            self.ep_ac[i][self.ep_i] = actions[i]   # actions are indexed differently
            self.ep_rew[i][self.ep_i] = rewards[:, i]
            self.ep_next_obs[i][self.ep_i] = next_observations[:, i]
            self.ep_done[i][self.ep_i] = dones[:, i]
        
        self.ep_i += 1

    def finish_episode(self):

        n_entries = self.num_threads
        if self.curr_i + n_entries > self.max_eps:
            rollover = self.max_eps - self.curr_i
            for i in range(self.num_agents):
                self.obs_buffs[i] = np.roll(self.obs_buffs[i], rollover, axis=0)
                self.hidden_buffs[i] = np.roll(self.hidden_buffs[i], rollover, axis=0)
                self.ac_buffs[i] = np.roll(self.ac_buffs[i], rollover, axis=0)
                self.rew_buffs[i] = np.roll(self.rew_buffs[i], rollover, axis=0)
                self.next_obs_buffs[i] = np.roll(self.next_obs_buffs[i], rollover, axis=0)
                self.done_buffs[i] = np.roll(self.done_buffs[i], rollover, axis=0)
            
            self.curr_i = 0
            self.filled_i = self.max_eps

        for i in range(self.num_agents):
            self.obs_buffs[i][self.curr_i:self.curr_i+n_entries] = np.transpose(self.ep_obs[i], axes=(1, 0, 2))
            self.hidden_buffs[i][self.curr_i:self.curr_i+n_entries] = np.transpose(self.ep_hidden[i], axes=(1, 0, 2))
            self.ac_buffs[i][self.curr_i:self.curr_i+n_entries] = np.transpose(self.ep_ac[i], axes=(1, 0, 2))
            self.rew_buffs[i][self.curr_i:self.curr_i+n_entries] = np.transpose(self.ep_rew[i], axes=(1, 0))
            self.next_obs_buffs[i][self.curr_i:self.curr_i+n_entries] = np.transpose(self.ep_next_obs[i], axes=(1, 0, 2))
            self.done_buffs[i][self.curr_i:self.curr_i+n_entries] = np.transpose(self.ep_done[0], axes=(1, 0))

        self.curr_i += n_entries
        if self.filled_i < self.max_eps:
            self.filled_i += n_entries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std()) 
                             if self.rew_buffs[i][:self.filled_i].std() != 0 
                             else cast(np.zeros(self.rew_buffs[i][inds].shape))
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]

        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.hidden_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
