import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributions as D
from torch.distributions import kl_divergence
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
from utils.role import RoleLearner

MSELoss = torch.nn.MSELoss()

class RoleAttentionSAC(object):

    def __init__(self, agent_init_params, sa_size,
            gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
            reward_scale=10.0,
            pol_hidden_dim=128,
            critic_hidden_dim=128,
            attend_heads=4,
            config=None,
            **kwargs):
        
        # Number of agents
        self.nagents = len(sa_size)
        self.n_teams = 2 # TODO: Hard coded team number

        self.agents = [AttentionAgent(lr=pi_lr, role_dim=3*config.latent_dim, hidden_dim=pol_hidden_dim, **params)
                        for params in agent_init_params]
        self.critic = AttentionCritic(sa_size, role_dim=3*config.latent_dim, hidden_dim=critic_hidden_dim, 
                        attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, role_dim=3*config.latent_dim, hidden_dim=critic_hidden_dim,
                        attend_heads=attend_heads)

        hard_update(self.target_critic, self.critic)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)

        # TODO: Allow different input size for different agents
        input_shape = sa_size[0][0] + sa_size[0][1]
        self.actor_dim = (pol_hidden_dim, sa_size[0][1])
        self.critic_dim = (critic_hidden_dim, sa_size[0][1])
        self.role_learner = [RoleLearner(
                    input_shape, self.nagents//self.n_teams, sa_size[1], 
                    config.latent_dim, config.rnn_hidden_dim, config.nn_hidden_dim, 
                    self.actor_dim, self.critic_dim, config
                ) for _ in range(self.n_teams)]
        self.hidden = None
        self.prev_acs = None
        self.time = 0

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'
        self.niter = 0
        self.config = config

        self.loss1 = 0
        self.ce_loss1 = 0
        self.dis_loss1 = 0
        self.loss2 = 0
        self.ce_loss2 = 0
        self.dis_loss2 = 0
        self.decay_coef= config.decay_coef



    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def divergence(self, latent_dis1, latent_dis2):
        latent_dim = self.config.latent_dim
        actual_embed0, actual_embed1, pred_embed2, pred_embed3 = latent_dis1
        pred_embed0, pred_embed1, actual_embed2, actual_embed3 = latent_dis2

        # print("\n\nEmbeddings\n\n")
        # print(pred_embed0.grad_fn)
        # print(pred_embed1.grad_fn)
        # print(actual_embed0.grad_fn)
        # print(actual_embed1.grad_fn)

        actual_dist0 = D.Normal(actual_embed0[:, :latent_dim], (actual_embed0[:, latent_dim:])**(1/2))
        actual_dist1 = D.Normal(actual_embed1[:, :latent_dim], (actual_embed1[:, latent_dim:])**(1/2))
        actual_dist2 = D.Normal(actual_embed2[:, :latent_dim], (actual_embed2[:, latent_dim:])**(1/2))
        actual_dist3 = D.Normal(actual_embed3[:, :latent_dim], (actual_embed3[:, latent_dim:])**(1/2))

        pred_dist0 = D.Normal(pred_embed0[:, :latent_dim], (pred_embed0[:, latent_dim:])**(1/2))
        pred_dist1 = D.Normal(pred_embed1[:, :latent_dim], (pred_embed1[:, latent_dim:])**(1/2))
        pred_dist2 = D.Normal(pred_embed2[:, :latent_dim], (pred_embed2[:, latent_dim:])**(1/2))
        pred_dist3 = D.Normal(pred_embed3[:, :latent_dim], (pred_embed3[:, latent_dim:])**(1/2))

        # print('Distributions of divergence function\n\n')
        # print('*'*50)
        # print(pred_dist0)
        # print(pred_dist1)
        # print(pred_dist2)
        # print(pred_dist3)

        # print(actual_dist0)
        # print(actual_dist1)
        # print(actual_dist2)
        # print(actual_dist3)


        opp_loss1 = kl_divergence(actual_dist2, pred_dist2).sum(dim=-1).mean() + kl_divergence(actual_dist3, pred_dist3).sum(dim=-1).mean()
        opp_loss2 = kl_divergence(actual_dist0, pred_dist0).sum(dim=-1).mean() + kl_divergence(actual_dist1, pred_dist1).sum(dim=-1).mean()
        
        # print("Type and size of opp loss 1 ")
        # print( type(opp_loss1))
        # print((opp_loss1.shape))
        # print(opp_loss1)

        # print("Type and size of opp loss 2 ")
        # print( type(opp_loss2))
        # print((opp_loss2.shape))
        # print(opp_loss2)

        pred_dist = (pred_dist0, pred_dist1, pred_dist2, pred_dist3)

        return opp_loss1, opp_loss2, pred_dist

    def sample_opponent_role(self, pred_dist):
        latent_dim = self.config.latent_dim

        pred_dist0, pred_dist1, pred_dist2, pred_dist3 = pred_dist

        opp_role2 = torch.transpose(pred_dist2.rsample().view(-1, self.nagents//self.n_teams, latent_dim), 0, 1)
        opp_role3 = torch.transpose(pred_dist3.rsample().view(-1, self.nagents//self.n_teams, latent_dim), 0, 1)
        opp_role0 = torch.transpose(pred_dist0.rsample().view(-1, self.nagents//self.n_teams, latent_dim), 0, 1)
        opp_role1 = torch.transpose(pred_dist1.rsample().view(-1, self.nagents//self.n_teams, latent_dim), 0, 1)

        return opp_role0, opp_role1, opp_role2, opp_role3


    def init_episode(self):
        self.time = 0
        self.loss1 = 0
        self.loss2 = 0

        self.hidden = [torch.zeros(self.config.n_rollout_threads, self.nagents//self.n_teams, self.config.rnn_hidden_dim)
                       for _ in range(self.n_teams)]
        self.prev_acs = torch.zeros(self.config.n_rollout_threads, self.nagents, self.actor_dim[1])
        self.role_learner[0].bs = self.config.n_rollout_threads
        self.role_learner[1].bs = self.config.n_rollout_threads

    def step(self, observations, explore=False, train=False):
        # TODO: Generalize for many teams
 
        observations = torch.transpose(torch.stack(observations), 0, 1)
        inputs = torch.cat([observations, self.prev_acs], dim=2)

        hidden = torch.cat([
                    self.hidden[0].view(-1, self.nagents//self.n_teams, self.config.rnn_hidden_dim),
                    self.hidden[1].view(-1, self.nagents//self.n_teams, self.config.rnn_hidden_dim)
                    ], dim=1).detach().numpy()

        latent1, self.hidden[0], latent_dis1, loss1, ce_loss1, dis_loss1 = self.role_learner[0].forward(inputs[:,:2], self.hidden[0], t=self.time, train_mode=train)
        latent2, self.hidden[1], latent_dis2, loss2, ce_loss2, dis_loss2 = self.role_learner[1].forward(inputs[:,2:], self.hidden[1], t=self.time, train_mode=train)

        # print("step\n\n")
        opp_loss1, opp_loss2, pred_dist = self.divergence(latent_dis1, latent_dis2)

        # self.loss1 += loss1
        # self.loss1 += opp_loss1
        # self.ce_loss1 += ce_loss1
        # self.dis_loss1 += dis_loss1
        # self.loss2 += loss2
        # self.loss2 += opp_loss2
        # self.ce_loss2 += ce_loss2
        # self.dis_loss2 += dis_loss2

        self.time += 1

        self_role = [latent1[0], latent1[1], latent2[0], latent2[1]]
        opp_role0, opp_role1, opp_role2, opp_role3 = self.sample_opponent_role(pred_dist)

        opp_role_enc1 = [opp_role0[0], opp_role0[1], opp_role2[0], opp_role2[1]]
        opp_role_enc2 = [opp_role1[0], opp_role1[1], opp_role3[0], opp_role3[1]]

        role_encoding = []
        for s, o1, o2 in zip(self_role, opp_role_enc1, opp_role_enc2):
            role_encoding.append(torch.cat([s, o1, o2], dim=1))


        acs = []
        for pi, ob, role in zip(self.policies, torch.transpose(observations,0,1), role_encoding):
            ac, _ = pi(ob, role, return_log_pi=True)
            acs.append(ac)

        # TODO: Think - prev_acs are determined by policy. Do we need backprop
        self.prev_acs = torch.transpose(torch.stack(acs), 0, 1).detach()
        return acs, hidden


    def finish_episode(self):

        return 

        # loss1 = self.loss1 + self.ce_loss1 + self.dis_loss1
        loss1 = self.loss1
        self.role_learner[0].optimizer.zero_grad()
        loss1.backward()
        grad_norm = self.role_learner[0].grad_norm()
        self.role_learner[0].optimizer.step()

        # loss2 = self.loss2 + self.ce_loss2 + self.dis_loss2
        loss2 = self.loss2
        self.role_learner[1].optimizer.zero_grad()
        loss2.backward()
        grad_norm = self.role_learner[1].grad_norm()
        self.role_learner[1].optimizer.step()

        self.loss1 = 0
        # self.ce_loss1 = 0
        # self.dis_loss1 = 0
        self.loss2 = 0
        # self.ce_loss2 = 0
        # self.dis_loss2 = 0


    def update_critic_episode(self, obs, acs, rews, next_obs, dones, role_embedding,
                        soft=True, logger=None, **kwargs):
 
        # Q loss
        next_acs = []
        next_log_pis = []

        for pi, ob, ac, role in zip(self.target_policies, torch.transpose(next_obs,0,1), torch.transpose(acs,0,1), role_embedding):

            curr_next_ac, curr_next_log_pi = pi(ob, role, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)

        trgt_critic_in = list(zip(torch.transpose(next_obs,0,1), next_acs))
        critic_in = list(zip(torch.transpose(obs,0,1), torch.transpose(acs,0,1)))
        next_qs = self.target_critic(trgt_critic_in, role_embedding)
        critic_rets = self.critic(critic_in, role_embedding, regularize=True, logger=logger, niter=self.niter)

        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs, next_log_pis, critic_rets):
            target_q = (rews[:,a_i].view(-1,1) + self.gamma*nq*(1 - dones[:,a_i].view(-1,1)))

            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg

        q_loss.backward(retain_graph=True)
        self.critic.scale_shared_grads()
            
        grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10*self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1


    def update_policies_episodes(self, obs, acs, rews, next_obs, dones, role_embedding,
                        soft=True, logger=None, **kwargs):

        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob, role in zip(range(self.nagents), self.policies, torch.transpose(obs,0,1), role_embedding):
            curr_ac, probs, log_pi, pol_regs, ent = pi(ob, role,
                                    return_all_probs=True, return_log_pi=True,
                                    regularize=True, return_entropy=True
                                )
            logger.add_scalar('agent%i/policy_entropy'%a_i, ent, self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(torch.transpose(obs,0,1), samp_acs))
        critic_rets = self.critic(critic_in, role_embedding, return_all_q=True)

        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs, all_log_pis, all_pol_regs, critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi/self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg

            disable_gradients(self.critic)
            pol_loss.backward(retain_graph=True)
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss'%a_i, pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi'%a_i, grad_norm, self.niter)


    def update_all_targets(self):

        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)


    def random_start(self, window=20, ep_len=20, batch_size=1024):
        return np.random.randint(low=0, high=ep_len-window+1, size=batch_size)


    def update(self, sample, start_ts=0, end_ts=20, window=20, soft=True, logger=None, **kwargs):
        
        obs, hidden, acs, rews, next_obs, dones = sample

        # obs = torch.transpose(torch.stack(obs), 0, 2)
        # hidden = torch.transpose(torch.stack(hidden), 0, 2)
        # acs = torch.transpose(torch.stack(acs), 0, 2)
        # rews = torch.transpose(torch.stack(rews), 0, 2)
        # next_obs = torch.transpose(torch.stack(next_obs), 0, 2)
        # dones = torch.transpose(torch.stack(dones), 0, 2)

        obs = torch.stack(obs).permute(1, 2, 0, 3)
        hidden = torch.stack(hidden).permute(1, 2, 0, 3)
        acs = torch.stack(acs).permute(1, 2, 0, 3)
        rews = torch.stack(rews).permute(1, 2, 0)
        next_obs = torch.stack(next_obs).permute(1, 2, 0, 3)
        dones = torch.stack(dones).permute(1, 2, 0)

        start_ts = self.random_start(window=window, ep_len=obs.shape[1], batch_size=obs.shape[0])
        batch = np.arange(obs.shape[0])

        # time_steps = obs.shape[0]
        # prev_acs = torch.zeros(self.config.batch_size, self.nagents, self.actor_dim[1])
        prev_acs = acs[batch, np.maximum(start_ts - 1, 0)]

        # hidden1 = torch.zeros(self.config.batch_size, self.nagents//self.n_teams, self.config.rnn_hidden_dim)
        # hidden2 = torch.zeros(self.config.batch_size, self.nagents//self.n_teams, self.config.rnn_hidden_dim)
        hidden = hidden[batch, start_ts]
        hidden1 = hidden[:,:2]
        hidden2 = hidden[:,2:]

        # print(obs.shape, hidden.shape)

        self.role_learner[0].bs = self.config.batch_size
        self.role_learner[1].bs = self.config.batch_size

        self.role_learner[0].optimizer.zero_grad()
        self.role_learner[1].optimizer.zero_grad()

        total_loss1 = 0
        total_loss2 = 0
        total_ce_loss1 = 0
        total_ce_loss2 = 0

        total_dis_loss1 = 0
        total_dis_loss2 = 0


        total_opp_loss1 = 0
        total_opp_loss2 = 0

        for i in range(window):

            inputs = torch.cat([obs[batch, start_ts], prev_acs], dim=2)

            latent1, hidden1, latent_dis1, loss1, ce_loss1, dis_loss1 = self.role_learner[0].forward(inputs[:,:2], hidden1, t=self.time, train_mode=True)
            latent2, hidden2, latent_dis2, loss2, ce_loss2, dis_loss2 = self.role_learner[1].forward(inputs[:,2:], hidden2, t=self.time, train_mode=True)

            total_loss1 += loss1
            total_loss2 += loss2
            total_ce_loss1 += ce_loss1
            total_ce_loss2 += ce_loss2

            total_dis_loss1 += dis_loss1
            total_dis_loss2 += dis_loss2


            # print("update\n\n")
            opp_loss1, opp_loss2, pred_dist = self.divergence(latent_dis1, latent_dis2)

            total_opp_loss1 += opp_loss1
            total_opp_loss2 += opp_loss2
                 
        

            self_role = [latent1[0], latent1[1], latent2[0], latent2[1]]

            opp_role0, opp_role1, opp_role2, opp_role3 = self.sample_opponent_role(pred_dist)
            opp_role_enc1 = [opp_role0[0], opp_role0[1], opp_role2[0], opp_role2[1]]
            opp_role_enc2 = [opp_role1[0], opp_role1[1], opp_role3[0], opp_role3[1]]

            role_embedding = []
            for s, o1, o2 in zip(self_role, opp_role_enc1, opp_role_enc2):
                role_embedding.append(torch.cat([s, o1, o2], dim=1))

            self.update_critic_episode(obs[batch, start_ts], acs[batch, start_ts], 
                                rews[batch, start_ts], next_obs[batch, start_ts], dones[batch, start_ts], 
                                role_embedding, soft, logger, **kwargs)

            self.update_policies_episodes(obs[batch, start_ts], acs[batch, start_ts], 
                                rews[batch, start_ts], next_obs[batch, start_ts], dones[batch, start_ts],
                                role_embedding, soft, logger, **kwargs)

            self.update_all_targets()

            prev_acs = acs[batch, start_ts]

            start_ts = start_ts + 1

        total_loss1 *=  (self.decay_coef)**(float(self.niter)/1000)
        total_loss2 *= (self.decay_coef)**(float(self.niter)/1000)
        total_ce_loss1 *= (self.decay_coef)**(float(self.niter)/1000) 
        total_ce_loss2 *= (self.decay_coef)**(float(self.niter)/1000) 
        total_dis_loss1 *= (self.decay_coef)**(float(self.niter)/1000)
        total_dis_loss2 *= (self.decay_coef)**(float(self.niter)/1000) 
        total_opp_loss1 *= (self.decay_coef)**(float(self.niter)/1000)
        total_opp_loss2 *= (self.decay_coef)**(float(self.niter)/1000) 

        total_loss1.backward(retain_graph=True)
        total_loss2.backward(retain_graph=True)
        total_ce_loss1.backward(retain_graph=True)# += ce_loss1
        total_ce_loss2.backward(retain_graph=True)# += ce_loss2
        total_dis_loss1.backward(retain_graph=True)# += dis_loss1
        total_dis_loss2.backward(retain_graph=True)# += dis_loss2
        total_opp_loss1.backward(retain_graph=True)
        total_opp_loss2.backward(retain_graph=True)

        # self.role_learner[0].optimizer.zero_grad()
        # self.role_learner[1].optimizer.zero_grad()
        self.role_learner[0].grad_norm()
        self.role_learner[1].grad_norm()
        self.role_learner[0].optimizer.step()
        self.role_learner[1].optimizer.step()


    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device


    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
                'init_dict': self.init_dict,
                'agent_params': [a.get_params() for a in self.agents],
                'critic_params': {'critic': self.critic.state_dict(),
                                  'target_critic': self.target_critic.state_dict(),
                                  'critic_optimizer': self.critic_optimizer.state_dict()
                                  },
                'role_learner_params': [role_learner.get_params() for role_learner in self.role_learner]
                }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, config):
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            agent_init_params.append({'num_in_pol':obsp.shape[0],
                                      'num_out_pol':acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': config.gamma, 'tau': config.tau,
                     'pi_lr': config.pi_lr, 'q_lr': config.q_lr,
                     'reward_scale': config.reward_scale,
                     'pol_hidden_dim': config.pol_hidden_dim,
                     'critic_hidden_dim': config.critic_hidden_dim,
                     'attend_heads': config.attend_heads, 
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'config': config
                     }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']

        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])

        for role_learner, params in zip(instance.role_learner, save_dict['role_learner_params']):
            role_learner.load_params(params)
        return instance
