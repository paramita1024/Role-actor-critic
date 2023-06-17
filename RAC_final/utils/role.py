import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.optim import Adam
from torch.distributions import kl_divergence
import torch.distributions as D
import math
from tensorboardX import SummaryWriter
import time


class RoleLearner(nn.Module):
    def __init__(self, 
                input_shape,
                n_agents,
                n_actions,
                latent_dim,
                rnn_hidden_dim,
                nn_hidden_dim,
                actor_dim,
                critic_dim,
                config
            ):

        super(RoleLearner, self).__init__()

        self.input_shape = input_shape
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim

        self.var_floor = config.var_floor
        self.device = 'cpu'
        self.h_loss_weight = config.h_loss_weight
        self.kl_loss_weight = config.kl_loss_weight
        self.dis_loss_weight = config.dis_loss_weight
        self.soft_constraint_weight = config.soft_constraint_weight
        self.bs = 0
        self.lr_coef=config.lr_coef

        self.embed_fc_input_size = input_shape
        self.rnn_hidden_dim = rnn_hidden_dim
        self.nn_hidden_dim = nn_hidden_dim
        activation_func = nn.LeakyReLU(inplace=False)
        
        self.embed_net = nn.Sequential(
                    nn.Linear(self.embed_fc_input_size, self.nn_hidden_dim),
                    nn.BatchNorm1d(self.nn_hidden_dim),
                    activation_func,
                    nn.Linear(self.nn_hidden_dim, self.latent_dim*2)
                )

        self.inference_net = nn.Sequential(
                    nn.Linear(self.rnn_hidden_dim + input_shape, self.nn_hidden_dim),
                    nn.BatchNorm1d(self.nn_hidden_dim),
                    activation_func,
                    nn.Linear(self.nn_hidden_dim, self.latent_dim*2)
                )

        self.opp_net0 = nn.Sequential(
                    nn.Linear(self.rnn_hidden_dim + input_shape, self.nn_hidden_dim),
                    nn.BatchNorm1d(self.nn_hidden_dim),
                    activation_func,
                    nn.Linear(self.nn_hidden_dim, self.latent_dim*2)
                )

        self.opp_net1 = nn.Sequential(
                    nn.Linear(self.rnn_hidden_dim + input_shape, self.nn_hidden_dim),
                    nn.BatchNorm1d(self.nn_hidden_dim),
                    activation_func,
                    nn.Linear(self.nn_hidden_dim, self.latent_dim*2)
                )

        self.latent = th.rand(self.n_agents, self.latent_dim*2)
        self.latent_infer = th.rand(self.n_agents, self.latent_dim*2)

        self.opp_latent0 = th.rand(self.n_agents, self.latent_dim*2)
        self.opp_latent1 = th.rand(self.n_agents, self.latent_dim*2)

        # self.latent_net = nn.Sequential(
                    # nn.Linear(self.latent_dim, self.nn_hidden_dim),
                    # nn.BatchNorm1d(self.nn_hidden_dim),
                    # activation_func
                # )

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        # print('ACTOR_DIM:', self.actor_dim[0], self.actor_dim[1], self.nn_hidden_dim)

        # Actor and critic parameters
        # self.actor_w_nn = nn.Linear(self.nn_hidden_dim, self.actor_dim[0]*self.actor_dim[1])
        # self.actor_b_nn = nn.Linear(self.nn_hidden_dim, self.actor_dim[1])

        # self.critic_w_nn = nn.Linear(self.nn_hidden_dim, self.critic_dim[0]*self.critic_dim[1])
        # self.critic_b_nn = nn.Linear(self.nn_hidden_dim, self.critic_dim[1])
        
        # Dissimilarity Net
        # self.dis_net = nn.Sequential(
                    # nn.Linear(self.latent_dim*2, self.nn_hidden_dim),
                    # nn.BatchNorm1d(self.nn_hidden_dim),
                    # activation_func,
                    # nn.Linear(self.nn_hidden_dim, 1)
                # )

        self.mi = th.rand(self.n_agents * self.n_agents)
        # self.dissimilarity = th.rand(self.n_agents * self.n_agents)

        dis_sigmoid = False
        if dis_sigmoid:
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid
        else:
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_step

        self.optimizer = self.init_optimizer()


    def forward(self, inputs, hidden_state, t=0, batch=None, train_mode=False, t_glob=0):

        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        self.latent = self.embed_net(inputs)
        self.latent[:, -self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)

        # latent_embed = self.latent.reshape(self.bs * self.n_agents, self.latent_dim*2)
        latent_embed = self.latent.reshape(-1, self.latent_dim*2)
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:])**(1/2))
        latent = gaussian_embed.rsample()

        # opponent modelling
        self.opp_latent0 = self.opp_net0(th.cat([h_in.detach(), inputs], dim=1))
        self.opp_latent0[:, -self.latent_dim:] = th.clamp(th.exp(self.opp_latent0[:, -self.latent_dim:]), min=self.var_floor)
        opp_embed0 = self.opp_latent0.reshape(-1, self.latent_dim*2)

        self.opp_latent1 = self.opp_net1(th.cat([h_in.detach(), inputs], dim=1))
        self.opp_latent1[:, -self.latent_dim:] = th.clamp(th.exp(self.opp_latent1[:, -self.latent_dim:]), min=self.var_floor)
        opp_embed1 = self.opp_latent1.reshape(-1, self.latent_dim*2)

        c_dis_loss = th.tensor(0.0).to(self.device)
        ce_loss = th.tensor(0.0).to(self.device)
        loss = th.tensor(0.0).to(self.device)

        if train_mode:
            self.latent_infer = self.inference_net(th.cat([h_in.detach(), inputs], dim=1))
            self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]), min=self.var_floor)
            gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim], (self.latent_infer[:, self.latent_dim:])**(1/2))
            latent_infer = gaussian_infer.rsample()

            loss = gaussian_embed.entropy().sum(dim=-1).mean() * self.h_loss_weight + kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean() * self.kl_loss_weight
            loss = th.clamp(loss, max=2e3)
            ce_loss = th.log(1 + th.exp(loss))

            # Dissimilarity Loss
            curr_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
            if curr_dis_loss_weight > 0:
                dis_loss = 0
                dissimilarity_cat = None
                mi_cat = None
                latent_dis = latent.clone().view(self.bs, self.n_agents, -1)
                latent_move = latent.clone().view(self.bs, self.n_agents, -1)

                for agent_i in range(self.n_agents):
                    latent_move = th.cat([
                                latent_move[:, -1, :].unsqueeze(1),
                                latent_move[:, :-1, :]], 
                                dim=1
                            )
                    latent_dis_pair = th.cat([
                                latent_dis[:, :, :self.latent_dim],
                                latent_move[:, :, :self.latent_dim]],
                                dim=2
                            )

                    mi = th.clamp(gaussian_infer.log_prob(latent_move.view(self.bs*self.n_agents, -1)) + 13.9, min=-13.9).sum(dim=1, keepdim=True)/self.latent_dim

                    # dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2*self.latent_dim)))

                    # if dissimilarity_cat is None:
                        # dissimilarity_cat = dissimilarity.view(self.bs, -1).clone()
                    # else:
                        # dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs, -1)], dim=1)

                    if mi_cat is None:
                        mi_cat = mi.view(self.bs, -1).clone()
                    else:
                        mi_cat = th.cat([mi_cat, mi.view(self.bs, -1)], dim=1)

                mi_min = mi_cat.min(dim=1, keepdim=True)[0]
                mi_max = mi_cat.max(dim=1, keepdim=True)[0]
                # di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
                # di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

                mi_cat = (mi_cat - mi_min)/(mi_max - mi_min + 1e-12)
                # dissimilarity_cat = (dissimilarity_cat - di_min)/(di_max - di_min + 1e-12)

                # dis_loss = -th.clamp(mi_cat + dissimilarity_cat, max=1.0).sum()/self.bs/self.n_agents
                dis_loss = -th.clamp(mi_cat, max=1.0).sum()/self.bs/self.n_agents
                
                # dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum()/self.bs/self.n_agents

                # c_dis_loss = (dis_norm + self.soft_constraint_weight*dis_loss)/self.n_agents * curr_dis_loss_weight
                c_dis_loss = (self.soft_constraint_weight*dis_loss)/self.n_agents * curr_dis_loss_weight

                loss = ce_loss + c_dis_loss

                self.mi = mi_cat[0]
                # self.dissimilarity = dissimilarity_cat[0]

            else:
                c_dis_loss = th.zeros.like(loss)
                loss = ce_loss

        # RNN
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h_in)
        h = h.reshape(-1, self.rnn_hidden_dim)

        # Role -> Actor and critic params
        latent = torch.transpose(latent.view(-1,self.n_agents, self.latent_dim),0,1)
        # latent = self.latent_net(latent)
        # actor_w = self.actor_w_nn(latent)
        # actor_b = self.actor_b_nn(latent)
        # critic_w = self.critic_w_nn(latent)
        # critic_b = self.critic_b_nn(latent)

        # actor_w = actor_w.view(self.n_agents, -1, self.actor_dim[0], self.actor_dim[1])
        # actor_b = actor_b.view(self.n_agents, -1, 1, self.actor_dim[1])
        # critic_w = critic_w.view(self.n_agents, -1, self.critic_dim[0], self.critic_dim[1])
        # critic_b = critic_b.view(self.n_agents, -1, 1, self.critic_dim[1])

        # actor_params = [(a_w, a_b) for a_w, a_b in zip(actor_w, actor_b)]
        # critic_params = [(c_w, c_b) for c_w, c_b in zip(critic_w, critic_b)]

        # TODO: SummaryWriter
        self_embed0 = latent_embed.view(-1, self.n_agents, 2*self.latent_dim).clone()
        self_embed1 = th.zeros(self_embed0.shape)
        self_embed1[:,:1] = self_embed0[:,1:].clone()
        self_embed1[:,1:] = self_embed0[:,:1].clone()
        self_embed0 = self_embed0.view(-1, 2*self.latent_dim)
        self_embed1 = self_embed1.view(-1, 2*self.latent_dim)
        latent_dis = (self_embed0, self_embed1, opp_embed0, opp_embed1)

        return latent, h, latent_dis, loss, ce_loss, c_dis_loss


    def dis_loss_weight_schedule_step(self, t_glob):
        return self.dis_loss_weight
        

    def dis_loss_weight_schedule_sigmoid(self, t_glob):
        return self.dis_loss_weight/(1 + math.exp((1e7 - t_glob)/2e6))

    def get_params(self):
        params = self.state_dict()
        return params


    def init_optimizer(self):
        optimizer = Adam(self.parameters(),lr=self.lr_coef,weight_decay=0.001)
        return optimizer


    def grad_norm(self):
        grad_norm = torch.nn.utils.clip_grad_norm(self.parameters(), 1.0)
        return grad_norm
    

    def load_params(self, params):
        self.load_state_dict(params)

