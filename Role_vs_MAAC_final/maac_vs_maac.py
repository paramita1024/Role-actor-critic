import random
import argparse
import torch
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import pdb
from gym.spaces import Box, Discrete
from gym import wrappers
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from maac.utils.make_env import make_env
from maac.utils.buffer import ReplayBuffer
from role_maac.utils.episode_buffer import EpisodeReplayBuffer
from maac.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from maac.algorithms.attention_sac import AttentionSAC
from role_maac.algorithms.role_sac import RoleAttentionSAC

def make_parallel_env(env_id, n_rollout_threads, seed,ep_pos):
    def get_env_fn(rank,ep_pos):
        def init_env():
            env = make_env(env_id, discrete_action=True,ep_pos=ep_pos)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0,ep_pos)])
    else:
        return SubprocVecEnv([get_env_fn(i,ep_pos) for i in range(n_rollout_threads)])

def update(env_id, cum_stat_n,stat_n):

    if 'catch_goal' in env_id:

        for i,ag_stat in enumerate(stat_n):
            if ag_stat==1:
                cum_stat_n[i]+=1
        return cum_stat_n

    if 'market' in env_id:

        for stat, cum_stat in zip(stat_n,cum_stat_n):
            for i,stat_re in enumerate(stat):
                if stat_re == 1:
                    cum_stat[i][0] += 1
                if stat_re == 2:
                    cum_stat[i][1] += 1
        return cum_stat_n

    print('Incorrect environment')

def init_stat(env_id):

    if 'catch_goal' in env_id:
        return [ 0 for i in range(4)] 
    if 'market' in env_id:
        return [ [[0,0],[0,0]] for i in range(4)] 
    print('Unseen scenario')



def init_traj(env_id):

    if 'catch_goal' in env_id:
        trajectory = {}
        trajectory['lm1'] = []
        trajectory['lm2'] = []
        trajectory['agent0'] = []
        trajectory['agent1'] = []
        trajectory['agent2'] = []
        trajectory['agent3'] = []
        
    if 'market' in env_id:
        trajectory = {}
        trajectory['cons1'] = []
        trajectory['cons2'] = []
        trajectory['agent0'] = []
        trajectory['agent1'] = []
        trajectory['agent2'] = []
        trajectory['agent3'] = []
        trajectory['prod0']=[]
        trajectory['prod1']=[]
        trajectory['prod2']=[]
        trajectory['prod3']=[]

    return trajectory

def update_traj(env_id,trajectory,obs_n):

    if 'catch_goal' in env_id:
        # print(trajectory.keys())
        trajectory['lm1'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][2:4]))
        trajectory['lm2'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][4:6]))
        trajectory['agent0'].append(np.array(obs_n[0][:2]))
        trajectory['agent1'].append(np.array(obs_n[1][:2]))
        trajectory['agent2'].append(np.array(obs_n[2][:2]))
        trajectory['agent3'].append(np.array(obs_n[3][:2]))


    if 'market' in env_id:
        trajectory['cons1'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][48:50]))
        trajectory['cons2'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][54:56]))
        trajectory['agent0'].append(np.array(obs_n[0][:2]))
        trajectory['agent1'].append(np.array(obs_n[1][:2]))
        trajectory['agent2'].append(np.array(obs_n[2][:2]))
        trajectory['agent3'].append(np.array(obs_n[3][:2]))
        trajectory['prod0'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][24:26]))
        trajectory['prod1'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][30:32]))
        trajectory['prod2'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][36:38]))
        trajectory['prod3'].append(np.array(obs_n[0][:2])+np.array(obs_n[0][42:44]))

    return trajectory

def get_positions(env_id):
	fname=''
	if 'catch_goal' in env_id:
		fname='random_pos_catch_goal.pkl'
	if 'market' in env_id:
		fname='random_pos_market.pkl'
	if fname=='':
		print('Unknown environment')
		return None
	with open(fname, 'rb') as f:
	    ep_pos = pickle.load(f)
	return ep_pos

def run_episode(config):
    
    res_path = 'results/' #+ str(config.model_no) + '/'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    res_path += config.method+'/'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    res_path += str(config.model_name)+'/'
    if os.path.isdir(res_path):
        os.system('rm -rf '+res_path)
    os.mkdir(res_path)
    os.mkdir(res_path+'trajectory')
    # run_num=1
    model_dir = config.model_path.replace('env_id',config.env_id)
    opp_model_dir = config.role_model_path.replace('env_id',config.env_id)

    #TODO: model_dir for RoleAttentionSAC(agent_init_params, sa_size)
    # Path('./models') / config.env_id / 'model/run1/incremental'\
    #  / config.model_name
    # # config.model = 'model_epX.pt' 
    # curr_run = 'run%i' % run_num
    # model_dir = model_dir / curr_run / 'model.pt'
    # log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)
    # logger = SummaryWriter(str(log_dir))

    run_num=random.randint(1,10)
    with open('seed','a+') as f:
    	f.write('Seed: '+str(run_num))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    
    # with U.single_threaded_session():
    if True:#with open('maac_vs_maac.py') as f:
        # Create environment

        ep_pos=get_positions(config.env_id)
        
        env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num,ep_pos)
    
        # env = make_env(config.scenario, config, config.benchmark)
        # Create agent trainers
        '''
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, config.num_adversaries)
        trainers1 = get_trainers_win(env, num_adversaries, obs_shape_n, config)
        trainers2 = get_trainers_lose(env, num_adversaries, obs_shape_n, config)
        print('Using good policy {} and adv policy {}'.format(config.good_policy, config.adv_policy))
        '''
        # model_dir='models/new_catch_v4.4/model/run9/model.pt'
        model = AttentionSAC.init_from_save(model_dir, load_critic=True)
        model.prep_rollouts(device='cpu')
        # model.config.n_rollout_threads=config.n_rollout_threads

        opp_model = AttentionSAC.init_from_save(opp_model_dir, load_critic=True) 
        opp_model.prep_rollouts(device='cpu')

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(len(env.envs[0].agents))]  # individual agent reward
        # obs_n = env.reset(speed[int(config.model_no)])
        obs_n = env.reset()
        cum_stat_n_list=[]
        cum_stat_n=init_stat(config.env_id)#[ 0 for i in range(4)] 
        # role_model.init_episode()
        
        episode_step = 0
        episode_counter = 0 

        # print('Starting episode play ... ')
        if config.trajectory:
            trajectory=init_traj(config.env_id)

        while True:

            # start=time.time()
            # print('time',start)
            torch_obs = [Variable(torch.Tensor(np.vstack(obs_n[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            

            torch_agent_actions = model.step(torch_obs, explore=True)
            
            
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # return 
            opp_agent_actions = opp_model.step(torch_obs, explore=True)
            opp_agent_actions = [ac.data.numpy() for ac in opp_agent_actions]
            # return 

            # if config.method=='maac_vs_role':
            action_n = agent_actions[:2] + opp_agent_actions[2:]
            # if config.method=='role_vs_maac':
            #     action_n = role_agent_actions[:2] + agent_actions[2:]

            # print('actions before',np.array(action_n).shape)
            action_n = [[ac[i] for ac in action_n] for i in range(config.n_rollout_threads)]
            # print('actions after',np.array(action_n).shape)

            # environment step
            new_obs_n, rew_stat_n, done_n, _ = env.step(action_n)
            rew_n=[ rew_stat[0] for rew_stat in rew_stat_n[0]]
            stat_n=[ rew_stat[1] for rew_stat in rew_stat_n[0]]
            
            cum_stat_n=update(config.env_id,cum_stat_n,stat_n) # cumulative stat n 

            new_obs_n = new_obs_n[0]
            # rew_n = rew_n[0]
            done_n = done_n[0]
            new_rew_n = []
            new_rew_n.append(rew_n[0])
            new_rew_n.append(rew_n[1])
            new_rew_n.append(rew_n[2])
            new_rew_n.append(rew_n[3])
            episode_step += 1
            done = all(done_n)

            # print(config.episode_length)
            terminal = (episode_step >= config.episode_length)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            # ( 25:26, 31:32, 37:38, 43:44),12 (49:50, 55:56, )
            # trajectory
            if config.trajectory:
                trajectory=update_traj(config.env_id, trajectory, obs_n)
            
            obs_n = np.array([obs_n])

            if done or terminal:

                ep_no = len(episode_rewards)-1
                if config.trajectory:
                    with open(res_path + 'trajectory/' + str(episode_counter) + '_trajectory.pkl', 'wb') as f:
                        pickle.dump(trajectory, f)
                    trajectory=init_traj(config.env_id)

                
                obs_n = env.reset()#speed[int(config.model_no)])
                cum_stat_n_list.append(cum_stat_n)
                cum_stat_n=init_stat(config.env_id)#[ 0 for i in range(4)] 
                for a in agent_rewards:
                    a.append(0)
                episode_rewards.append(0)
            
                episode_step = 0
                episode_counter+=1
                # role_model.init_episode()
                
                if(len(episode_rewards)>=config.n_episodes):
                    
                    with open(res_path + 'agent_rewards.pkl', 'wb') as f:
                        pickle.dump(agent_rewards, f)
                    with open(res_path + 'cum_stat.pkl', 'wb') as f:
                        pickle.dump(cum_stat_n_list, f)
                    exit()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id",default='market',help="Name of environment")
    parser.add_argument("--model_name", default=0,
                        help="Name of directory to store " +
                             "model/training contents",type=int)
    parser.add_argument("--model_path",default='',type=str)
    parser.add_argument("--role_model_path",default='',type=str)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--trajectory",default=0,type=int)
    parser.add_argument("--method",default='maac_vs_role',type=str)
    config = parser.parse_args()
    run_episode(config)