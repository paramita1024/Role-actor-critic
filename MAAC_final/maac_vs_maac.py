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
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC

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

# def plot_episode(trajectory, ep_no, save_dir):
#     import matplotlib.animation as animation
#     from celluloid import Camera

#     pos0 = trajectory['agent0']
#     pos1 = trajectory['agent1']
#     pos2 = trajectory['agent2']
#     pos3 = trajectory['agent3']
#     pos4 = trajectory['lm1']
#     pos5 = trajectory['lm2']
#     ep_length = len(pos0)

#     fig = plt.figure()
#     camera = Camera(fig)
#     for i in range(ep_length):
#         plt.scatter(pos4[i][0], pos4[i][1], color='#1f6650', sizes=[200])
#         plt.scatter(pos5[i][0], pos5[i][1], color='#1f6650', sizes=[200])
#         plt.scatter(pos0[i][0], pos0[i][1], color='#ff8080', sizes=[200])
#         plt.scatter(pos1[i][0], pos1[i][1], color='#ff0000', sizes=[200])
#         plt.scatter(pos2[i][0], pos2[i][1], color='#80bfff', sizes=[200])
#         plt.scatter(pos3[i][0], pos3[i][1], color='#0066cc', sizes=[200])
#         plt.xlim(-1.5, 1.5)
#         plt.ylim(-1.5, 1.5)
#         camera.snap()
#     anim = camera.animate(blit=True)
#     anim.save(save_dir + '.gif')
#     plt.close()

def init_traj(trajectory=None):
    if trajectory:
        trajectory.clear()
    trajectory={}
    cons=['cons'+str(i+1) for i in range(2)]
    agents=['agent'+str(i) for i in range(4)]
    prods=['prod'+str(i) for i in range(4)]
    keys=cons + agents + prods
    for key in keys:
        trajectory[key]=[]
    return trajectory

def get_positions(obs_n):
    cons={}
    ag={}
    prod={}
    cons['1']=np.array(obs_n[0][:2])+np.array(obs_n[0][48:50])
    cons['2']=np.array(obs_n[0][:2])+np.array(obs_n[0][54:56])
    for i in range(4):
    	ag[str(i)]=np.array(obs_n[i][:2])
    	prod[str(i)]=np.array(obs_n[0][:2])+np.array(obs_n[0][6*(4+i):6*(4+i)+2])
    return ag,prod,cons

def get_resource_stat(obs_n,curr_stat):
	ag,prod,cons=get_positions(obs_n)

    for i in range(4):
    	for j in range(2):
			if curr_stat[str(i)][str(j)]==0:
                for c in self.consumer(world):
                    if c.type==agent.holding:
                        reward = -np.sqrt(np.sum(np.square(c.state.p_pos - agent.state.p_pos)))/30.0
	        for a in self.team_1(world):
	            for r in self.resources(world):
	                if r.alive and a.holding is None and self.is_collision(a, r, world):
	                    reward += 20
	            for c in self.consumers(world):
	                if c.alive and c.type==a.holding and self.is_collision(a, c, world):
	                    reward += 40
	        agent.reward += reward


def run_episode(config):
    
    # s = open('../models/new_classify_maddpg/speed.pkl', 'rb') # tbd 
    # import pickle
    # speed = pickle.load(s)
    '''
    res_path = '../results/classify_vs_classify/' + str(config.model_no) + '/'

    if not os.path.isdir( '../results/classify_vs_classify/'):
        os.mkdir('../results/classify_vs_classify/')
    if os.path.isdir( res_path ):
        os.system( 'rm -rf ' + res_path  )
    os.mkdir(res_path)
    os.mkdir(res_path + 'trajectory' )
    '''
    res_path = 'results/' #+ str(config.model_no) + '/'
    if not os.path.isdir( res_path):
        os.mkdir(res_path)
    res_path += str(config.model_name)+'/'
    if os.path.isdir(res_path):
        os.system('rm -rf '+res_path)
    os.mkdir(res_path)
    os.mkdir(res_path + 'trajectory' )
    
    
    # run_num=1
    model_dir = config.model_path.replace('env_id',config.env_id)
    # Path('./models') / config.env_id / 'model/run1/incremental'\
    #  / config.model_name
    # # config.model = 'model_epX.pt' 
    # curr_run = 'run%i' % run_num
    # model_dir = model_dir / curr_run / 'model.pt'
    # log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)
    # logger = SummaryWriter(str(log_dir))
    run_num=random.randint(1,10)
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    
    # with U.single_threaded_session():
    with open('maac_vs_maac.py') as f:
        # Create environment
        f1 = open('random_pos.pkl', 'rb')
        ep_pos = pickle.load(f1)
        f1.close()

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
        # replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
        #                          [obsp.shape[0] for obsp in env.observation_space],
        #                          [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
        #                           for acsp in env.action_space])
 
        # Initialize
        # U.initialize()

        # Load previous results, if necessary
        '''
        if config.load_dir == "":
            config.load_dir = config.save_dir
        if config.display or config.restore or config.benchmark:
            print('Loading previous state...')
            U.load_state(config.load_dir)

        classifiers = []
        for i in range(env.n):
            classifiers.append(keras.models.load_model( config.classifier + 'classifier_'+str(i)))
        '''

        

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(len(env.envs[0].agents))]  # individual agent reward
        # obs_n = env.reset(speed[int(config.model_no)])
        obs_n = env.reset()
        episode_step = 0
        episode_no = 1
        episode_counter = 0
        cum_stat_n_list=[]
        cum_stat_n=[ [[0,0],[0,0]] for i in range(4)] 

        # print('Starting episode play ... ')
        if config.trajectory:
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

        while True:
            # get action
            # trainers = choose_trainers(obs_n, classifiers, trainers1, trainers2, len(episode_rewards))
            # action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # for obs in obs_n:
            #     print(np.array(obs).shape)
            obs_n = [np.array(obs).reshape(4,60) for obs in obs_n]
            # print(type(obs_n[0]), obs_n[0])
            torch_obs = [Variable(torch.Tensor(np.vstack(obs)), requires_grad=False) for obs in obs_n]
            torch_agent_actions = model.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # action_n = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            action_n = agent_actions
            # environment step
            new_obs_n, rew_stat_n, done_n, _ = env.step(action_n)
            rew_n=[ rew for rew,stat in rew_stat_n]
            stat_n=[ stat for rew,stat in rew_stat_n]
            def update(cum_stat_n,stat_n):
            	for stat, cum_stat in zip(stat_n,cum_stat_n):
            		for i,stat_re in enumerate(stat):
            			if stat_re == 1:
            				cum_stat[i][0] += 1
            			if stat_re == 2:
            				cum_stat[i][1] += 1
            			
            cum_stat_n=update(cum_stat_n,stat_n) # cumulative stat n 
            new_obs_n = new_obs_n[0]
            rew_n = rew_n[0]
            done_n = done_n[0]
            new_rew_n = []
            new_rew_n.append(rew_n[0])
            new_rew_n.append(rew_n[1])
            new_rew_n.append(rew_n[2])
            new_rew_n.append(rew_n[3])
            episode_step += 1
            done = all(done_n)
            # print(done)
            terminal = (episode_step >= config.episode_length)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            # ( 25:26, 31:32, 37:38, 43:44),12 (49:50, 55:56, )
            # trajectory
            if config.trajectory:
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
            obs_n = [obs_n]

            # print('running')

            if done or terminal:

                ep_no = len(episode_rewards)-1
                if config.trajectory:
                    with open(res_path + 'trajectory/' + str(episode_counter) + '_trajectory.pkl', 'wb') as f:
                        pickle.dump(trajectory, f)
                
                    trajectory.clear()
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

                episode_no += 1

                obs_n = env.reset()#speed[int(config.model_no)])
                cum_stat_n_list.append(cum_stat_n)
		        cum_stat_n=[ [[0,0],[0,0]] for i in range(4)] 
                episode_step = 0
                episode_counter+=1
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                if(len(episode_rewards)>=config.n_episodes):
                	# with open(res_path + 'episode_rewards.pkl', 'wb') as f:
        	        #     pickle.dump(episode_rewards, f)
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
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=3, type=int)
    parser.add_argument("--episode_length", default=30, type=int)
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

    config = parser.parse_args()

    run_episode(config)
