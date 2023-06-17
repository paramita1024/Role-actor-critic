import pickle
import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


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

    if 'rps' in env_id:

        for i,ag_stat in enumerate(stat_n):
            cum_stat_n[i]=[a+b for a,b in zip(ag_stat,cum_stat_n[i])]
        return cum_stat_n

    print('Incorrect environment')

def init_stat(env_id):

    if 'catch_goal' in env_id:
        return [ 0 for i in range(4)] 
    if 'market' in env_id:
        return [ [[0,0],[0,0]] for i in range(4)] 
    if 'rps' in env_id:
        return [ [0,0,0] for i in range(4)] 
        


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    # run_num=1
    rew=[]
    cum_stat_n_list=[]
    # init_stat(config.env_id)
    # cum_stat_n=[ 0 for i in range(4)] 

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    seed=np.random.randint(1,10000)
    with open('seed','a+') as f:
        f.write('Run'+str(curr_run)+': Seed: '+str(seed))

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, seed)
    if config.load_from_model:
        model = AttentionSAC.init_from_save(model_dir, load_critic=True)
        with open(run_dir+'/replay_buffer','rb') as f_in:
            replay_buffer=pickle.load(f_in)
    else:
        model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)        
        replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                      for acsp in env.action_space])
    # print(env.observation_space, env.action_space)
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        cum_stat_n = init_stat(config.env_id)
        # cum_stat_n=[ 0 for i in range(4)] 
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            # return 
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            # return 
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, reward_stats, dones, infos = env.step(actions)
            # return 
            # print(type(rewards))
            # print(reward_stats.shape)
            rewards=np.array([ [ag_rew_stat[0] for ag_rew_stat in rew_stat_thread] for rew_stat_thread in reward_stats])
            # print(np.array(rew_n).shape)
            stats=[ [ag_rew_stat[1] for ag_rew_stat in rew_stat_thread] for rew_stat_thread in reward_stats]
            # print(rew_n)
            # print(stat_n)
            # return 
            for stat in stats:          
                cum_stat_n=update(config.env_id, cum_stat_n,stat) # cumulative stat n 
            
            # import time
            # time.sleep(2300)
            # rew_ep
            # return 
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            # return 
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
            # return 
        cum_stat_n=np.array(cum_stat_n)/config.n_rollout_threads
        cum_stat_n_list.append(cum_stat_n)
        
        # return 
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        rew.append(ep_rews)
        for a_i, a_ep_rew, cum_stat in zip(range(4),ep_rews,cum_stat_n):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)
            if np.isscalar(cum_stat):
                logger.add_scalar('agent%i/mean_performance' % a_i,
                                  cum_stat, ep_i)
            else:
                for i,elm in enumerate(np.array(cum_stat).flatten()):
                    logger.add_scalar('agent%i/mean_performance%i' % (a_i,i),
                          elm, ep_i)    

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')
        if len(rew) % 100 == 0 :
            with open(run_dir / 'rewards.pkl','wb') as f_out:
                pickle.dump(rew,f_out) 
            with open(run_dir / 'stats.pkl','wb') as f_out:
                pickle.dump(cum_stat_n_list,f_out) 


    
    model.save(run_dir / 'model.pt')
    # with open(run_dir / 'replay_buffer','wb') as f_out:
    #     pickle.dump(replay_buffer,f_out) 
    # print(np.array(rew).shape)
    with open(run_dir / 'rewards.pkl','wb') as f_out:
        pickle.dump(rew,f_out) 
    # print(np.array(cum_stat_n_list).shape)
    with open(run_dir / 'stats.pkl','wb') as f_out:
        pickle.dump(cum_stat_n_list,f_out) 

    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=128, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=150000, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--steps_per_update", default=512, type=int)
    parser.add_argument("--num_updates", default=2, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    # parser.add_argument("--n_rollout_threads", default=2, type=int)
    # parser.add_argument("--buffer_length", default=int(1e6), type=int)
    # parser.add_argument("--n_episodes", default=10, type=int)
    # parser.add_argument("--episode_length", default=3, type=int)
    # parser.add_argument("--steps_per_update", default=16, type=int)
    # parser.add_argument("--num_updates", default=1, type=int,
    #                     help="Number of updates per update cycle")
    # parser.add_argument("--batch_size",
    #                     default=2, type=int,
    #                     help="Batch size for training")
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--load_from_model",default=0, type=int)
    config = parser.parse_args()

    run(config)
