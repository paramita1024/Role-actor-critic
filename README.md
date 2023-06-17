# Role-actor-critic for role-oriented policy learning in actor-critic framework for multi-agent team competitions

This is the source code for our work on role-oriented policy learning in actor-critic framework for multi-agent team competitions. 

## Setting Environment

The file `environment.yml` contains the list of necessary packages. Using conda, a virtual environment can be easily created as:

```console
$ conda env create -f environment.yml
$ conda activate maac
```

After activating the maac environment, install versions of maddpg and multagent-particle-envs provided here.

Install maddpg:
```console
$ cd maddpg
$ pip install -e .
$ cd ..
```
Install multagent-particle-envs:
```console
$ cd multiagent-particle-envs
$ pip install -e .
$ cd ..
```

## Running C-MADDPG and MADDPG

The codes for training agents with Role-Actor-Critic are in the directory `RAC_final/`.
To train agents with RAC run
```console
$ cd RAC_final
$ python run_train.py
```

The codes for training agents with one of the baselines MAAC are in the directory `MAAC_final/`.

To train agents with RAC run
```console
$ cd MAAC_final
$ python run_train.py
```

For both of the above cases, `run_train.py` can take several arguments which can be checked by running `python run_train.py --help`.

| Argument | Meaning | Default value |
|----------|---------|---------------|
| save_rate | Save the model at an interval of this many episodes | 5000 |
| num_episodes | Number of training episodes | 150000

The trained models will be stored in `RAC_final[MAAC_final]/models/[env]/` at regular intervals as determined by `save_rate`, where `[module]` can be `cmaddpg` or `maddpg`. The directory contains the following essential directories and files:

**models_new:** Trained c-maddpg actor and critic model parameters </br>
**classifiers:** Trained classfier model parameters </br>
**new_catch_v4.4_faster_agrewards.pkl:** Rewards obtained by each agent across different episodes. It is an array of shape (4, num_episodes) </br>
**new_catch_v4.4_faster_landmark.pkl:** A binary array of shape (num_episodes, 4), where 1 at the (i, j) denotes that the j-th agent reached the landmark in the i-th episode </br>
**new_catch_v4.4_faster_speed.pkl:** The speed of the agents in different episodes </br>

To run C-MADDPG vs MADDPG, run the following command in `CMaddpg_Maddpg` directory.
```console
$ python run_test.py
```
`run_test.py` can take argument `steps_trained`,  which can be checked by running `python run_test.py --help`.

The results are stored in `CMaddpg_Maddpg/[module]/results/` where `[module]` can be `cmaddpg_vs_maddpg` or `maddpg_vs_cmaddpg`


## Running static and Dyanamic Incentive schemes

The codes for running C-Maddpg with various incentive schemes are in the directory `CMaddpg/code/src`
```console
$ cd CMaddpg/code/src
$ python run_train.py
```

`run_train.py` can take several arguments which can be checked by running `python run_train.py --help`.

| Argument | Meaning | Default value |
|----------|---------|---------------|
| weak_team_scale | Takes floating point values denoting the fraction of additional fixed incentive for weaker team (alpha_T in the paper) | 0 (no extra reward is given) |
| weak_reward_scale | Takes floating point values denoting the additional fixed incentive for weaker member of the weaker team (alpha_A in the paper) | 0 (no extra reward is given) |
| incentive_reward | Flag for landmark-based or speed-based dynamic incentive scheme. Can take values 0 or 1. | 1 (use dynamic incentive)
| incentive_opt | 'landmark' for landmark-based incentive and 'speed' for speed-based incentive | landmark |
| save_rate | Save the model at an interval of this many episodes | 5000 |
| num_episodes | Number of training episodes | 150000

The trained models will be stored in `CMaddpg/code/models/new_classify_maddpg/` at regular intervals as determined by `save_rate`. The directory contains the following essential directories and files:

**models_new:** Trained c-maddpg actor and critic model parameters </br>
**classifiers:** Trained classfier model parameters </br>
**new_catch_v4.4_faster_agrewards.pkl:** Rewards obtained by each agent across different episodes. It is an array of shape (4, num_episodes) </br>
**new_catch_v4.4_faster_landmark.pkl:** A binary array of shape (num_episodes, 4), where 1 at the (i, j) denotes that the j-th agent reached the landmark in the i-th episode. </br>
**speed_[n].txt:** The speed in different episodes of the i-th agent is stored in `speed_[i].txt`

## RL-designed dynamic Incentive scheme

### Team-RL-Agent-Dynamic
The codes for running Team-RL-Agent-Dynamic are in the directory `RL-incentive/team_rl_agent_dynamic/code/src`
```console
$ cd RL-incentive/team_rl_agent_dynamic/code/src
$ python run_rl_train.py
```
The save_rate and num_episodes can also be set using command line arguments. Details can be obtained by `python run_rl_train.py --help`.

The trained Soft Actor-Critic models are saved in `RL-incentive/team_rl_agent_dynamic/code/models/classify_maddpg_rl/` in zip files as specified by `stable-baselines`.

In order to test the RL-designed incentive scheme, within the directory `RL-incentive/team_rl_agent_dynamic/code/src` run
```console
$ python run_rl_test.py
```
The save_rate, num_episodes and the SAC as saved in `RL-incentive/team_rl_agent_dynamic/code/models/classify_maddpg_rl/` can be adjusted to need using command line arguments and can be checked using `python run_rl_test.py --help`. For example, to run the file with save_rate 5000 num_episodes 150000 and SAC model 25.zip
```console
python run_rl_test.py --save_rate 5000 --num_episodes 150000 --model 25
```

The results obtained are stored in the directory `RL-incentive/team_rl_agent_dynamic/code/models/classify_maddpg_rl_[model_no]/`. The essential files there are:

**new_catch_v4.4_agrewards.pkl:** Rewards of the agents in each episode. It is an array of shape (4, num_episodes) </br>
**new_catch_v4.4_agent_subsidy.pkl:** Incentive value for the weaker agent in each episode. It is an array of length num_episodes. </br>
**new_catch_v4.4_team_subsidy.pkl:** Incentive value for the weaker team in each episode. </br>

### Team-Dynamic-Agent-RL
The codes for running Team-RL-Agent-Dynamic are in the directory `RL-incentive/team_dynamic_agent_rl/code/src`
```console
$ cd RL-incentive/team_dynamic_agent_rl/code/src
$ python run_rl_train.py
```
The save_rate and num_episodes can also be set using command line arguments. Details can be obtained by `python run_rl_train.py --help`.

The trained Soft Actor-Critic models are saved in `RL-incentive/team_dynamic_agent_rl/code/models/classify_maddpg_rl/` in zip files as specified by `stable-baselines`.

In order to test the RL-designed incentive scheme, within the directory `RL-incentive/team_dynamic_agent_rl/code/src` run
```console
$ python run_rl_test.py
```
The save_rate, num_episodes and the SAC as saved in `RL-incentive/team_dynamic_agent_rl/code/models/classify_maddpg_rl/` can be adjusted to need using command line arguments and can be checked using `python run_rl_test.py --help`. For example, to run the file with save_rate 5000 num_episodes 150000 and SAC model 25.zip
```console
$ python run_rl_test.py --save_rate 5000 --num_episodes 150000 --model 25
```

The results obtained are stored in the directory `RL-incentive/team_dynamic_agent_rl/code/models/classify_maddpg_rl_[model_no]/`. The essential files there are:

**new_catch_v4.4_agrewards.pkl:** Rewards of the agents in each episode. It is an array of shape (4, num_episodes) </br>
**new_catch_v4.4_agent_subsidy.pkl:** Incentive value for the weaker agent in each episode. It is an array of length num_episodes. </br>
**new_catch_v4.4_team_subsidy.pkl:** Incentive value for the weaker team in each episode. </br>
