# Role-actor-critic for role-oriented policy learning in actor-critic framework for multi-agent team competitions

This is the source code for our work on role-oriented policy learning in actor-critic framework for multi-agent team competitions. 

## Setting Environment

The file `environment.yml` contains the list of necessary packages. Using conda, a virtual environment can be easily created as:

```console
$ conda env create -f environment.yml
$ conda activate maac
```

After activating the maac environment, install versions of multagent-particle-envs and baselines provided here.

Install multagent-particle-envs:
```console
$ cd multiagent-particle-envs
$ pip install -e .
$ cd ..
```

Similarly install baselines

## Running RAC and MAAC

The codes for training agents with Role-Actor-Critic are in the directory `RAC_final/`.
To train agents with RAC run
```console
$ cd RAC_final
$ ./run_train.sh
```

The codes for training agents with one of the baselines MAAC are in the directory `MAAC_final/`.
To train agents with RAC run
```console
$ cd MAAC_final
$ ./run_train.sh
```

For both of the above cases, `run_train.py` can take several arguments which can be checked by running `python run_train.py --help`.


The trained models will be stored in `[algo]_final/models/[env]/models/run1/model.pt` at regular intervals as determined by `save_rate`, where `[algo]` can be `RAC` for Role-actor-critic or `MAAC` for MAAC and `[env]` can be `catch_goal_fix_len` for Touch-mark or `market3` for Market. 


To run RAC vs MAAC, run the following command in `Role_vs_MAAC_final` directory.
```console
$ python run_test.py role_vs_maac [env] ../MAAC_final/  ../RAC_final/ 1
```

The results are stored in `Role_vs_MAAC_final/results/role_vs_maac/1/` where `[env]` can be `catch_goal_fix_len` for Touch-mark or `market3` for Market. The directory contains the following essential directories and files:
**agent_rewards.pkl** Trained c-maddpg actor and critic model parameters 
**cum_stat.pkl:** Cumulative statistics 
