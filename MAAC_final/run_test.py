import sys
import time
import os

train_dir='../MAAC_Market_devi_v1/'
files=os.listdir(train_dir+'models/market/models/')
run_no= max([int(f.split('run')[1])for f in files])
file_names=os.listdir(train_dir+'models/market/models/run'+str(run_no)+'/incremental/')
models=[ int(fname.split('_ep')[1].split('.')[0]) for fname in file_names]
models.sort()
i=0

for model_no in [96001]:
    if True:#i%10==0:
        # print(i)
        model_path = train_dir+'models/env_id/models/run'+str(run_no)+'/incremental/model_ep'+str(model_no)+'.pt'
    
        os.system('python3 maac_vs_maac.py market  '\
            + ' --model_path ' + model_path \
            + ' --n_episodes 3 '\
            + ' --model_name '+str(model_no)\
            + ' --trajectory 0')
    # i+=1
    # print(i,model_no)
    # time.sleep(5)
