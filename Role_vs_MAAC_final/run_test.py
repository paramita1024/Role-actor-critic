import sys
import time
import os

def ep_len(scenario):
	if 'catch_goal' in scenario:
		return 20
	if 'market' in scenario:
		return 50

method=sys.argv[1]#'../Market_MAAC_v1/'
scenario=sys.argv[2]
train_dir=sys.argv[3]
role_train_dir=sys.argv[4]

trajectory_flag='0'
if len(sys.argv)>5:
	trajectory_flag=sys.argv[5]
if len(sys.argv)>6:
	if sys.argv[6]=='1':
		common_models=[40065,60033,90113,120065]#sys.argv[6]

min_model_no=-1
if len(sys.argv)>7:
	min_model_no=int(sys.argv[7])


if method in ['maac_vs_role','role_vs_maac','maac_vs_maac']:

	if method in ['maac_vs_maac','maac_vs_role']:
		run_method=method
	if method in ['role_vs_maac']:
		run_method='maac_vs_role'
	
	files=os.listdir(train_dir+'models/'+scenario+'/models/')
	run_no=max([int(f.split('run')[1])for f in files])#int(sys.argv[5]) # 
	file_names=os.listdir(train_dir+'models/'+scenario+'/models/run'+str(run_no)+'/incremental/')
	models=[int(fname.split('_ep')[1].split('.')[0]) for fname in file_names]
	models.sort()

	role_files=os.listdir(role_train_dir+'models/'+scenario+'/models/')
	role_run_no=max([int(f.split('run')[1]) for f in role_files])#int(sys.argv[6]) # 
	role_file_names=os.listdir(role_train_dir+'models/'+scenario+'/models/run'+str(role_run_no)+'/incremental/')
	role_models=[ int(fname.split('_ep')[1].split('.')[0]) for fname in role_file_names]
	role_models.sort()
	
	common_models=[i for i in role_models if i in models and i > min_model_no]

	
	eps_len=ep_len(scenario) 
	
	for model_no in common_models:
	    if True:#i%10==0:
	        # print(i)
	        model_path = train_dir+'models/'+scenario+'/models/run'+str(run_no)+'/incremental/model_ep'+str(model_no)+'.pt'
	        role_model_path = role_train_dir+'models/'+scenario+'/models/run'+str(role_run_no)+'/incremental/model_ep'+str(model_no)+'.pt'
	        os.system('python '+run_method+'.py '+scenario\
	        	+ ' --method '+method \
	            + ' --model_path ' + model_path \
	            + ' --role_model_path '+role_model_path \
	            + ' --n_episodes 3 '\
	            + ' --episode_length '+str(eps_len)\
	            + ' --model_name '+str(model_no)\
	            + ' --trajectory '+trajectory_flag\
	            + '  ')
	        # exit()#***********************************************
	

# if method in ['maac_vs_maac','role_vs_role']:
# 	train_dir=sys.argv[3]
# 	files=os.listdir(train_dir+'models/'+scenario+'/models/')
# 	run_no= max([int(f.split('run')[1])for f in files])
# 	file_names=os.listdir(train_dir+'models/'+scenario+'/models/run'+str(run_no)+'/incremental/')
# 	models=[ int(fname.split('_ep')[1].split('.')[0]) for fname in file_names]
# 	models.sort()
# 	i=0

# 	for model_no in models:
# 	    if True:#i%10==0:
# 	        # print(i)
	        
# 	        model_path = train_dir+'models/'+scenario+'/models/run'+str(run_no)+'/incremental/model_ep'+str(model_no)+'.pt'
# 	        os.system('python '+method+'.py  '+scenario+' '\
# 	            + ' --model_path ' + model_path \
# 	            + ' --n_episodes 500 '\
# 	            + ' --model_name '+str(model_no)\
# 	            + ' --trajectory 0')
# 	        # exit()
# 	    # i+=1
# 	    # print(i,model_no)
# 	    # time.sleep(5)
