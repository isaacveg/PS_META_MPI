# PS_META_MPI

## Attention   
本仓库不再更新，新版已经迁移至 github.com/isaacveg/fireworks
已升级成面向对象版本，更加简洁方便

## Changes  
Federated Meta-learning version of isaacveg/PS_MPI   
1. Use config.yml to simplify settings  
2. Simplified process to make it general  
3. Cut out useless code to make it elegant  
4. add your own datasets and models  

## Own dataset  
1. create corresponding dataset_name.py in /datasets  
2. define it in /datasets/dataset_name.py and return train\test dataset transformed  
3. import it in /datasets/__init__.py


## Own model
1. create corresponding model_name.py in /models  
2. define it in /models/model_name.py and return model class 
3. import it in /models/__init__.py and decide when to use your model in create_model_instance()


## Run
The number after should be larger than config.yml : selected_num + eval_clients_num  
mpiexec --oversubscribe -n 1 python server.py : -n 12 python client.py  


## Save last training record to anther directory 
This will move all results and config to /results  
python ./analysis_tools/gathering_results.py


## Generate pictures  
This will plot all results from server_logs and save to /plots  
python ./analysis_tools/draw_pics.py


## Clear ALL log and saved models  
WARNING: Clear all logs!  
rm -rf model_save server_log clients_log config_save  __pycache__ 


## Debug
1. Only the server:
python server.py
2. Only the client:
TODO

## 