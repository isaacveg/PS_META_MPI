# PS_MPI

## Changes  
Refined version of ymliao98/PS_MPI   
1. Use config.yaml to simplify settings  
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
The number after should be larger than config.yaml:num_workers  
mpiexec --oversubscribe -n 1 python server.py : -n 10 python client.py  


## Save last training record to anther direction  
TODO  


## Generate pictures  
TODO  


## Clear ALL log and saved models  
WARNING: Clear all logs!  
rm -rf model_save server_log clients_log  __pycache__ 