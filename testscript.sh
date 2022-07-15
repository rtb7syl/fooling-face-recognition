#!/usr/local_rwth/bin/zsh


source ~/.zshrc

pwd; hostname; date

# Define a timestamp function
timestamp() {
  date +"%T" # current time
}

# do something...
timestamp # print timestamp
# do something else...
timestamp # print another timestamp
# continue...

conda activate tf_gpu
cd patch
###### THIS IS THE REAL CODE 
python test_impersonation.py
