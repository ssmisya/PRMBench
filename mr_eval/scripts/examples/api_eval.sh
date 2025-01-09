source ~/.bashrc
source ~/anaconda3/bin/activate smoe

## Change Dir to PRMBench

# environment variables
new_proxy_address="your_http_proxy_address"
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

accelerate_config=./mr_eval/scripts/examples/accelerate_configs/cpu.yaml
config_file=$1


gpus=0
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval --config  ${config_file} 
