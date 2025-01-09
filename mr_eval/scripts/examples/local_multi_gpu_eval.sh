source ~/.bashrc
source ~/anaconda3/bin/activate reasoneval

new_proxy_address=your_http_proxy_address
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address


job_id=3962886
export SLURM_JOB_ID=${job_id}


accelerate_config=./mr_eval/scripts/examples/accelerate_configs/4gpus.yaml
config_file=$1

gpus=4
cpus=32
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --jobid=${job_id} --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval --config  ${config_file} 


# salloc --partition=MoE --job-name="eval" --gres=gpu:8 -n1 --ntasks-per-node=1 -c 64 --quotatype="reserved"
# salloc --partition=MoE --job-name="interact" --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --quotatype="reserved"
