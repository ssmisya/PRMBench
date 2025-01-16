source ~/.bashrc
source ~/anaconda3/bin/activate reasoneval

job_id=4034906
export SLURM_JOB_ID=${job_id}

accelerate_config=./mr_eval/scripts/examples/accelerate_configs/4gpus.yaml
config_file=$1

gpus=4
cpus=32
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --jobid=${job_id} --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval \
--model qwen_prm \
--model_args pretrained=/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/Qwen2.5-Math-PRM-7B \
--batch_size 2 \
--task_name prmtest_classified \
--verbosity INFO \
--output_path /mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/mr_eval/scripts/logs/prmtest_classified/qwen_prm7b.jsonl 