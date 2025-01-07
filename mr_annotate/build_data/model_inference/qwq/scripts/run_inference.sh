source ~/.bashrc
source ~/anaconda3/bin/activate smoe

# environment variables
export OMP_NUM_THREADS=8
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export HF_ENDPOINT=https://hf-mirror.com

code_base=/mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/mr_annotate/build_data/model_inference/qwq
cd $code_base

export SLURM_JOB_ID=3873423 
# unset SLURM_JOB_ID      

gpus=1
cpus=16
quotatype="auto"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="inference" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/mr_annotate/build_data/model_inference/qwq/run_inference.py \
--model_path /mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/QwQ-32B-Preview \
--input_path /mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/mr_annotate/build_data/selection_of_data/prm_correct_data/prm_test_p1.jsonl \
--output_path /mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/mr_annotate/build_data/selection_of_data/new_8_classes/one_question_multi_answer/prm_test_p1_qwq.jsonl \
