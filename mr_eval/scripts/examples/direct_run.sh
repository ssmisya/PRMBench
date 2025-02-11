accelerate_config=./mr_eval/scripts/examples/accelerate_configs/1gpu.yaml

accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval \
--model pure_prm \
--model_args pretrained=/mnt/petrelfs/chengjie/ceph6/qwen25-math-7b-PRM800k-bs128-lr1e-6-epoch-1-stage2 \
--batch_size 2 \
--task_name prmtest_classified \
--verbosity INFO \
--output_path ./scripts/logs/prmtest_classified/pure_prm_7b.jsonl