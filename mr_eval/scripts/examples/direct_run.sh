accelerate_config=./mr_eval/scripts/examples/accelerate_configs/1gpu.yaml

accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval \
--model pure_prm \
--batch_size 2 \
--num_workers 2 \
--task_name prmtest_classified \
--verbosity INFO \
--output_path ./mr_eval/scripts/logs/prmtest_classified/pure_prm_7b.jsonl