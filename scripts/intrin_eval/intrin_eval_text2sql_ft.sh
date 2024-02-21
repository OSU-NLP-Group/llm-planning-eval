python -u intrin_eval.py \
    --test_fname data/spider_intrin_eval.json \
    --log_fname spider_intrin_eval_cl13b_ft_pro.json \
    --dataset_name spider \
    --db_path ../spider/database \
    --evaluator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_peft_dir /research/nfs_sun_397/chen.8336/codellama/spider-evaluator-cls-exec-42 \
    --evaluation_config evaluation_configs/pro.json \
    --seed 42