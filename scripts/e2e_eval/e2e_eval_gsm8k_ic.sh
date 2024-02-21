python -u inference_py.py \
    --test_fname data/gsm8k_dev_500.json \
    --log_fname gsm8k_dev_cl13b_pro_ic.json \
    --method_name iter_corr \
    --generator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_name codellama/CodeLlama-13b-Instruct-hf \
    --seed 42 \
    --generation_config generation_configs/temp_sampling.json \
    --evaluation_config evaluation_configs/pro.json