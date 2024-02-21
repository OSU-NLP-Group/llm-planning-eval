python -u inference_py.py \
    --test_fname data/gsm8k_dev_100.json \
    --log_fname gsm8k_dev_oracle_10_ts.json \
    --method_name mctot \
    --generator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_name oracle \
    --oracle_prob 1.0 \
    --seed 42 \
    --generation_config generation_configs/temp_sampling.json \
    --evaluation_config evaluation_configs/pro.json