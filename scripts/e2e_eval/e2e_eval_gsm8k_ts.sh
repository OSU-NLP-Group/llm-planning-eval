export OPENAI_API_KEY=ENTER_YOUR_KEY

python -u inference_py.py \
    --test_fname data/gsm8k_dev_100.json \
    --log_fname gsm8k_dev_gpt35_pro_ts.json \
    --method_name mctot \
    --generator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_name gpt-3.5-turbo-1106 \
    --seed 42 \
    --generation_config generation_configs/temp_sampling.json \
    --evaluation_config evaluation_configs/pro.json