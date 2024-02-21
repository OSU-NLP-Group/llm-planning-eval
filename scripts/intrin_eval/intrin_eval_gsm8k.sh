export OPENAI_API_KEY=ENTER_YOUR_KEY

python -u intrin_eval_py.py \
    --test_fname data/gsm8k_intrin_eval.json \
    --log_fname gsm8k_intrin_eval_gpt35_pro.json \
    --evaluator_name gpt-3.5-turbo-1106 \
    --evaluation_config evaluation_configs/pro.json \
    --seed 42