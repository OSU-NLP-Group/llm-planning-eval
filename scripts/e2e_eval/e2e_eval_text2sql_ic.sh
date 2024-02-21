python -u inference.py \
    --test_fname data/bird_dev_300.json \
    --result_fname results/bird_dev_cl13b_base_ic.json \
    --log_fname bird_dev_cl13b_base_ic.json \
    --dataset_name bird \
    --db_path ../bird-sql/databases \
    --method_name iter_corr \
    --generator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_name codellama/CodeLlama-13b-Instruct-hf \
    --retriever_name bm25 \
    --retriever_corpus_gen data/bird_train.json \
    --retriever_corpus_eval data/bird_evaluator_train_prompt.json \
    --retrieve_k 1 \
    --seed 42 \
    --generation_config generation_configs/temp_sampling.json \
    --evaluation_config evaluation_configs/base.json

python -u ../bird-sql/evaluation.py \
    --db_root_path ../bird-sql/databases/ \
    --predicted_sql_path results/bird_dev_cl13b_base_ic.json \
    --ground_truth_path data/bird_dev_300_gold.sql \
    --num_cpus 16 \
    --time_out 60 \
    --mode_gt "gt" \
    --mode_predict "gpt" \
    --diff_json_path data/bird_dev_300.json