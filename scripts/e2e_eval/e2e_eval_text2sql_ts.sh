python -u inference.py \
    --test_fname data/spider_dev_400.json \
    --result_fname results/spider_dev_cl13b_ft_pro_ts.sql \
    --log_fname spider_dev_cl13b_ft_pro_ts.json \
    --dataset_name spider \
    --db_path ../spider/database \
    --method_name mctot \
    --generator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_name codellama/CodeLlama-13b-Instruct-hf \
    --evaluator_peft_dir /research/nfs_sun_397/chen.8336/codellama/spider-evaluator-cls-exec-42 \
    --retriever_name bm25 \
    --retriever_corpus_gen data/spider_train.json \
    --retrieve_k 1 \
    --seed 42 \
    --generation_config generation_configs/temp_sampling.json \
    --evaluation_config evaluation_configs/pro.json

python -u ../test-suite-sql-eval/evaluation.py \
    --gold data/spider_dev_400_gold.sql \
    --pred results/spider_dev_oracle_rerank_10.sql \
    --db ../spider/database \
    --table ../spider/tables.json \
    --etype all