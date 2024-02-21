python -u inference.py \
    --test_fname data/spider_dev_400.json \
    --result_fname results/spider_dev_oracle_rerank_10.sql \
    --log_fname spider_dev_oracle_rerank_10.json \
    --dataset_name spider \
    --db_path ../spider/database \
    --method_name rerank \
    --generator_name codellama/CodeLlama-13b-Instruct-hf \
    --retriever_name bm25 \
    --retriever_corpus_gen data/spider_train.json \
    --retrieve_k 1 \
    --evaluator_name oracle \
    --oracle_prob 1.0 \
    --seed 42 \
    --generation_config generation_configs/temp_sampling.json \
    --evaluation_config evaluation_configs/pro.json

python -u ../test-suite-sql-eval/evaluation.py \
    --gold data/spider_dev_400_gold.sql \
    --pred results/spider_dev_oracle_rerank_10.sql \
    --db ../spider/database \
    --table ../spider/tables.json \
    --etype all