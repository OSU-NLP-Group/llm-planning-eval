from utils.exec_eval import eval_exec_match
from utils.normalize_sql import normalize_sql

from tqdm import tqdm

import argparse
import json
import sqlite3


#DB_FOLDER = "../spider/database"


def preproc_ex(example):
    processed_ex = {
        "db_id": example["db_id"],
        "schema": example["schema"],
        "question": example["question"],
        "sql": example["sql"],
        "exec_res": example["exec_res"] #execute_sql(cursor, example["sql"])
    }

    for i in range(len(example["top_n"])):
        if example["top_n_label"][i] == 0 and example["top_n_exec_res"][i] != example["exec_res"]:
            processed_ex["neg_sql"] = example["top_n"][i]
            processed_ex["neg_exec_res"] = example["top_n_exec_res"][i]
            break

    return processed_ex


def preprocess(args):
    data = [
        preproc_ex(example) 
        for example in tqdm(json.load(open(args.data_fname)))
    ]

    # Keep those with negative examples
    data = [
        ex for ex in data if "neg_sql" in ex
    ]

    print(len(data))

    out = open(args.out_fname, "w+")
    json.dump(data, out, indent=2)
    out.close()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_fname', type=str, default='')
    args_parser.add_argument('--out_fname', type=str, default='')
    args = args_parser.parse_args()

    preprocess(args)