from utils.exec_eval import eval_exec_match
from utils.normalize_sql import normalize_sql

from func_timeout import func_timeout
from tqdm import tqdm

import argparse
import json
import sqlite3


def execute_sql(cursor, sql_str):
    try:
        cursor.execute(sql_str)
    except:
        return "ERROR"

    result = [(c[0], []) for c in cursor.description]
    rows = []
    for i in range(5):
        row = cursor.fetchone()
        if row is None:
            break
        rows.append(row)

    if i == 0:
        return "None"

    for values in rows:
        for c, v in zip(result, values):
            c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

    result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
    return result


def preproc_ex(example, database_dir):
    db_id = example["db_id"]
    db_path = f'{database_dir}/{db_id}/{db_id}.sqlite'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    processed_ex = {
        "db_id": example["db_id"],
        "schema": example["schema"],
        "question": example["question"],
        "sql": example["sql"],
        "exec_res": execute_sql(cursor, example["sql"]),
        "top_n": [normalize_sql(s) for s in example["top_n"]],
        "top_n_exec_res": [execute_sql(cursor, s) for s in example["top_n"]],
    }

    top_n_label = []
    for s in example["top_n"]:
        try:
            label = func_timeout(60.0, eval_exec_match, args=(db_path, s, example["sql"], False, False, False))
            top_n_label.append(label)
        except:
            top_n_label.append(0)
    processed_ex["top_n_label"] = top_n_label

    return processed_ex


def preprocess(args):
    data = []
    for example in tqdm(json.load(open(args.data_fname))):
        try:
            ex = func_timeout(180.0, preproc_ex, args=(example, args.database_dir))
            data.append(ex)
        except:
            continue

    print(len(data))

    out = open(args.out_fname, "w+")
    json.dump(data, out, indent=2)
    out.close()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_fname', type=str, default='')
    args_parser.add_argument('--out_fname', type=str, default='')
    args_parser.add_argument('--dataset_name', type=str, default='spider')
    args_parser.add_argument('--database_dir', type=str, default='')
    args = args_parser.parse_args()

    preprocess(args)