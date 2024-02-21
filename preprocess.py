from utils.normalize_sql import normalize_sql

import argparse
import json
import sqlite3


DB_FOLDER = "../spider/database"


def read_schema(db_dir, db_id):
    schema = {}
    db_path = f'{db_dir}/{db_id}/{db_id}.sqlite'
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            results = cursor.fetchall()
            schema[table_name] = [r[1] for r in results]

    return "-- Table " + "\n-- Table ".join(
        [t.lower() + ": " + ", ".join([c.lower() for c in schema[t]]) for t in schema]
    )


def preproc_ex(example, db_dir, dataset_name):
    return {
        "db_id": example["db_id"],
        "schema": read_schema(db_dir, example["db_id"]),
        "question": (example["question"] if dataset_name == "spider" else example["question"] + " " + example["evidence"]),
        "sql": normalize_sql(example["query"] if dataset_name == "spider" else example["SQL"])
    }


def preprocess(args):
    train = [preproc_ex(example, args.database_dir, args.dataset_name) for example in json.load(open(args.raw_train_fname)) if example["db_id"] != "retail_world"]
    dev = [preproc_ex(example, args.database_dir, args.dataset_name) for example in json.load(open(args.raw_dev_fname))]

    print(len(train))
    print(len(dev))

    out = open(f"data/{args.dataset_name}_train.json", "w+")
    json.dump(train, out, indent=2)
    out.close()
    out = open(f"data/{args.dataset_name}_dev.json", "w+")
    json.dump(dev, out, indent=2)
    out.close()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--raw_train_fname', type=str, default='../spider/train_spider.json')
    args_parser.add_argument('--raw_dev_fname', type=str, default='../spider/dev.json')
    args_parser.add_argument('--dataset_name', type=str, default='spider')
    args_parser.add_argument('--database_dir', type=str, default='../spider/database')
    args = args_parser.parse_args()

    preprocess(args)