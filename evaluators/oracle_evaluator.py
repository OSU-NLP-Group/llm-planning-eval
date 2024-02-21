from collections import Counter
from copy import deepcopy

import numpy as np
import random
import sqlite3
import torch


class OracleEvaluator():

    def __init__(self, db_path, oracle_prob):
        self.db_path = db_path
        self.oracle_prob = oracle_prob


    def score(self, db_id, question, candidates, gold_sql):
        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        try:
            cursor.execute(gold_sql)
        except:
            return [1 for cand_sql in candidates]

        gold_res = []
        for i in range(5):
            row = cursor.fetchone()
            if row is None:
                break
            gold_res.append(Counter(row))

        scores = []
        for cand_sql in candidates:
            try:
                cursor.execute(cand_sql)
            except:
                if random.random() < self.oracle_prob:
                    scores.append(1)
                else:
                    scores.append(-1)
                continue

            cand_res = []
            for i in range(5):
                row = cursor.fetchone()
                if row is None:
                    break
                cand_res.append(Counter(row))

            if len(gold_res) == 0:
                if len(cand_res) == 0:
                    if random.random() < self.oracle_prob:
                        scores.append(-1)
                    else:
                        scores.append(0)
                else:
                    if random.random() < self.oracle_prob:
                        scores.append(0)
                    else:
                        scores.append(-1)
            else:
                overlap = 0
                for i, row in enumerate(cand_res):
                    if i >= len(gold_res):
                        break
                    
                    gold_row = gold_res[i]
                    for col in row:
                        if col in gold_row:
                            overlap += row[col]
                
                gold_sum = sum([sum([v for v in gold_row.values()]) for gold_row in gold_res])
                cand_sum = sum([sum([v for v in cand_row.values()]) for cand_row in cand_res])

                prec = overlap / gold_sum
                rec = (overlap / cand_sum) if cand_sum > 0 else 0

                if random.random() < self.oracle_prob:
                    scores.append(
                        - (2 * prec * rec / (prec + rec + 1e-8))
                    )
                else:
                    scores.append(
                        (2 * prec * rec / (prec + rec + 1e-8)) - 1
                    )

        return scores
