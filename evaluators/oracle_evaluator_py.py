from utils.exec_py import exec_py

import random


class OracleEvaluatorPy():

    def __init__(self, oracle_prob):
        self.oracle_prob = oracle_prob


    def score(self, question, candidates, gold_answer):
        
        scores = []
        for cand_py in candidates:
            try:
                answer = exec_py(cand_py)
            except:
                if random.random() < self.oracle_prob:
                    scores.append(1)
                else:
                    scores.append(-1)
                continue

            if answer is None:
                if random.random() < self.oracle_prob:
                    scores.append(0)
                else:
                    scores.append(-1)
            else:
                try:
                    s = int(answer == gold_answer) # rare case: answer is not number
                    if random.random() < self.oracle_prob:
                        scores.append(-s)
                    else:
                        scores.append(s - 1)
                except:
                    if random.random() < self.oracle_prob:
                        scores.append(0)
                    else:
                        scores.append(-1)

        return scores
