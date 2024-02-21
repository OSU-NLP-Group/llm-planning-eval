from nltk import word_tokenize
from rank_bm25 import BM25Okapi

import json

class BM25Retriever():

    def __init__(self, model_name_or_dir, corpus_fname, retrieve_k):
        self.corpus = json.load(open(corpus_fname))
        self.corpus_questions = [ex["question"] for ex in self.corpus]
        self.corpus_index = dict([(ex["question"], i) for i, ex in enumerate(self.corpus)])
        self.model = BM25Okapi(self.corpus_questions, tokenizer=word_tokenize)
        self.retrieve_k = retrieve_k


    def retrieve(self, query):
        top_questions = self.model.get_top_n(word_tokenize(query), self.corpus_questions, n=self.retrieve_k)
        return [self.corpus[self.corpus_index[q]] for q in top_questions]
