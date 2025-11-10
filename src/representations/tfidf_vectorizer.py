from math import log
from src.core.interfaces import Vectorizer

class TfidfVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, corpus):
        df = {}
        total_docs = len(corpus)

        for doc in corpus:
            tokens = set(self.tokenizer.tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        self.vocabulary_ = {token: i for i, token in enumerate(sorted(df.keys()))}

        self.idf_ = {token: log(total_docs / (df[token] + 1)) + 1 for token in self.vocabulary_}

    def transform(self, documents):
        vectors = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            tf = {}

            for token in tokens:
                if token in self.vocabulary_:
                    tf[token] = tf.get(token, 0) + 1

            vector = [0.0] * len(self.vocabulary_)

            for token, freq in tf.items():
                idx = self.vocabulary_[token]
                vector[idx] = freq * self.idf_[token]

            vectors.append(vector)

        return vectors

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
