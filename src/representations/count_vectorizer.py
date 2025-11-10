from src.core.interfaces import Vectorizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, corpus):
        tokens = set()
        for doc in corpus:
            tokens.update(self.tokenizer.tokenize(doc))

        self.vocabulary_ = {key: i for i, key in enumerate(sorted(tokens))}

    def transform(self, documents):
        vectors = []
        for doc in documents:
            vector = [0] * len(self.vocabulary_)
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            
            vectors.append(vector)        
        return vectors
    
    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)