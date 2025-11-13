import numpy as np
import gensim
import gensim.downloader

class WordEmbedder():
    def __init__(self, model_name: str):
        self.model = gensim.downloader.load(model_name)
        self.dim = len(self.model['the'])

    def get_vector(self, word: str):
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            print(f'Word "{word}" is not in model.')
            return np.zeros(shape=self.dim)

    def get_similarity(self, word1: str, word2: str):
        if word1 not in self.model.key_to_index:
            print(f'Word "{word1}" is not in model.')
            return 0.0
        
        if word2 not in self.model.key_to_index:
            print(f'Word "{word2}" is not in model.')
            return 0.0
        
        return self.model.similarity(word1, word2)

    def get_most_similar(self, word: str, top_n: int = 10):
        if word not in self.model.key_to_index:
            print(f'Word "{word}" is not in model.')
            return []
        return self.model.most_similar(word, topn=top_n)
    
    def embed_document(self, document: str, tokenizer):
        tokens = tokenizer.tokenize(document)
        word_count = 0
        accum = np.zeros(shape=self.dim, dtype=float)
        for token in tokens:
            if token in self.model.key_to_index:
                accum += self.model[token]
                word_count +=1
        
        if word_count == 0:
            return accum
        
        return accum / word_count