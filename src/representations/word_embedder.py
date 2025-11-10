import numpy as np
import gensim

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
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        return np.cos(v1, v2)

    def get_most_similar(self, word: str, top_n: int = 10):
        if word not in self.model.key_to_index:
            print(f'Word "{word}" is not in model.')
            return []
        return self.model.most_similar(word, topn=top_n)