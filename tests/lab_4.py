from src.representations.word_embedder import WordEMbedder

print('Lab 4 test:')

word_embedder = WordEMbedder('glove-wiki-gigaword-50')

print('Vector for the word "king"', word_embedder.get_vector('king'))