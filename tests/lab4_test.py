from src.representations.word_embedder import WordEmbedder
from src.preprocessing.regex_tokenizer import RegexTokenizer

word_embedder = WordEmbedder('glove-wiki-gigaword-50')
tokenizer = RegexTokenizer()

print('Lab 4 test:')
print('Vector for the word "king":', word_embedder.get_vector('king'))
print('Similarity between "king" and "queen":', word_embedder.get_similarity('king', 'queen'))
print('Similarity between "king" and "man":', word_embedder.get_similarity('king', 'man'))
print('10 most similar words to "computer":', word_embedder.get_most_similar('computer'))
print('Document: The queen rules the country.')
print('Vector:', word_embedder.embed_document(document='The queen rules the country.', tokenizer=tokenizer))