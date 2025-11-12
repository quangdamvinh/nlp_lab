from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
    ]

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)
vectors = vectorizer.fit_transform(corpus)

print('Learned vocabulary:', vectorizer.vocabulary_)
print('Document-term matrix:')
print(vectors)