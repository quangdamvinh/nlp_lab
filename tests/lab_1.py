from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

print('Task 1 test:')

text = 'Hello, world!'
print('Input text:', text)

simple_tokenizer = SimpleTokenizer()
tokens = simple_tokenizer.tokenize(text=text)
print('Output tokens:', tokens)

print('Task 2 test:')
regex_tokenizer = RegexTokenizer()
regex_tokens = regex_tokenizer.tokenize(text=text)
print('Output regex tokens:', regex_tokens)