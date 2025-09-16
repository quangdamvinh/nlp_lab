from src.preprocessing.simple_tokenizer import SimpleTokenizer

print('Task 1 test:')

text = 'Hello, world!'
print('Input text:', text)

simple_tokenizer = SimpleTokenizer()
tokens = simple_tokenizer.tokenize(text=text)
print('Output tokens:', tokens)