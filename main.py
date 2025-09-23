from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

text_1 = 'Hello, world! This is a test.'
text_2 = "NLP is fascinating... isn't it?"
text_3 = "Let's see how it handles 123 numbers and punctuation!"

print('Text 1:', text_1)

simple_tokens_1 = simple_tokenizer.tokenize(text=text_1)
print('Simple tokens:', simple_tokens_1)

regex_tokens_1 = regex_tokenizer.tokenize(text=text_1)
print('Regex tokens:', regex_tokens_1)

print('----------')
print('Text 2:', text_2)

simple_tokens_2 = simple_tokenizer.tokenize(text=text_2)
print('Simple tokens:', simple_tokens_2)

regex_tokens_2 = regex_tokenizer.tokenize(text=text_2)
print('Regex tokens:', regex_tokens_2)

print('----------')
print('Text 3:', text_3)

simple_tokens_3 = simple_tokenizer.tokenize(text=text_3)
print('Simple tokens:', simple_tokens_3)

regex_tokens_3 = regex_tokenizer.tokenize(text=text_3)
print('Regex tokens:', regex_tokens_3)