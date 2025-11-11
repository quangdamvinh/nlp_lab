from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

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
print('--------------------')
print()

print('Task 3:')
dataset_path = 'datasets/UD_English-EWT/en_ewt-ud-train.txt'
raw_text = load_raw_text_data(dataset_path)
sample_text = raw_text[:500]

print("--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")

simple_tokens = simple_tokenizer.tokenize(sample_text)
print('----------')
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

regex_tokens = regex_tokenizer.tokenize(sample_text)
print('----------')
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")