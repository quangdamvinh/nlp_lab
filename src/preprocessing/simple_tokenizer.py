import re
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\w+|[.,?!]", text)

        return tokens