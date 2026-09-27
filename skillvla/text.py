"""Word-level tokenizer over the closed instruction vocabulary."""
import re

MAX_LEN = 20


class Tokenizer:
    def __init__(self, sentences):
        words = sorted({w for s in sentences for w in self.split(s)})
        self.vocab = {w: i + 1 for i, w in enumerate(words)}  # 0 = pad

    @staticmethod
    def split(s):
        return re.findall(r"[a-z]+|[.:]", s.lower())

    def encode(self, s):
        ids = [self.vocab[w] for w in self.split(s)][:MAX_LEN]
        return ids + [0] * (MAX_LEN - len(ids))

    def __len__(self):
        return len(self.vocab) + 1
