import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import numpy as np
import re
import unicodedata

SOS_token = 0
EOS_token = 1

class Lang():
    def __init__(self, name):
        self.name = name

        self.word_to_idx = { 
            "SOS": SOS_token,
            "EOS": EOS_token,
        }

        self.idx_to_word = { 
            SOS_token: "SOS",
            EOS_token: "EOS",
        }

        self.word_to_count = { 
            SOS_token: "SOS",
            EOS_token: "EOS",
        }

    def add_sentence(self, sentence):
        words = sentence.split()
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_idx:
            idx = max(self.idx_to_word.keys()) + 1
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            self.word_to_count[word] = 0

        self.word_to_count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_file(lang1, lang2, reverse=False):
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

input_lang, output_lang, pairs = read_file("eng", "fra", True)
filtered_pairs = filter_pairs(pairs)
for pair in filtered_pairs:
    input_lang.add_sentence(pair[0])
    output_lang.add_sentence(pair[1])
