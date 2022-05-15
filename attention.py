import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import random
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

        self.n = 2

    def add_sentence(self, sentence):
        words = sentence.split()
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n
            self.idx_to_word[self.n] = word
            self.word_to_count[word] = 0
            self.n += 1
            
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

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden # output is used *only* if we have attention

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

input_size = input_lang.n
output_size = output_lang.n
hidden_size = 128
epochs = 10
batch_size = 32
teacher_forcing_ratio = 0.5

encoder = Encoder(input_size, hidden_size)
decoder = Encoder(hidden_size, output_size)
loss_criterion = nn.NLLLoss()

encoder_optim = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optim = optim.SGD(decoder.parameters(), lr=0.01)

for epoch in range(epochs):
    training_batch = [random.choice(filtered_pairs) for _ in range(batch_size)]
    training_batch = [
        (
            torch.tensor([SOS_token] + [input_lang.word_to_idx[word]  for word in sentence_pair[0].split()] + [EOS_token]),
            torch.tensor([SOS_token] + [output_lang.word_to_idx[word] for word in sentence_pair[1].split()] + [EOS_token]),
        )
        for sentence_pair in training_batch
    ]

    for training_pair in training_batch:
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        loss = 0
        
        input_sentence, output_sentence = training_pair
        
        hidden_state = torch.zeros(hidden_size)
        for word in input_sentence:
            _, hidden_state = encoder(word, hidden_state)

        prev_word = [SOS_token]
        for word in output_sentence:
            output, hidden_state = decoder(prev_word, hidden_state)
            loss += loss_criterion(output, word)

            if random.random() < teacher_forcing_ratio:
                prev_word = word
            else:
                topv, topi = output.topk(1)
                prev_word = topi.squeeze().detach()

            if prev_word == EOS_token:
                break

        loss.backward()

        encoder_optim.step()
        decoder_optim.step()

    print(f"Epoch: {epoch} / {epochs} -- Loss : {loss}")