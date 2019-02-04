import torch
import os
from .vocab import *

UNK_TOK = '<unk>'
EOS_TOK = '<eos>'


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def load_lm_corpus(path, vocab, encoding='utf-8'):

    # first pass: count the number of tokens
    with open(path, 'r', encoding=encoding) as f:
        ntokens = 0
        for line in f:
            words = line.rstrip().split() + [EOS_TOK]
            ntokens += len(words)

    # second pass: convert tokens to ids
    with open(path, 'r', encoding=encoding) as f:
        ids = torch.LongTensor(ntokens)
        p = 0
        for line in f:
            words = line.rstrip().split() + [EOS_TOK]
            for w in words:
                if w in vocab.w2idx:
                    ids[p] = vocab.w2idx[w]
                else:
                    ids[p] = vocab.w2idx[UNK_TOK]
                p += 1

    return ids


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    size = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, size * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data  # shape (size, batch_size)


def get_batch(source, i, bptt, seq_len=None, evaluation=False):
    assert isinstance(bptt, int)

    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target
