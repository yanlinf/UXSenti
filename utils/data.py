import torch
import os
import json
import numpy as np
import collections
import argparse
from .vocab import *

UNK_TOK = '<unk>'
EOS_TOK = '<eos>'


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_lm_corpus(path, vocab, encoding='utf-8', random_state=None):
    """
    path: str
    vocab: Vocab
    encoding: str
    random_state: int (optional)

    returns: torch.LongTensor of shape (corpus_size,)
    """
    if random_state is None:
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

    else:
        with open(path, 'r', encoding=encoding) as f:
            corpus = [line.rstrip() for line in f]

        ntokens = sum(len(line.split()) for line in corpus)
        ids = torch.LongTensor(ntokens)
        p = 0
        np.random.seed(random_state)
        for i in np.random.permutation(len(corpus)):
            for w in corpus[i].split():
                if w in vocab.w2idx:
                    ids[p] = vocab.w2idx[w]
                else:
                    ids[p] = vocab.w2idx[UNK_TOK]
                p += 1

    return ids


def load_lexicon(path, src_vocab, trg_vocab, encoding='utf-8', verbose=False):
    """
    path: str
    src_vocab: Vocab
    trg_vocab: Vocab
    encoding: str
    verbose: bool

    returns: collections.defautldict
    """
    lexicon = collections.defaultdict(set)
    vocab = set()
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            src, trg = line.rstrip().split()
            if src in src_vocab and trg in trg_vocab:
                lexicon[src_vocab.w2idx[src]].add(trg_vocab.w2idx[trg])
            vocab.add(src)
    if verbose:
        print('[{}] OOV rate = {:.4f}'.format(path, 1 - len(lexicon) / len(vocab)))

    return lexicon, len(vocab)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    size = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, size * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data  # shape (size, batch_size)


def get_batch(source, i, bptt, seq_len=None, evaluation=False, batch_first=False):
    assert isinstance(bptt, int)

    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    if batch_first:
        data = source[i:i + seq_len].t()
        target = source[i + 1:i + 1 + seq_len].t().contiguous().view(-1)
    else:
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target
