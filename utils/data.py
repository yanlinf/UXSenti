import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import collections
import argparse

UNK_TOK = '<unk>'
EOS_TOK = '<eos>'
PAD_TOK = '<pad>'


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


def load_vectors(path, maxload=-1):
    """

    """
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split(' '))
        if maxload > 0:
            n = min(n, maxload)
        x = np.zeros([n, d])
        words = []
        for i, line in enumerate(fin):
            if i >= n:
                break
            tokens = line.rstrip().split(' ')
            words.append(tokens[0])
            x[i] = np.array(tokens[1:], dtype=float)

    return words, x


def shuffle(random_state, *args):
    """
    random_state: int
    args: List[Tensor]

    returns: List[Tensor]
    """
    torch.manual_seed(random_state)
    size = args[0].size(0)
    perm = torch.randperm(size)
    res = [x[perm] for x in args]
    return res


def load_senti_corpus(path, vocab, encoding='utf-8', maxlen=512, random_state=None, labels=['__pos__', '__neg__']):
    """
    path: str
    vocab: Vocab
    encoding: str
    maxlen: int
    random_state: int
    labels: List[str]

    returns: LongTensor of shape (size, maxlen), LongTensor of shape (size,)
    """
    corpus, y = [], []
    l2i = {l: i for i, l in enumerate(labels)}
    with open(path, 'r', encoding=encoding) as fin:
        for line in fin:
            label, text = line.rstrip().split(' ', 1)
            y.append(l2i[label])
            corpus.append([vocab.w2idx[w] if w in vocab else vocab.w2idx[UNK_TOK]
                           for w in text.split(' ')] + [vocab.w2idx[EOS_TOK]])
    size = len(corpus)
    X = torch.full((size, maxlen), vocab.w2idx[PAD_TOK], dtype=torch.int64)
    l = torch.empty(size, dtype=torch.int64)
    y = torch.tensor(y)
    for i, xs in enumerate(corpus):
        sl = min(len(xs), maxlen)
        l[i] = sl
        X[i, :sl] = torch.tensor(xs[:sl])
    if random_state is not None:
        X, y, l = shuffle(random_state, X, y, l)
    return X, y, l


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
                words = line.rstrip().split(' ') + [EOS_TOK]
                ntokens += len(words)

        # second pass: convert tokens to ids
        with open(path, 'r', encoding=encoding) as f:
            ids = torch.LongTensor(ntokens)
            p = 0
            for line in f:
                words = line.rstrip().split(' ') + [EOS_TOK]
                for w in words:
                    if w in vocab.w2idx:
                        ids[p] = vocab.w2idx[w]
                    else:
                        ids[p] = vocab.w2idx[UNK_TOK]
                    p += 1

    else:
        with open(path, 'r', encoding=encoding) as f:
            corpus = [line.rstrip() + ' ' + EOS_TOK for line in f]

        ntokens = sum(len(line.split(' ')) for line in corpus)
        ids = torch.LongTensor(ntokens)
        p = 0
        np.random.seed(random_state)
        for i in np.random.permutation(len(corpus)):
            for w in corpus[i].split(' '):
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


def get_batch(source, i, bptt, seq_len=None, evaluation=False, batch_first=False, cuda=False):
    assert isinstance(bptt, int)

    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    if batch_first:
        data = source[i:i + seq_len].t()
        target = source[i + 1:i + 1 + seq_len].t().contiguous().view(-1)
    else:
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)

    if cuda:
        data = data.cuda()
        target = target.cuda()

    return data, target


class LMDataset(Dataset):

    def __init__(self, X, batch_size, redundant=10):
        super().__init__()
        size = X.size(0) // batch_size
        X = X.narrow(0, 0, size * batch_size)
        X = X.view(batch_size, -1).t().contiguous()
        self.X = X
        self.redundant = redundant

    def __len__(self):
        return self.X.size(0) - self.redundant

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx + 1]


class SentiDataset(Dataset):

    def __init__(self, X, y, l):
        super().__init__()
        self.X, self.y, self.l = X, y, l

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.l[idx]
