import argparse
import os
from utils.vocab import *
from utils.data import *

SRC_DIR = 'data/'
DOMAINS = ['books', 'dvd', 'music']
PART = ['train.review', 'test.review', 'unlabeled.review']
TRG_DIR = 'data/'
EXTRA_TOKENS = ['<eos>', '<unk>']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10000, help='vocabulary size')
    parser.add_argument('--fasttext', action='store_true', help='use fasttext')
    parser.add_argument('--lang', choices=['en', 'fr', 'de', 'ja'], default=['en', 'fr', 'de', 'ja'], nargs='+', help='lanuages to gen vocab')
    parser.add_argument('--encoding', default='utf-8', help='encoding format')
    args = parser.parse_args()

    if not os.path.exists(TRG_DIR):
        os.makedirs(TRG_DIR)

    for lang in args.lang:
        if args.fasttext:
            words, _ = load_vectors('data/wiki.{}.vec'.format(lang), maxload=(args.size - len(EXTRA_TOKENS)))
            vocab = Vocab()
            for w in words:
                vocab.add_word(w)
        else:
            corpus = []
            for dom in DOMAINS:
                for part in PART:
                    path = os.path.join(SRC_DIR, lang, dom, part)
                    with open(path, 'r', encoding=args.encoding) as fin:
                        corpus += [row.rstrip() for row in fin]
            vocab = Vocab(corpus)
            vocab.cutoff(args.size - len(EXTRA_TOKENS))

        for tok in EXTRA_TOKENS:
            vocab.add_word(tok)
        vocab.save(os.path.join(TRG_DIR, 'vocab_{}.txt'.format(lang)))

if __name__ == '__main__':
    main()
