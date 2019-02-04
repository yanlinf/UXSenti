import argparse
import os
from utils.vocab import *

SRC_DIR = 'data/'
LANGS = ['en', 'fr', 'de', 'jp']
DOMAINS = ['books', 'dvd', 'music']
PART = ['train.review', 'test.review', 'unlabeled.review']
TRG_DIR = 'data/'
EXTRA_TOKENS = ['<eos>', '<unk>']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10000, help='vocabulary size')
    parser.add_argument('--encoding', default='utf-8', help='encoding format')
    args = parser.parse_args()

    if not os.path.exists(TRG_DIR):
        os.makedirs(TRG_DIR)

    for lang in LANGS:
        corpus = []
        for dom in DOMAINS:
            for part in PART:
                path = os.path.join(SRC_DIR, lang, dom, part)
                with open(path, 'r', encoding=args.encoding) as fin:
                    corpus += [row.rstrip() for row in fin]

        vocab = Vocab(corpus)
        vocab.cutoff(args.size)
        for tok in EXTRA_TOKENS:
            vocab.add_word(tok)
        vocab.save(os.path.join(TRG_DIR, 'vocab_{}.txt'.format(lang)))

if __name__ == '__main__':
    main()
