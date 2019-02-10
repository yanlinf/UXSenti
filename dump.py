import pickle
import os
import argparse
from utils.vocab import *
from utils.data import *


DOMAINS = ['books', 'dvd', 'music']
TRG_DIR = 'pickle/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--lang', choices=['en', 'fr', 'de', 'jp'], default=['en', 'fr', 'de', 'jp'], nargs='+', help='lanuages to gen vocab')
    args = parser.parse_args()

    for lang in args.lang:
        vocab = Vocab(path='data/vocab_{}.txt'.format(lang))
        print('[{}] vocab size = {}'.format(lang, len(vocab)))

        vocab_f = os.path.join(TRG_DIR, 'vocab_{}.txt'.format(lang))
        check_path(vocab_f)
        with open(vocab_f, 'wb') as fout:
            pickle.dump(vocab, fout)

        for f in ['full.txt']:  # + [os.path.join(lang, dom, 'unl') for dom in DOMAINS]:
            x = load_lm_corpus(os.path.join('data', lang,  f), vocab, random_state=args.seed)
            size = x.size(0)
            print('[{}] corpus size = {}'.format(f,  size))
            print('[{}] OOV rate = {:.2f}'.format(f, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

            trg_f = os.path.join(TRG_DIR, lang, f)
            check_path(trg_f)
            with open(trg_f, 'wb') as fout:
                pickle.dump(x, fout)


if __name__ == '__main__':
    main()
