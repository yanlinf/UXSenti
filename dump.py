import pickle
import os
import argparse
from utils.vocab import *
from utils.data import *


LANGS = ['en', 'fr', 'de', 'jp']
DOMAINS = ['books', 'dvd', 'music']
TRG_DIR = 'pickle/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()

    for lang in LANGS:
        vocab = Vocab(path='data/vocab_{}.txt'.format(lang))
        print('[{}] vocab size = {}'.format(lang, len(vocab)))

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
