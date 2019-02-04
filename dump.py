import pickle
import os
from utils.vocab import *
from utils.data import *


LANGS = ['en', 'fr', 'de', 'jp']
TRG_DIR = 'pickle/'


def main():
    for lang in LANGS:
        vocab = Vocab(path='data/vocab_{}.txt'.format(lang))
        print('[{}] vocab size = {}'.format(lang, len(vocab)))

        x = load_lm_corpus(os.path.join('data', lang, 'full.txt'), vocab)
        size = x.size(0)
        print('[{}] corpus size = {}'.format(lang, size))
        print('[{}] OOV rate = {}'.format(lang, x.bincount()[vocab.w2idx[UNK_TOK]] / size))

        trg_f = os.path.join(TRG_DIR, lang)
        check_path(trg_f)
        with open(trg_f, 'wb') as fout:
            pickle.dump(x, fout)


if __name__ == '__main__':
    main()
