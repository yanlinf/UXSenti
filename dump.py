import pickle
import os
import argparse
from utils.vocab import *
from utils.data import *

LANGS = ['en', 'fr', 'de', 'ja']
DOMAINS = ['books', 'dvd', 'music']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--maxlen', type=int, default=512, help='random seed')
    parser.add_argument('--data', default='data/', help='output file')
    parser.add_argument('--vocab', default='data/vocab_{}.txt', help='output file')
    parser.add_argument('-o', '--output', default='pickle/amazon.dataset', help='output file')
    args = parser.parse_args()
    print(str(args))

    dataset = {lang: {dom: {} for dom in DOMAINS} for lang in LANGS}

    for lang in LANGS:
        vocab = Vocab(path=args.vocab.format(lang))
        dataset[lang]['vocab'] = vocab
        x = load_lm_corpus(os.path.join(args.data, lang, 'full.txt'), vocab, random_state=args.seed)
        dataset[lang]['full'] = x
        size = x.size(0)
        print('[{}] vocab size = {}'.format(lang, len(vocab)))
        print('[{}] corpus size = {}'.format(lang,  size))
        print('[{}] corpus OOV rate = {:.2f}'.format(lang, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

        for dom in DOMAINS:
            # load unlabeled data
            f = os.path.join(args.data, lang, dom, 'unlabeled.review')
            x = load_lm_corpus(f, vocab, random_state=args.seed)
            size = x.size(0)
            dataset[lang][dom]['full'] = x
            print('[{}_{}] corpus size = {}'.format(lang, dom,  size))
            print('[{}_{}] corpus OOV rate = {:.2f}'.format(lang, dom, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

            # load train / test data
            train_x, train_y, train_l = load_senti_corpus(os.path.join(args.data, lang, dom, 'train.review'), vocab, maxlen=args.maxlen, random_state=args.seed)
            test_x, test_y, test_l = load_senti_corpus(os.path.join(args.data, lang, dom, 'test.review'), vocab, maxlen=args.maxlen, random_state=args.seed)
            dataset[lang][dom]['train'] = [train_x, train_y, train_l]
            dataset[lang][dom]['test'] = [test_x, test_y, test_l]
            print('[{}_{}] train size = {}'.format(lang, dom,  train_x.size(0)))
            print('[{}_{}] test size = {}'.format(lang, dom, test_x.size(0)))

        with open(args.output, 'wb')as fout:
            pickle.dump(dataset, fout)


if __name__ == '__main__':
    main()
