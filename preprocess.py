import MeCab
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import argparse
import os
import torch
from utils.vocab import *
from utils.data import *


LANGS = ['en', 'fr', 'de', 'ja']
DOMS = ['books', 'dvd', 'music']
EXTRA_TOKENS = ['<EOS>', '<UNK>', '<PAD>']


ja_tagger = MeCab.Tagger("-Owakati")


def tokenize(sent, lang):
    if lang == 'ja':
        toks = ja_tagger.parse(sent).split()
    else:
        toks = word_tokenize(sent)
    toks = [t.lower() for t in toks]
    toks = ['<NUM>' if t.isdigit() else t for t in toks]
    return toks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/cls-acl10-unprocessed/', help='input directory')
    parser.add_argument('--output_dir', default='data/', help='output directory')
    parser.add_argument('--vocab_cutoff', type=int, default=15000, help='maximum vocab size')
    parser.add_argument('--maxlen', type=int, default=256, help='maximum length for each labeled example')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    output_dir = args.output_dir
    tokenized_dir = os.path.join(output_dir, 'tokenized')
    vocab_dir = os.path.join(output_dir, 'vocab')

    for d in [output_dir, tokenized_dir, vocab_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    print()
    print('Tokenizing raw text...')
    for lang in LANGS:
        f_lang = os.path.join(tokenized_dir, f'{lang}.unlabeled')
        f_lang = open(f_lang, 'w', encoding='utf-8')

        for dom in DOMS:
            f_lang_dom = os.path.join(tokenized_dir, f'{lang}.{dom}.unlabeled')
            f_lang_dom = open(f_lang_dom, 'w', encoding='utf-8')

            for part in ['train', 'test', 'unlabeled']:
                if part != 'unlabeled':
                    f_label = os.path.join(tokenized_dir, f'{lang}.{dom}.{part}')
                    f_label = open(f_label, 'w', encoding='utf-8')

                fname = os.path.join(args.input_dir, 'jp' if lang == 'ja' else lang, dom, f'{part}.review')
                root = ET.parse(fname).getroot()
                nitem, npos, nneg = 0, 0, 0
                for t in root:
                    try:
                        dic = {x.tag: x.text for x in t}
                        tokens = tokenize(str(dic['text']), lang)
                        if part != 'unlabeled':
                            if float(dic['rating']) > 3:
                                label = '__pos__'
                                npos += 1
                            else:
                                label = '__neg__'
                                nneg += 1
                            f_label.write(label + ' ' + ' '.join(tokens) + '\n')
                        if part != 'test':
                            f_lang.write(' '.join(tokens) + '\n')
                            f_lang_dom.write(' '.join(tokens) + '\n')
                        nitem += 1
                    except Exception as e:
                        print('[ERROR] ignoring item - {}'.format(e))

                if part != 'unlabeled':
                    f_label.close()

                print('file: {:60}\tvalid: {:7}\tpos: {:5}\tneg: {:5}'.format(fname, nitem, npos, nneg))

            f_lang_dom.close()
        f_lang.close()

    print()
    print('Generating vocabularies...')
    for lang in LANGS:
        with open(os.path.join(tokenized_dir, f'{lang}.unlabeled'), 'r', encoding='utf-8') as fin:
            corpus = [row.rstrip() for row in fin]
        vocab = Vocab(corpus)
        vocab.cutoff(args.vocab_cutoff - len(EXTRA_TOKENS))
        for tok in EXTRA_TOKENS:
            vocab.add_word(tok)
        vocab.save(os.path.join(vocab_dir, f'{lang}.vocab'))
        print('{} vocab of size {} generated'.format(lang, len(vocab)))

    print()
    print('Binarizing data...')
    train_set = {lang: {dom: {} for dom in DOMS} for lang in LANGS}
    test_set = {lang: {dom: {} for dom in DOMS} for lang in LANGS}
    for lang in LANGS:
        # load vocab
        vocab = Vocab(path=os.path.join(vocab_dir, f'{lang}.vocab'))
        train_set[lang]['vocab'] = test_set[lang]['vocab'] = vocab

        # load unlabeled data from a language
        x = load_lm_corpus(os.path.join(tokenized_dir, f'{lang}.unlabeled'), vocab, random_state=args.seed)
        train_set[lang]['unlabeled'] = x
        size = x.size(0)
        print('[{}]\tsize = {}'.format(lang,  size))
        print('[{}]\tOOV rate = {:.2f}'.format(lang, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

        for dom in DOMS:
            # load unlabeled data from a language-domain pair
            x = load_lm_corpus(os.path.join(tokenized_dir, f'{lang}.{dom}.unlabeled'), vocab, random_state=args.seed)
            size = x.size(0)
            train_set[lang][dom]['unlabeled'] = x
            print('[{}_{}]\tunlabeled size = {}'.format(lang, dom,  size))
            print('[{}_{}]\tOOV rate = {:.2f}'.format(lang, dom, x.bincount()[vocab.w2idx[UNK_TOK]].item() / size))

            # load train / test data
            train_x, train_y, train_l = load_senti_corpus(os.path.join(tokenized_dir,  f'{lang}.{dom}.train'),
                                                          vocab, maxlen=args.maxlen, random_state=args.seed)
            test_x, test_y, test_l = load_senti_corpus(os.path.join(tokenized_dir,  f'{lang}.{dom}.test'),
                                                       vocab, maxlen=args.maxlen, random_state=args.seed)
            train_set[lang][dom]['train'] = [train_x, train_y, train_l]
            test_set[lang][dom]['test'] = [test_x, test_y, test_l]
            print('[{}_{}]\ttrain size = {}'.format(lang, dom,  train_x.size(0)))
            print('[{}_{}]\ttest size = {}'.format(lang, dom, test_x.size(0)))

    f_train = os.path.join(output_dir, 'train.pth')
    f_test = os.path.join(output_dir, 'test.pth')
    torch.save(train_set, f_train)
    torch.save(test_set, f_test)
    print('training set saved to {}'.format(f_train))
    print('test set saved to {}'.format(f_test))


if __name__ == '__main__':
    main()
