import pickle
import os
import argparse
import numpy as np
from collections import Counter


DOMAINS = ['books', 'dvd', 'music']
LANGS = ['en', 'fr', 'de', 'jp']
BINS = list(range(0, 1050, 50)) + [np.inf]
TRG_DIR = 'pickle/'


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for lang in LANGS:
        for dom in DOMAINS:
            with open(os.path.join('data', lang,  dom, 'train.review'), 'r', encoding='utf-8') as fin:
                counts = [len(row.rstrip().split(' ')) for row in fin]
                histo, _ = np.histogram(counts, BINS)

            print('LANG = {}  DOM = {}'.format(lang, dom))
            print('\ttotal: {}'.format(len(counts)))
            print('\tmax: {}'.format(max(counts)))
            print('\tmin: {}'.format(min(counts)))
            for i, c in enumerate(histo):
                print('\t{} - {}: {}'.format(BINS[i], BINS[i + 1], c))
            print()

if __name__ == '__main__':
    main()
