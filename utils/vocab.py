class Vocab(object):

    def __init__(self, corpus=None, path=None, encoding='utf-8'):
        if corpus is not None:
            counts = {}
            for text in corpus:
                for w in text.split():
                    counts[w] = counts.get(w, 0) + 1
            self._idx2w = [t[0] for t in sorted(counts.items(), key=lambda x: -x[1])]
            self._w2idx = {w: i for i, w in enumerate(self._idx2w)}
            self._counts = counts

        elif path is not None:
            self._idx2w = []
            self._counts = {}
            with open(path, 'r', encoding=encoding) as fin:
                for line in fin:
                    w, c = line.rstrip().split(' ')
                    self._idx2w.append(w)
                    self._counts[w] = c
                self._w2idx = {w: i for i, w in enumerate(self._idx2w)}

        else:
            self._idx2w = []
            self._w2idx = {}
            self._counts = {}

    def add_word(self, w, count=1):
        if w not in self.w2idx:
            self._w2idx[w] = len(self._idx2w)
            self._idx2w.append(w)
            self._counts[w] = count
        else:
            self._counts[w] += count
        return self

    def cutoff(self, size):
        if size < len(self._idx2w):
            for w in self._idx2w[size:]:
                self._w2idx.pop(w)
                self._counts.pop(w)
            self._idx2w = self._idx2w[:size]

        assert len(self._idx2w) == len(self._w2idx)
        assert len(self._idx2w) == len(self._counts)
        return self

    def save(self, path, encoding='utf-8'):
        with open(path, 'w', encoding=encoding) as fout:
            for w in self._idx2w:
                fout.write(w + ' ' + str(self._counts[w]) + '\n')

    def __len__(self):
        return len(self._idx2w)

    def __contains__(self, word):
        return word in self._w2idx

    @property
    def w2idx(self):
        return self._w2idx

    @property
    def idx2w(self):
        return self._idx2w

    @property
    def stoi(self):
        return self._w2idx

    @property
    def itos(self):
        return self._idx2w

    @property
    def counts(self):
        return self._counts
