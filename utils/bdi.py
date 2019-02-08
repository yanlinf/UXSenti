import numpy as np


def compute_nn_accuracy(x_src, x_trg, lexicon, batch_size=5000, lexicon_size=-1):
    """
    compute word translation presicion using nn retrieval (consine similairty metric)

    x_src: np.ndarray of shape (src_vocab_size, emb_dim)
    x_trg: np.ndarray of shape (trg_vocab_size, emb_dim)
    lexicon: defaultdict
    batch_size: int
    lexicon_size: int

    returns: float
    """
    if lexicon_size < 0:
        lexicon_size = len(lexicon)

    idx_src = list(lexicon.keys())
    acc = 0.
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_trg /= np.linalg.norm(x_trg, axis=1)[:, np.newaxis] + 1e-8

    for i in range(0, len(idx_src), batch_size):
        j = min(i + batch_size, len(idx_src))
        scores = np.dot(x_trg, x_src[idx_src[i:j]].T)
        pred = scores.argmax(axis=0)
        for k in range(i, j):
            if pred[k - i] in lexicon[idx_src[k]]:
                acc += 1.

    return acc / lexicon_size
