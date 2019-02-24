import argparse
import os
import time
import math
import hashlib
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.vocab import *
from utils.data import *
from utils.utils import *
from utils.bdi import *
from utils.module import *
from model import MultiLingualMultiDomainClassifier, Discriminator

LINE_WIDTH = 138


def print_line():
    print('-' * LINE_WIDTH)


def evaluate(model, clf, ds, lid, did, batch_size):
    acc = 0
    size = 0
    for xs, ys, ls in ds:
        _, hidden, _ = model(xs, lid, did)
        hidden, _ = hidden[-1].max(1)
        pred = clf(hidden)
        _, pred = pred.max(-1)
        acc += (pred == ys).sum().item()
        size += xs.size(0)
    return acc / size


def model_save(model, clf, dis, lm_opt, dis_opt, path):
    with open(path, 'wb') as f:
        torch.save([model, clf, dis, lm_opt, dis_opt], f)


def model_load(path):
    with open(path, 'rb') as f:
        model, clf, dis, lm_opt, dis_opt = torch.load(f)
    return model, clf, dis, lm_opt, dis_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['en', 'fr', 'de', 'ja'], nargs='+', default=['en', 'fr', 'de', 'ja'], help='languages')
    parser.add_argument('-src', '--src', choices=['en', 'fr', 'de', 'ja'], default='en', help='source_language')
    parser.add_argument('-trg', '--trg', choices=['en', 'fr', 'de', 'ja'], nargs='+', default=['fr', 'de', 'ja'], help='target_language')
    parser.add_argument('--sup_dom', choices=['books', 'dvd', 'music'], default='books', help='domain to perform supervised learning')
    parser.add_argument('--dom', choices=['books', 'dvd', 'music'], nargs='+', default=['books', 'dvd', 'music'], help='domains')
    parser.add_argument('--data', default='pickle/amazon.10000.dataset', help='traning and testing data')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--val_size', type=int, default=400, help='validation set size')

    # architecture
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
    parser.add_argument('--dis_nhid', type=int, default=1024, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--dis_nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--nshare', type=int, default=1, help='number of rnn layers to share')
    parser.add_argument('--tied', type=bool_flag, nargs='?', const=True, default=True, help='tied embeddings')
    parser.add_argument('--pool', choices=['mean', 'max', 'meanmax'], default='max', help='pooling layer')
    parser.add_argument('--lambd', type=float, default=1, help='coefficient of the adversarial loss')
    parser.add_argument('--gamma', type=float, default=1, help='coefficient of the classification loss')

    # regularization
    parser.add_argument('--clf_dropout', type=float, default=0.6, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1e-6, help='weight decay applied to all weights')

    # loss function
    parser.add_argument('--criterion', choices=['nll', 'wsnll'], default='nll')
    parser.add_argument('--smooth_eps', type=float, default=0.2, help='eps for wsnll')
    parser.add_argument('--smooth_size', type=int, default=3, help='window size for wsnll')

    # optimization
    parser.add_argument('--epochs', type=int, default=8000000, help='upper epoch limit')
    parser.add_argument('-bs', '--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=20, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=50, help='classification batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--fix_bptt', action='store_true', help='fix bptt length')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--adam_beta', type=float, default=0.7, help='beta of adam')
    parser.add_argument('--dis_nsteps', type=int, help='n discriminator steps for each lm step')
    parser.add_argument('--lm_lr', type=float, default=0.003, help='initial learning rate for the language model')
    parser.add_argument('--dis_lr', type=float, default=0.0003, help='initial learning rate for the discriminators')
    parser.add_argument('--clf_lr', type=float, default=0.003, help='initial learning rate for the classifier')
    parser.add_argument('--lm_clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--dis_clip', type=float, default=0.01, help='gradient clipping')
    parser.add_argument('--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

    # device / logging settings
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--val_interval', type=int, default=1000, metavar='N', help='validation interval')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--export', type=str,  default='export/', help='dir to save the model')

    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(log_interval=20, val_interval=40)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    model_path = os.path.join(args.export, 'model.pt')
    config_path = os.path.join(args.export, 'config.json')
    export_config(args, config_path)
    check_path(model_path)

    ###############################################################################
    # Load data
    ###############################################################################

    n_trg = len(args.trg)
    lang2id = {lang: i for i, lang in enumerate(args.lang)}
    dom2id = {dom: i for i, dom in enumerate(args.dom)}
    src_id, dom_id = lang2id[args.src], dom2id[args.sup_dom]
    trg_ids = [lang2id[t] for t in args.trg]

    with open(args.data, 'rb') as fin:
        dataset = pickle.load(fin)
    vocabs = [dataset[lang]['vocab'] for lang in args.lang]

    train_x, train_y, train_l = to_device(dataset[args.src][args.sup_dom]['train'], args.cuda)
    val_ds = [sample(to_device(dataset[t][args.sup_dom]['train'], args.cuda), args.val_size) for t in args.trg]
    test_ds = [to_device(dataset[t][args.sup_dom]['test'], args.cuda) for t in args.trg]

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = [DataLoader(SentiDataset(*data), batch_size=args.test_batch_size) for data in val_ds]
    test_ds = [DataLoader(SentiDataset(*data), batch_size=args.test_batch_size) for data in test_ds]

    print('Statistics:')
    for lang, v in zip(args.lang, vocabs):
        print('\t{} vocab size: {}'.format(lang, len(v)))
    print()

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.resume:
        model, dis, lm_opt, dis_opt = model_load(args.resume)

    else:
        raise NotImplementedError

    clf = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(args.emsize, 2))
    clf_opt = optim.Adam(clf.parameters(), lr=args.clf_lr)

    cross_entropy = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda(), dis.cuda(), clf.cuda(), cross_entropy.cuda()
    else:
        model.cpu(), dis.cpu(), clf.cuda(), cross_entropy.cpu()

    print('Parameters:')
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters() if x.size())
    print('\ttotal params:   {}'.format(total_params))
    print('\tparam list:     {}'.format(len(list(model.parameters()))))
    for name, x in model.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    for name, x in dis.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    print()

    ###############################################################################
    # Training code
    ###############################################################################

    # Loop over epochs.
    best_accs = {tlang: 0. for tlang in args.trg}
    print('Traning:')
    print_line()
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        p = 0
        ptrs = np.zeros((len(args.lang), len(args.dom)), dtype=np.int64)  # shape (n_lang, n_dom)
        total_loss = np.zeros((len(args.lang), len(args.dom)))  # shape (n_lang, n_dom)
        total_clf_loss = 0
        start_time = time.time()
        model.train()
        clf.train()
        for epoch in range(args.epochs):

            # classification loss
            try:
                xs, ys, ls = next(train_iter)
            except StopIteration:
                train_iter = iter(senti_train)
                xs, ys, ls = next(train_iter)

            clf_opt.zero_grad()
            _, hidden, _ = model(xs, src_id, dom_id)
            hidden, _ = hidden[-1].max(1)
            clf_loss = cross_entropy(clf(hidden), ys)
            total_clf_loss += clf_loss.item()
            clf_loss.backward()
            clf_opt.step()

            # nn.utils.clip_grad_norm_(model.parameters(), args.lm_clip)

            if (epoch + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                total_clf_loss /= args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:4d} | lm_lr {:05.5f} | ms/batch {:7.2f} | clf_loss {:7.4f} |'.format(
                    epoch, lm_opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, total_clf_loss))
                total_clf_loss = 0
                start_time = time.time()

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                clf.eval()
                train_acc = evaluate(model, clf, train_ds, src_id, dom_id, args.test_batch_size)
                val_accs = [evaluate(model, clf, ds, tid, dom_id, args.test_batch_size) for tid, ds in zip(trg_ids, val_ds)]
                test_accs = [evaluate(model, clf, ds, tid, dom_id, args.test_batch_size) for tid, ds in zip(trg_ids, test_ds)]
                print_line()
                print(('| epoch {:4d} | train {:.4f} |' +
                       ' val' + ' {} {:.4f}' * n_trg + ' |' +
                       ' test' + ' {} {:.4f}' * n_trg + ' |').format(epoch, train_acc,
                                                                     *sum([[tlang, acc] for tlang, acc in zip(args.trg, val_accs)], []),
                                                                     *sum([[tlang, acc] for tlang, acc in zip(args.trg, test_accs)], [])))
                print_line()
                for tlang, val_acc in zip(args.trg, val_accs):
                    if val_acc > best_accs[tlang]:
                        save_path = model_path.replace('.pt', '_{}.pt'.format(tlang))
                        print('saving {} model to {}'.format(tlang, save_path))
                        model_save(model, clf, dis, lm_opt, dis_opt, save_path)
                        best_accs[tlang] = val_acc

                model.train()
                clf.train()
                start_time = time.time()

    except KeyboardInterrupt:
        print_line()
        print('Keyboard Interrupte - Exiting from training early')

    ###############################################################################
    # Testing
    ###############################################################################

    test_accs = []
    for tid, tlang, ds in zip(trg_ids, args.trg, test_ds):
        save_path = model_path.replace('.pt', '_{}.pt'.format(tlang))
        model, clf, _, _, _ = model_load(save_path)   # Load the best saved model.
        model.eval()
        test_accs.append(evaluate(model, clf, ds, tid, dom_id, args.test_batch_size))
    print_line()
    print(('|' + ' {}_test {:.4f} |' * n_trg).format(*sum([[tlang, acc] for tlang, acc in zip(args.trg, test_accs)], [])))
    print_line()

if __name__ == '__main__':
    main()
