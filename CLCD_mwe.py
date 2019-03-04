import argparse
import os
import time
import math
import hashlib
import numpy as np
import json
import pickle
import torch
import torch.nn as nn
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
from model import MultiLingualMultiDomainClassifier, Discriminator, get_pooling_layer

LINE_WIDTH = 140

LANGS = ['en', 'fr', 'de', 'ja']
DOMS = ['books', 'dvd', 'music']
LANG_DOM_PARIS = [lang + '-' + dom for lang in LANGS for dom in DOMS]
DEFAULT_TRG_PAIRS = [lang + '-' + dom for lang in ['fr', 'de'] for dom in ['dvd', 'music']]


def print_line():
    print('-' * LINE_WIDTH)


def evaluate(model, ds, lid, did, batch_size):
    acc = 0
    size = 0
    for xs, ys, ls in ds:
        pred = model(xs, ls, lid, did)
        _, pred = pred.max(-1)
        acc += (pred == ys).sum().item()
        size += xs.size(0)
    return acc / size


def model_save(model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt, path):
    with open(path, 'wb') as f:
        torch.save([model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt], f)


def model_load(path):
    with open(path, 'rb') as f:
        model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt = torch.load(f)
    return model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt


def load_config(model_path, args):
    with open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r') as fin:
        dic = json.load(fin)

    for k in dict(vars(args)):
        if k != 'resume':
            setattr(args, k, dic[k])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--src', choices=LANG_DOM_PARIS, default='en-dvd', help='source pair')
    parser.add_argument('-trg', '--trg', choices=LANG_DOM_PARIS, nargs='+', default=DEFAULT_TRG_PAIRS, help='target pair')
    parser.add_argument('--data', default='pickle/amazon.15000.256.dataset', help='traning and testing data')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--val_size', type=int, default=600, help='validation set size')
    parser.add_argument('--embedding', default='data/vectors-{}.txt', help='multi-lingual word embeddings')

    # architecture
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
    parser.add_argument('--dis_nhid', type=int, default=400, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--dis_nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--nshare', type=int, default=1, help='number of rnn layers to share')
    parser.add_argument('--tied', type=bool_flag, nargs='?', const=True, default=True, help='tied embeddings')
    parser.add_argument('--pool', choices=['mean', 'max', 'meanmax'], default='mean', help='pooling layer')
    parser.add_argument('--lambd_lang', type=float, default=0.1, help='coefficient of the adversarial loss')
    parser.add_argument('--lambd_dom', type=float, default=0, help='coefficient of the adversarial loss')
    parser.add_argument('--lambd_lm', type=float, default=1, help='coefficient of the language modeling')
    parser.add_argument('--gamma', type=float, default=0.01, help='coefficient of the classification loss')

    # regularization
    parser.add_argument('--clf_dropout', type=float, default=0.6, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.4, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')

    # loss function
    parser.add_argument('--criterion', choices=['nll', 'wsnll'], default='nll')
    parser.add_argument('--smooth_eps', type=float, default=0.2, help='eps for wsnll')
    parser.add_argument('--smooth_size', type=int, default=3, help='window size for wsnll')

    # optimization
    parser.add_argument('--epochs', type=int, default=20000, help='upper epoch limit')
    parser.add_argument('-bs', '--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=20, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=100, help='classification batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--fix_bptt', action='store_true', help='fix bptt length')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--adam_beta', type=float, default=0.7, help='beta of adam')
    parser.add_argument('--dis_nsteps', type=int, help='n discriminator steps for each lm step')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='initial learning rate for the language model')
    parser.add_argument('--lm_clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--dis_clip', type=float, default=-1, help='gradient clipping')
    parser.add_argument('--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

    # device / logging settings
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1000), help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N', help='report interval')
    parser.add_argument('--val_interval', type=int, default=300, metavar='N', help='validation interval')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--export', type=str,  default='export/', help='dir to save the model')

    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(log_interval=20, val_interval=40)
    args = parser.parse_args()

    if args.resume:
        args = load_config(args.resume, args)

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

    langs = list(set([t.split('-')[0] for t in [args.src] + args.trg]))
    lang2id = {l: i for i, l in enumerate(langs)}

    src_lang, src_dom = args.src.split('-')
    trg_pairs = [t.split('-') for t in args.trg]
    trg_lang_ids = [lang2id[l] for l, _ in trg_pairs]

    with open(args.data, 'rb') as fin:
        dataset = pickle.load(fin)

    vocabs = [dataset[l]['vocab'] for l in langs]

    train_x, train_y, train_l = to_device(dataset[src_lang][src_dom]['train'], args.cuda)
    val_ds = [sample(to_device(dataset[tl][td]['train'], args.cuda), args.val_size) for tl, td in trg_pairs]
    test_ds = [to_device(dataset[tl][td]['test'], args.cuda) for tl, td in trg_pairs]

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = [DataLoader(SentiDataset(*data), batch_size=args.test_batch_size) for data in val_ds]
    test_ds = [DataLoader(SentiDataset(*data), batch_size=args.test_batch_size) for data in test_ds]

    oov_rates = []
    mwe = []
    for v, lang in zip(vocabs, langs):
        x, count = load_vectors_with_vocab(args.embedding.format(lang), v, 50000 if args.debug else -1)
        mwe.append(x)
        oov_rates.append(1 - count / len(v))

    print('Statistics:')
    for lang, v in zip(langs, vocabs):
        print('\t{} vocab size: {}'.format(lang, len(v)))
    for lang, oov_rate in zip(langs, oov_rates):
        print('\t{} oov rate:   {:.4f}'.format(lang, oov_rate))
    print()

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.resume:
        model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt = model_load(args.resume)

    else:
        model = MultiLingualMultiDomainClassifier(n_classes=2, pool_layer=args.pool, clf_dropout=args.clf_dropout,
                                                  n_langs=len(langs), n_doms=1, n_tok=[len(v) for v in vocabs],
                                                  emb_sz=args.emsize, n_hid=args.nhid, n_layers=args.nlayers,
                                                  n_share=args.nshare, tie_weights=args.tied,
                                                  output_p=args.dropout, hidden_p=args.dropouth, input_p=args.dropouti,
                                                  embed_p=args.dropoute, weight_p=args.wdrop, alpha=args.alpha, beta=args.beta)

        for lid in range(len(langs)):
            model.encoder_weight(lid).copy_(torch.from_numpy(mwe[lid]))
            freeze_net(model.encoders[lid])

        lang_dis_in_dim = args.emsize if args.tied else args.nhid
        lang_dis = Discriminator(lang_dis_in_dim, args.dis_nhid, 2, nlayers=args.dis_nlayers, dropout=0.1)
        dom_dis = None
        pool_layer = get_pooling_layer(args.pool)

        if args.optimizer == 'sgd':
            lm_opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            dis_opt = None
        if args.optimizer == 'adam':
            lm_opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))
            dis_opt = None

    criterion = nn.NLLLoss() if args.criterion == 'nll' else WindowSmoothedNLLLoss(args.smooth_eps)
    cross_entropy = nn.CrossEntropyLoss()

    bs = args.batch_size
    n_langs = 2

    if args.cuda:
        model.cuda(), lang_dis.cuda(), criterion.cuda(), cross_entropy.cuda()
    else:
        model.cpu(), lang_dis.cpu(), criterion.cpu(), cross_entropy.cpu()

    print('Parameters:')
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters() if x.size())
    print('\ttotal params:   {}'.format(total_params))
    print('\tparam list:     {}'.format(len(list(model.parameters()))))
    for name, x in model.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    for name, x in lang_dis.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    print()

    ###############################################################################
    # Training code
    ###############################################################################

    # Loop over epochs.
    best_acc = np.zeros(len(args.trg))
    print('Traning:')
    print_line()
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        p = 0
        ptrs = np.zeros(3, dtype=np.int64)
        total_loss = np.zeros(3)  # shape (n_lang, n_dom)
        total_clf_loss = 0
        total_lang_dis_loss = 0
        start_time = time.time()
        model.train()
        model.reset_all()
        for epoch in range(args.epochs):

            batch_loss = 0
            lm_opt.zero_grad()

            # classification loss
            try:
                xs, ys, ls = next(train_iter)
            except StopIteration:
                train_iter = iter(senti_train)
                xs, ys, ls = next(train_iter)
            clf_loss = cross_entropy(model(xs, ls, lang2id[src_lang], 0), ys)
            clf_loss.backward()
            total_clf_loss += clf_loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), args.lm_clip)

            lm_opt.step()

            if (epoch + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                total_clf_loss /= args.log_interval
                total_lang_dis_loss /= args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:4d} | lm_lr {:05.5f} | ms/batch {:7.2f} | lm_loss {:7.4f} | avg_ppl {:7.2f} | clf {:7.4f} | lang {:7.4f} |'.format(
                    epoch, lm_opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                    total_loss.mean(), np.exp(total_loss).mean(), total_clf_loss, total_lang_dis_loss))
                total_loss[:] = 0
                total_clf_loss = 0
                total_lang_dis_loss = 0
                start_time = time.time()

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    train_acc = evaluate(model, train_ds, lang2id[src_lang], 0, args.test_batch_size)
                    val_acc = [evaluate(model, ds, tid, 0, args.test_batch_size) for tid, ds in zip(trg_lang_ids, val_ds)]
                    test_acc = [evaluate(model, ds, tid, 0, args.test_batch_size) for tid, ds in zip(trg_lang_ids, test_ds)]
                    print_line()
                    print(('| epoch {:4d} | train {:.4f} |' +
                           ' val' + ' {} {:.4f}' * len(args.trg) + ' |' +
                           ' test' + ' {} {:.4f}' * len(args.trg) + ' |').format(epoch, train_acc,
                                                                                 *sum([[tpair, acc] for tpair, acc in zip(args.trg, val_acc)], []),
                                                                                 *sum([[tpair, acc] for tpair, acc in zip(args.trg, test_acc)], [])))
                    print_line()
                    for j, (acc, tpair) in enumerate(zip(val_acc, args.trg)):
                        if acc > best_acc[j]:
                            save_path = model_path.replace('.pt', '_[{}].pt'.format(tpair))
                            print('saving {} model to {}'.format(tpair, save_path))
                            model_save(model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt, save_path)
                            best_acc[j] = acc

                model.train()
                start_time = time.time()

    except KeyboardInterrupt:
        print_line()
        print('Keyboard Interrupte - Exiting from training early')

    ###############################################################################
    # Testing
    ###############################################################################

    with torch.no_grad():
        model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt = model_load(model_path)   # Load the best saved model.
        model.eval()
        test_acc = evaluate(model, test_ds, 1, 0, args.test_batch_size)
    print_line()
    print('| [{}|{}]_test {:.4f} |'.format(args.src, args.trg, test_acc))
    print_line()

    test_accs = []
    for tid, tpair, ds in zip(trg_lang_ids, args.trg, test_ds):
        save_path = model_path.replace('.pt', '_[{}].pt'.format(tpair))
        model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt = model_load(save_path)   # Load the best saved model.
        model.eval()
        test_accs.append(evaluate(model, ds, tid, 0, args.test_batch_size))
    print_line()
    print(('| src {} |' + ' [{}]_test {:.4f} |' * len(args.trg)).format(args.src, *sum([[tpair, acc] for tpair, acc in zip(args.trg, test_accs)], [])))
    print_line()

if __name__ == '__main__':
    main()
