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
    parser.add_argument('-trg', '--trg', choices=LANG_DOM_PARIS, default='de-books', help='target pair')
    parser.add_argument('--data', default='pickle/amazon.15000.256.dataset', help='traning and testing data')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--val_size', type=int, default=600, help='validation set size')

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
    parser.add_argument('--epochs', type=int, default=80000, help='upper epoch limit')
    parser.add_argument('-bs', '--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=20, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=100, help='classification batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--fix_bptt', action='store_true', help='fix bptt length')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--adam_beta', type=float, default=0.7, help='beta of adam')
    parser.add_argument('--dis_nsteps', type=int, help='n discriminator steps for each lm step')
    parser.add_argument('--lm_lr', type=float, default=0.003, help='initial learning rate for the language model')
    parser.add_argument('--dis_lr', type=float, default=0.003, help='initial learning rate for the discriminators')
    parser.add_argument('--clf_lr', type=float, default=0.003, help='initial learning rate for the classifier')
    parser.add_argument('--lm_clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--dis_clip', type=float, default=-1, help='gradient clipping')
    parser.add_argument('--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

    # device / logging settings
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1000), help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--val_interval', type=int, default=1000, metavar='N', help='validation interval')
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

    src_lang, src_dom = args.src.split('-')
    trg_lang, trg_dom = args.trg.split('-')
    lang_dom_pairs = [[src_lang, src_dom], [src_lang, trg_dom], [trg_lang, trg_dom]]
    id_pairs = [[0, 0], [0, 1], [1, 1]]

    with open(args.data, 'rb') as fin:
        dataset = pickle.load(fin)

    src_vocab = dataset[src_lang]['vocab']
    trg_vocab = dataset[trg_lang]['vocab']
    unlabeled = to_device([batchify(dataset[lang][dom]['full'], args.batch_size) for lang, dom in lang_dom_pairs], args.cuda)

    train_x, train_y, train_l = to_device(dataset[src_lang][src_dom]['train'], args.cuda)
    val_x, val_y, val_l = sample(to_device(dataset[trg_lang][trg_dom]['train'], args.cuda), args.val_size)
    test_x, test_y, test_l = to_device(dataset[trg_lang][trg_dom]['test'], args.cuda)

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = DataLoader(SentiDataset(val_x, val_y, val_l), batch_size=args.test_batch_size)
    test_ds = DataLoader(SentiDataset(test_x, test_y, test_l), batch_size=args.test_batch_size)
    lexicon, lexsz = load_lexicon('data/muse/{}-{}.0-5000.txt'.format(src_lang, trg_lang), src_vocab, trg_vocab)

    print('Statistics:')
    print('\t{} vocab size: {}'.format(src_lang, len(src_vocab)))
    print('\t{} vocab size: {}'.format(trg_lang, len(trg_vocab)))
    print()

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.resume:
        model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt = model_load(args.resume)

    else:
        model = MultiLingualMultiDomainClassifier(n_classes=2, pool_layer=args.pool, clf_dropout=args.clf_dropout,
                                                  n_langs=2, n_doms=2, n_tok=[len(src_vocab), len(trg_vocab)],
                                                  emb_sz=args.emsize, n_hid=args.nhid, n_layers=args.nlayers,
                                                  n_share=args.nshare, tie_weights=args.tied,
                                                  output_p=args.dropout, hidden_p=args.dropouth, input_p=args.dropouti,
                                                  embed_p=args.dropoute, weight_p=args.wdrop, alpha=args.alpha, beta=args.beta)

        lang_dis_in_dim = args.emsize if args.tied else args.nhid
        lang_dis = Discriminator(lang_dis_in_dim, args.dis_nhid, 2, nlayers=args.dis_nlayers, dropout=0.1)
        dom_dis = None
        pool_layer = get_pooling_layer(args.pool)

        param_splits = [{'params': model.models.parameters(),  'lr': args.lm_lr},
                        {'params': model.clfs.parameters(), 'lr': args.clf_lr}]

        if args.optimizer == 'sgd':
            lm_opt = torch.optim.SGD(param_splits, weight_decay=args.wdecay)
            dis_opt = torch.optim.SGD(lang_dis.parameters(), lr=args.dis_lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            lm_opt = torch.optim.Adam(param_splits, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))
            dis_opt = torch.optim.Adam(lang_dis.parameters(), lr=args.dis_lr, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))

    criterion = nn.NLLLoss() if args.criterion == 'nll' else WindowSmoothedNLLLoss(args.smooth_eps)
    cross_entropy = nn.CrossEntropyLoss()

    bs = args.batch_size
    n_langs = 2
    lang_dis_y = to_device(torch.arange(n_langs).unsqueeze(-1).expand(n_langs, bs).contiguous().view(-1).contiguous(), args.cuda)

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
    best_acc = 0.
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

            # sample seq_len
            if args.fix_bptt:
                seq_len = args.bptt
            else:
                bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))

            lr0 = lm_opt.param_groups[0]['lr']
            lm_opt.param_groups[0]['lr'] = lr0 * seq_len / args.bptt

            batch_loss = 0
            lm_opt.zero_grad()
            dis_opt.zero_grad()

            lang_dis_x = []
            # lm loss
            for i, ((lid, did), lm_x) in enumerate(zip(id_pairs, unlabeled)):
                if ptrs[i] + seq_len > lm_x.size(0) - 2 - args.smooth_size:
                    ptrs[i] = 0
                    model.reset(lid=lid, did=did)
                p = ptrs[i]
                xs = lm_x[p:p + seq_len].t().contiguous()
                ys = lm_x[p + 1:p + 1 + seq_len].t().contiguous()
                if args.criterion == 'nll':
                    smooth_ids = None
                else:
                    smooth_ids = torch.stack([lm_x[p + k:p + k + seq_len].t() for k in range(1, 1 + args.smooth_size)], -1)
                    smooth_ids = smooth_ids.view(-1, args.smooth_size)
                raw_loss, loss, hid = model.single_loss(xs, ys, lid=lid, did=did, return_h=True, criterion=criterion, smooth_ids=smooth_ids)
                if args.lambd_dom == 0.and args.lambd_lang == 0:
                    (args.lambd_lm * loss).backward()
                else:
                    batch_loss = batch_loss + loss * args.lambd_lm
                    if did == 0:
                        lang_dis_x.append(pool_layer(hid[-1]))
                    elif lid == 1 and did == 1:
                        _, _, hid = model.single_loss(xs, ys, 1, 0, return_h=True)
                        lang_dis_x.append(pool_layer(hid[-1]))

                total_loss[i] += raw_loss.item()
                ptrs[i] += seq_len

            if args.lambd_dom != 0 or args.lambd_lang != 0:
                xtmp = GradReverse.apply(torch.cat(lang_dis_x, 0))
                lang_dis_loss = cross_entropy(lang_dis(xtmp), lang_dis_y)
                batch_loss = batch_loss + args.lambd_lang * lang_dis_loss
                total_lang_dis_loss += lang_dis_loss.item()

                batch_loss.backward()

            # classification loss
            try:
                xs, ys, ls = next(train_iter)
            except StopIteration:
                train_iter = iter(senti_train)
                xs, ys, ls = next(train_iter)
            clf_loss = cross_entropy(model(xs, ls, 0, 0), ys)
            total_clf_loss += clf_loss.item()
            (args.gamma * clf_loss).backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.lm_clip)
            if args.dis_clip > 0:
                for x in list(lang_dis.parameters()) + list(dom_dis.parameters()):
                    x.data.clamp_(-args.dis_clip, args.dis_clip)
            dis_opt.step()
            lm_opt.step()
            lm_opt.param_groups[0]['lr'] = lr0

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
                    train_acc = evaluate(model, train_ds, 0, 0, args.test_batch_size)
                    val_acc = evaluate(model, val_ds, 1, 0, args.test_batch_size)
                    test_acc = evaluate(model, test_ds, 1, 0, args.test_batch_size)
                    bdi_acc = compute_nn_accuracy(model.encoder_weight(0).cpu().numpy(),
                                                  model.encoder_weight(1).cpu().numpy(),
                                                  lexicon, 10000, lexicon_size=lexsz)
                    print_line()
                    print('| epoch {:4d} | train {:.4f} |  val {:.4f} | test {:.4f} | bdi {:.4f} |'.format(epoch, train_acc, val_acc, test_acc, bdi_acc))
                    print_line()
                    if val_acc > best_acc:
                        print('saving model to {}'.format(model_path))
                        model_save(model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt, model_path)
                        best_acc = val_acc

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

if __name__ == '__main__':
    main()
