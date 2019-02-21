import argparse
import os
import time
import math
import hashlib
import numpy as np
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
from model import MultiLingualMultiDomainClassifier, Discriminator

LINE_WIDTH = 102


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


def model_save(model, dis, lm_opt, dis_opt, path):
    with open(path, 'wb') as f:
        torch.save([model, dis, lm_opt, dis_opt], f)


def model_load(path):
    with open(path, 'rb') as f:
        model, dis, lm_opt, dis_opt = torch.load(f)
    return model, dis, lm_opt, dis_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['en', 'fr', 'de', 'ja'], nargs='+', default=['en', 'fr', 'de', 'ja'], help='languages')
    parser.add_argument('-src', '--src', choices=['en', 'fr', 'de', 'ja'], default='en', help='source_language')
    parser.add_argument('-trg', '--trg', choices=['en', 'fr', 'de', 'ja'], default='fr', help='target_language')
    parser.add_argument('--sup_dom', choices=['books', 'dvd', 'music'], default='books', help='domain to perform supervised learning')
    parser.add_argument('--dom', choices=['books', 'dvd', 'music'], nargs='+', default=['books', 'dvd', 'music'], help='domains')
    parser.add_argument('--data', default='pickle/amazon.10000.dataset', help='traning and testing data')
    parser.add_argument('--resume', help='path of model to resume')

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
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=10, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=50, help='classification batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--adam_beta', type=float, default=0.7, help='beta of adam')
    parser.add_argument('--dis_nsteps', type=int, help='n discriminator steps for each lm step')
    parser.add_argument('--lm_lr', type=float, default=0.003, help='initial learning rate')
    parser.add_argument('--dis_lr', type=float, default=0.0003, help='initial learning rate')
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

    lang2id = {lang: i for i, lang in enumerate(args.lang)}
    dom2id = {dom: i for i, dom in enumerate(args.dom)}
    src_lang_id, trg_lang_id, dom_id = lang2id[args.src], lang2id[args.trg], dom2id[args.sup_dom]

    with open(args.data, 'rb') as fin:
        dataset = pickle.load(fin)
    vocabs = [dataset[lang]['vocab'] for lang in args.lang]
    unlabeled = to_device([[batchify(dataset[lang][dom]['full'], args.batch_size) for dom in args.dom] for lang in args.lang], args.cuda)

    train_x, train_y, train_l = to_device(dataset[args.src][args.sup_dom]['train'], args.cuda)
    val_x, val_y, val_l = to_device(dataset[args.src][args.sup_dom]['test'], args.cuda)
    test_x, test_y, test_l = to_device(dataset[args.trg][args.sup_dom]['test'], args.cuda)

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = DataLoader(SentiDataset(val_x, val_y, val_l), batch_size=args.test_batch_size)
    test_ds = DataLoader(SentiDataset(test_x, test_y, test_l), batch_size=args.test_batch_size)
    lexicon, lexsz = load_lexicon('data/muse/{}-{}.0-5000.txt'.format(args.src, args.trg), vocabs[src_lang_id], vocabs[trg_lang_id])
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
        model = MultiLingualMultiDomainClassifier(n_classes=2, pool_layer=args.pool,
                                                  n_langs=len(args.lang), n_doms=len(args.dom), n_tok=list(map(len, vocabs)),
                                                  emb_sz=args.emsize, n_hid=args.nhid, n_layers=args.nlayers,
                                                  n_share=args.nshare, tie_weights=args.tied,
                                                  output_p=args.dropout, hidden_p=args.dropouth, input_p=args.dropouti,
                                                  embed_p=args.dropoute, weight_p=args.wdrop, alpha=args.alpha, beta=args.beta)

        dis_in_dim = args.emsize if args.tied and args.nshare == args.nlayers else args.nhid
        dis_out_dim = len(args.dom)
        dis = Discriminator(dis_in_dim, args.dis_nhid, dis_out_dim, nlayers=args.dis_nlayers, dropout=0.1)

        if args.optimizer == 'sgd':
            lm_opt = torch.optim.SGD(model.parameters(), lr=args.lm_lr, weight_decay=args.wdecay)
            dis_opt = torch.optim.SGD(dis.parameters(), lr=args.dis_lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            lm_opt = torch.optim.Adam(model.parameters(), lr=args.lm_lr, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))
            dis_opt = torch.optim.Adam(dis.parameters(), lr=args.dis_lr, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))

    criterion = nn.NLLLoss() if args.criterion == 'nll' else WindowSmoothedNLLLoss(args.smooth_eps)
    cross_entropy = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda(), dis.cuda(), criterion.cuda(), cross_entropy.cuda()
    else:
        model.cpu(), dis.cpu(), criterion.cpu(), cross_entropy.cpu()

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
    best_acc = 0.
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
        model.reset_all()
        for epoch in range(args.epochs):

            # sample seq_len
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            lr0 = lm_opt.param_groups[0]['lr']
            lm_opt.param_groups[0]['lr'] = lr0 * seq_len / args.bptt

            batch_loss = 0
            lm_opt.zero_grad()

            # lm loss
            for lid, t in enumerate(unlabeled):
                for did, lm_x in enumerate(t):
                    if ptrs[lid, did] + seq_len > lm_x.size(0) - 2 - args.smooth_size:
                        ptrs[lid, did] = 0
                        model.reset(lid=lid, did=did)
                    p = ptrs[lid, did]
                    xs = lm_x[p:p + seq_len].t().contiguous()
                    ys = lm_x[p + 1:p + 1 + seq_len].t().contiguous()
                    # if args.criterion == 'nll':
                    #     src_smooth_idx = trg_smooth_idx = None
                    # else:
                    #     src_smooth_idx = torch.stack([src_train[src_p + k:src_p + k + seq_len].t() for k in range(1, 1 + args.smooth_size)], -1)
                    #     src_smooth_idx = src_smooth_idx.view(-1, args.smooth_size)
                    #     trg_smooth_idx = torch.stack([trg_train[trg_p + k:trg_p + k + seq_len].t() for k in range(1, 1 + args.smooth_size)], -1)
                    #     trg_smooth_idx = trg_smooth_idx.view(-1, args.smooth_size)
                    raw_loss, loss = model.single_loss(xs, ys, lid=lid, did=did)
                    loss.backward()
                    batch_loss = batch_loss + loss
                    total_loss[lid, did] += raw_loss.item()
                    ptrs[lid, did] += seq_len
            # batch_loss.backward()

            # classification loss
            try:
                xs, ys, ls = next(train_iter)
            except StopIteration:
                train_iter = iter(senti_train)
                xs, ys, ls = next(train_iter)
            clf_loss = cross_entropy(model(xs, ls, src_lang_id, dom_id), ys)
            total_clf_loss += clf_loss.item()
            (args.gamma * clf_loss).backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.lm_clip)
            for x in dis.parameters():
                x.data.clamp_(-args.dis_clip, args.dis_clip)
            lm_opt.step()
            lm_opt.param_groups[0]['lr'] = lr0

            if (epoch + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                total_clf_loss /= args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:4d} | lm_lr {:05.5f} | ms/batch {:7.2f} | lm_loss {:5.2f} | avg_ppl {:7.2f} | clf_loss {:7.4f} |'.format(
                    epoch, lm_opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                    total_loss.mean(), np.exp(total_loss).mean(), total_clf_loss))
                total_loss[:, :] = 0
                total_clf_loss = 0
                start_time = time.time()

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                train_acc = evaluate(model, train_ds, src_lang_id, dom_id, args.test_batch_size)
                val_acc = evaluate(model, val_ds, src_lang_id, dom_id, args.test_batch_size)
                test_acc = evaluate(model, test_ds, trg_lang_id, dom_id, args.test_batch_size)
                bdi_acc = compute_nn_accuracy(model.encoder_weight(src_lang_id).cpu().numpy(),
                                              model.encoder_weight(trg_lang_id).cpu().numpy(),
                                              lexicon, lexicon_size=lexsz)
                print_line()
                print('| epoch {:4d} | train {:.4f} | val {:.4f} | test {:.4f} | bdi {:.4f} |'.format(epoch, train_acc, val_acc, test_acc, bdi_acc))
                print_line()
                if test_acc > best_acc:
                    print('saving model to {}'.format(model_path))
                    model_save(model, dis, lm_opt, dis_opt, model_path)
                    best_acc = test_acc
                model.train()
                start_time = time.time()

    except KeyboardInterrupt:
        print_line()
        print('Keyboard Interrupte - Exiting from training early')

    ###############################################################################
    # Testing
    ###############################################################################

    trainer = model_load(model_path)   # Load the best saved model.


if __name__ == '__main__':
    main()
