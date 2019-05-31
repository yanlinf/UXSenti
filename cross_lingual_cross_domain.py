import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
import json
import pickle
from model import XLXDClassifier, Discriminator
from utils.vocab import *
from utils.data import *
from utils.utils import *
from utils.bdi import *
from utils.layers import *


LANGS = ['en', 'fr', 'de', 'ja']
DOMS = ['books', 'dvd', 'music']
VALID_PAIRS = [lang + '-' + dom for lang in LANGS for dom in DOMS]


def print_line():
    print('-' * 110)


def evaluate(model, ds, lid, did):
    acc = 0
    size = 0
    for xs, ys, ls in ds:
        pred = model(xs, ls, lid, did)
        _, pred = pred.max(-1)
        acc += (pred == ys).sum().item()
        size += xs.size(0)
    return acc / size


def model_save(model, dis, lm_opt, dis_opt, path):
    torch.save([model, dis, lm_opt, dis_opt], path)


def model_load(path):
    model, dis, lm_opt, dis_opt = torch.load(path)
    return model, dis, lm_opt, dis_opt


def load_config(config_dir, args):
    with open(os.path.join(config_dir, 'config.json'), 'r') as fin:
        dic = json.load(fin)

    for k in dict(vars(args)):
        if k not in ('resume', 'mode', 'early_stopping', 'cuda', 'test'):
            setattr(args, k, dic[k])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--src', choices=VALID_PAIRS, default='en-dvd', help='source pair')
    parser.add_argument('-trg', '--trg', choices=VALID_PAIRS, default='de-books', help='target pair')
    parser.add_argument('--mwe', action='store_true', help='run the MWE model variant')
    parser.add_argument('--mwe_path', default='data/vectors/vectors-{}.txt', help='path to multilingual word embeddings')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--early_stopping',  type=bool_flag, nargs='?', const=True, default=False, help='perform early stopping')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train', help='train or evaluate')

    # datasets
    parser.add_argument('--unlabeled', default='data/unlabeled.pth', help='binarized unlabeled data')
    parser.add_argument('--train', default='data/train.pth', help='binarized training data')
    parser.add_argument('--val', default='data/val.pth', help='binarized validation data')
    parser.add_argument('--test', default='data/test.pth', help='binarized test data')
    parser.add_argument('--sample_train', type=int, default=0, help='downsample training set to n examples (zero to disable)')
    parser.add_argument('--sample_unlabeled', type=int, default=0, help='downsample unlabeled_set to n tokens (zero to disable)')

    # architecture
    parser.add_argument('--emb_dim', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--hid_dim', type=int, default=1150, help='number of hidden units per layer of the language model')
    parser.add_argument('--dis_hid_dim', type=int, default=400, help='number of hidden units per layer of the discriminator')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--nshare', type=int, default=1, help='number of rnn layers to share')
    parser.add_argument('--dis_nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--tie_softmax', type=bool_flag, nargs='?', const=True, default=True, help='tied embeddings')
    parser.add_argument('--lambd_lm', type=float, default=1, help='coefficient of the language modeling')
    parser.add_argument('--lambd_dis', type=float, default=0.1, help='coefficient of the adversarial loss')
    parser.add_argument('--lambd_clf', type=float, default=0.003, help='coefficient of the classification loss')

    # regularization
    parser.add_argument('--dropoutc', type=float, default=0.6, help='dropout applied to classifier')
    parser.add_argument('--dropouto', type=float, default=0.4, help='dropout applied to rnn ouputs')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.4, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--dropoutw', type=float, default=0.5, help='weight dropout applied to the RNN hidden to hidden matrix')
    parser.add_argument('--dropoutd', type=float, default=0.1, help='dropout applied to language discriminator')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')

    # optimization
    parser.add_argument('--max_steps', type=int, default=30000, help='upper step limit')
    parser.add_argument('-bs', '--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=20, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=100, help='classification batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--beta1', type=float, default=0.7, help='beta1 for adam optimizer')
    parser.add_argument('-lr', '--lr', type=float, default=0.003, help='initial learning rate for the language model')
    parser.add_argument('--grad_clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--dis_clip', type=float, default=0, help='clipping discriminator weights')

    # device / logging settings
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--val_interval', type=int, default=1000, metavar='N', help='validation interval')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--export', type=str,  default='export/clcd/', help='dir to save the model')

    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(log_interval=20, val_interval=40)
    if args.mwe:
        parser.set_defaults(lr=0.001, lambd_clf=1.)
    args = parser.parse_args()
    if args.mode == 'eval':
        args = load_config(args.export, args)
    elif args.resume:
        args = load_config(os.path.dirname(args.resume), args)

    if args.mode == 'train':
        train(args)
    else:
        eval(args)


def train(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
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

    unlabeled_set = torch.load(args.unlabeled)
    train_set = torch.load(args.train)
    val_set = torch.load(args.val)
    test_set = torch.load(args.test)

    src_vocab = train_set[src_lang]['vocab']
    trg_vocab = train_set[trg_lang]['vocab']
    unlabeled = to_device([batchify(unlabeled_set[lang][dom], args.batch_size) for lang, dom in lang_dom_pairs], args.cuda)
    train_x, train_y, train_l = to_device(train_set[src_lang][src_dom], args.cuda)
    val_x, val_y, val_l = to_device(val_set[trg_lang][trg_dom], args.cuda)
    test_x, test_y, test_l = to_device(test_set[trg_lang][trg_dom], args.cuda)

    if args.sample_unlabeled > 0:
        print('Downsampling unlabeled set...')
        print()
        unlabeled = [x[:(args.sample_unlabeled // args.batch_size)] for x in unlabeled]
    if args.sample_train > 0:
        print('Downsampling training set...')
        print()
        train_x, train_y, train_l = sample([train_x, train_y, train_l], args.sample_train, True)

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = DataLoader(SentiDataset(val_x, val_y, val_l), batch_size=args.test_batch_size)
    test_ds = DataLoader(SentiDataset(test_x, test_y, test_l), batch_size=args.test_batch_size)
    lexicon, lexsz = load_lexicon('data/muse/{}-{}.0-5000.txt'.format(src_lang, trg_lang), src_vocab, trg_vocab)

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.resume:
        model, dis, lm_opt, dis_opt = model_load(args.resume)

    else:
        model = XLXDClassifier(n_classes=2, clf_p=args.dropoutc, n_langs=2, n_doms=2,
                               vocab_sizes=[len(src_vocab), len(trg_vocab)], emb_size=args.emb_dim, hidden_size=args.hid_dim,
                               num_layers=args.nlayers, num_share=args.nshare, tie_weights=args.tie_softmax,
                               output_p=args.dropouto, hidden_p=args.dropouth, input_p=args.dropouti, embed_p=args.dropoute, weight_p=args.dropoutw)

        dis = Discriminator(args.emb_dim, args.dis_hid_dim, 2, args.dis_nlayers, args.dropoutd)

        if args.mwe:
            x, count = load_vectors_with_vocab(args.mwe_path.format(src_lang), src_vocab, -1)
            model.encoders[0].weight.data.copy_(torch.from_numpy(x))
            x, count = load_vectors_with_vocab(args.mwe_path.format(trg_lang), trg_vocab, -1)
            model.encoders[1].weight.data.copy_(torch.from_numpy(x))
            freeze_net(model.encoders)

        if args.optimizer == 'sgd':
            lm_opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            dis_opt = torch.optim.SGD(dis.parameters(), lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            lm_opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.beta1, 0.999))
            dis_opt = torch.optim.Adam(dis.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.beta1, 0.999))

    crit = nn.CrossEntropyLoss()
    bs = args.batch_size
    dis_y = to_device(torch.tensor([0] * bs + [1] * bs), args.cuda)

    if args.cuda:
        model.cuda(), dis.cuda(), crit.cuda()
    else:
        model.cpu(), dis.cpu(), crit.cpu()

    print('Parameters:')
    total_params = sum([np.prod(x.size()) for x in model.parameters()])
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

    bptt = args.bptt
    best_acc = 0.
    print('Traning:')
    print_line()
    p = 0
    ptrs = np.zeros(3, dtype=np.int64)
    total_loss = np.zeros(3)  # shape (n_lang, n_dom)
    total_clf_loss = 0
    total_dis_loss = 0
    start_time = time.time()
    model.train()
    model.reset()
    for step in range(args.max_steps):
        loss = 0
        lm_opt.zero_grad()
        dis_opt.zero_grad()

        if not args.mwe:
            # language modeling loss
            dis_x = []
            for i, ((lid, did), lm_x) in enumerate(zip(id_pairs, unlabeled)):
                if ptrs[i] + bptt + 1 > lm_x.size(0):
                    ptrs[i] = 0
                    model.reset(lid=lid, did=did)
                p = ptrs[i]
                xs = lm_x[p:p + bptt].t().contiguous()
                ys = lm_x[p + 1:p + 1 + bptt].t().contiguous()
                lm_loss, hid = model.lm_loss(xs, ys, lid=lid, did=did, return_h=True)
                loss = loss + lm_loss * args.lambd_lm
                if lid == 0 and did == 0:
                    dis_x.append(hid[-1].mean(1))
                elif lid == 1 and did == 1:
                    _, hid = model.lm_loss(xs, ys, lid=1, did=0, return_h=True)
                    dis_x.append(hid[-1].mean(1))
                total_loss[i] += lm_loss.item()
                ptrs[i] += bptt

            # language adversarial loss
            dis_x_rev = GradReverse.apply(torch.cat(dis_x, 0))
            dis_loss = crit(dis(dis_x_rev), dis_y)
            loss = loss + args.lambd_dis * dis_loss
            total_dis_loss += dis_loss.item()
            loss.backward()

        # sentiment classification loss
        try:
            xs, ys, ls = next(train_iter)
        except StopIteration:
            train_iter = iter(senti_train)
            xs, ys, ls = next(train_iter)
        clf_loss = crit(model(xs, ls, lid=0, did=0), ys)
        total_clf_loss += clf_loss.item()
        (args.lambd_clf * clf_loss).backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.dis_clip > 0:
            for x in dis.parameters():
                x.data.clamp_(-args.dis_clip, args.dis_clip)
        lm_opt.step()
        dis_opt.step()

        if (step + 1) % args.log_interval == 0:
            total_loss /= args.log_interval
            total_clf_loss /= args.log_interval
            total_dis_loss /= args.log_interval
            elapsed = time.time() - start_time
            print('| step {:5d} | lr {:05.5f} | ms/batch {:7.2f} | lm_loss {:7.4f} | avg_ppl {:7.2f} | clf {:7.4f} | dis {:7.4f} |'.format(
                step, lm_opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                total_loss.mean(), np.exp(total_loss).mean(), total_clf_loss, total_dis_loss))
            total_loss[:], total_clf_loss, total_dis_loss = 0, 0, 0
            start_time = time.time()

        if (step + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                train_acc = evaluate(model, train_ds, 0, 0)
                val_acc = evaluate(model, val_ds, 1, 0)
                test_acc = evaluate(model, test_ds, 1, 0)
                bdi_acc = compute_nn_accuracy(model.encoder_weight(0),
                                              model.encoder_weight(1),
                                              lexicon, 10000, lexicon_size=lexsz)
                print_line()
                print('| step {:5d} | train {:.4f} |  val {:.4f} | test {:.4f} | bdi {:.4f} |'.format(step, train_acc, val_acc, test_acc, bdi_acc))
                print_line()
                print('saving model to {}'.format(model_path.replace('.pt', '_final.pt')))
                model_save(model, dis, lm_opt, dis_opt, model_path.replace('.pt', '_final.pt'))
                if val_acc > best_acc:
                    print('saving model to {}'.format(model_path))
                    model_save(model, dis, lm_opt, dis_opt, model_path)
                    best_acc = val_acc
                print_line()
            model.train()
            start_time = time.time()

    ###############################################################################
    # Testing
    ###############################################################################


def eval(args):
    test_set = torch.load(args.test)
    trg_lang, trg_dom = args.trg.split('-')
    test_x, test_y, test_l = to_device(test_set[trg_lang][trg_dom], args.cuda)
    test_ds = DataLoader(SentiDataset(test_x, test_y, test_l), batch_size=args.test_batch_size)
    with torch.no_grad():
        if args.early_stopping:
            model_path = os.path.join(args.export, 'model.pt')
        else:
            model_path = os.path.join(args.export, 'model_final.pt')
        model, dis, lm_opt, dis_opt = model_load(model_path)   # Load the best saved model.
        model.cuda() if args.cuda else model.cpu()
        model.eval()
        test_acc = evaluate(model, test_ds, 1, 0)
    print_line()
    print('| [{}|{}]_test {:.4f} |'.format(args.src, args.trg, test_acc))
    print_line()


if __name__ == '__main__':
    main()
