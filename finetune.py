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
    parser.add_argument('-src', '--src', choices=['en', 'fr', 'de', 'ja'], default='en', help='source_language')
    parser.add_argument('-trg', '--trg', choices=['en', 'fr', 'de', 'ja'], nargs='+', default=['fr', 'de', 'ja'], help='target_language')
    parser.add_argument('--sup_dom', choices=['books', 'dvd', 'music'], default='books', help='domain to perform supervised learning')
    parser.add_argument('--data', default='pickle/amazon.15000.256.dataset', help='traning and testing data')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--val_size', type=int, default=600, help='validation set size')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')

    # optimization
    parser.add_argument('--epochs', type=int, default=50000, help='upper epoch limit')
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=20, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=100, help='classification batch size')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--adam_beta', type=float, default=0.7, help='beta of adam')
    parser.add_argument('-lr', '--lr', type=float, default=0.003, help='initial learning rate for the language model')

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
    lang2id = {lang: i for i, lang in enumerate(LANGS)}
    dom2id = {dom: i for i, dom in enumerate(DOMS)}
    src_id, dom_id = lang2id[args.src], dom2id[args.sup_dom]
    trg_ids = [lang2id[t] for t in args.trg]

    with open(args.data, 'rb') as fin:
        dataset = pickle.load(fin)
    vocabs = [dataset[lang]['vocab'] for lang in LANGS]

    train_x, train_y, train_l = to_device(dataset[args.src][args.sup_dom]['train'], args.cuda)
    val_ds = [sample(to_device(dataset[t][args.sup_dom]['train'], args.cuda), args.val_size) for t in args.trg]
    test_ds = [to_device(dataset[t][args.sup_dom]['test'], args.cuda) for t in args.trg]

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = [DataLoader(SentiDataset(*data), batch_size=args.test_batch_size) for data in val_ds]
    test_ds = [DataLoader(SentiDataset(*data), batch_size=args.test_batch_size) for data in test_ds]
    lexicons = []
    for tid, tlang in zip(trg_ids, args.trg):
        sv, tv = vocabs[src_id], vocabs[tid]
        lex, lexsz = load_lexicon('data/muse/{}-{}.0-5000.txt'.format(args.src, tlang), sv, tv)
        lexicons.append((lex, lexsz, tid))

    print('Statistics:')
    for lang, v in zip(LANGS, vocabs):
        print('\t{} vocab size: {}'.format(lang, len(v)))
    print()

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.resume:
        model, _, _, pool_layer, _, _ = model_load(args.resume)
        lang_dis = dom_dis = dis_opt = None
    else:
        raise NotImplementedError
    freeze_net(model.models)
    if args.optimizer == 'sgd':
        lm_opt = torch.optim.SGD(model.clfs.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        lm_opt = torch.optim.Adam(model.clfs.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))

    cross_entropy = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda(),  cross_entropy.cuda()
    else:
        model.cpu(),  cross_entropy.cpu()

    print('Parameters:')
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters() if x.size())
    print('\ttotal params:   {}'.format(total_params))
    print('\tparam list:     {}'.format(len(list(model.parameters()))))
    for name, x in model.named_parameters():
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
        total_clf_loss = 0
        start_time = time.time()
        model.train()
        model.reset_all()
        for epoch in range(args.epochs):
            lm_opt.zero_grad()
            try:
                xs, ys, ls = next(train_iter)
            except StopIteration:
                train_iter = iter(senti_train)
                xs, ys, ls = next(train_iter)
            clf_loss = cross_entropy(model(xs, ls, src_id, dom_id), ys)
            total_clf_loss += clf_loss.item()
            clf_loss.backward()
            lm_opt.step()

            if (epoch + 1) % args.log_interval == 0:
                total_clf_loss /= args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:4d} | lr {:05.5f} | ms/batch {:7.2f} | clf {:7.4f} |'.format(
                      epoch, lm_opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, total_clf_loss))
                total_clf_loss = 0
                start_time = time.time()

            if (epoch + 1) % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    train_acc = evaluate(model, train_ds, src_id, dom_id, args.test_batch_size)
                    val_accs = [evaluate(model, ds, tid, dom_id, args.test_batch_size) for tid, ds in zip(trg_ids, val_ds)]
                    test_accs = [evaluate(model, ds, tid, dom_id, args.test_batch_size) for tid, ds in zip(trg_ids, test_ds)]
                    bdi_accs = [compute_nn_accuracy(model.encoder_weight(src_id).cpu().numpy(),
                                                    model.encoder_weight(tid).cpu().numpy(),
                                                    lexicon, 10000, lexicon_size=lexsz) for lexicon, lexsz, tid in lexicons]
                    print_line()
                    print(('| epoch {:4d} | train {:.4f} |' +
                           ' val' + ' {} {:.4f}' * n_trg + ' |' +
                           ' test' + ' {} {:.4f}' * n_trg + ' |' +
                           ' bdi' + ' {} {:.4f}' * n_trg + ' |').format(epoch, train_acc,
                                                                        *sum([[tlang, acc] for tlang, acc in zip(args.trg, val_accs)], []),
                                                                        *sum([[tlang, acc] for tlang, acc in zip(args.trg, test_accs)], []),
                                                                        *sum([[tlang, acc] for tlang, acc in zip(args.trg, bdi_accs)], [])))
                    print_line()
                    for tlang, val_acc in zip(args.trg, val_accs):
                        if val_acc > best_accs[tlang]:
                            save_path = model_path.replace('.pt', '_{}.pt'.format(tlang))
                            print('saving {} model to {}'.format(tlang, save_path))
                            model_save(model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt, save_path)
                            best_accs[tlang] = val_acc

                model.train()
                start_time = time.time()

    except KeyboardInterrupt:
        print_line()
        print('Keyboard Interrupte - Exiting from training early')

    ###############################################################################
    # Testing
    ###############################################################################

    with torch.no_grad():
        test_accs = []
        for tid, tlang, ds in zip(trg_ids, args.trg, test_ds):
            save_path = model_path.replace('.pt', '_{}.pt'.format(tlang))
            model, lang_dis, dom_dis, pool_layer, lm_opt, dis_opt = model_load(save_path)   # Load the best saved model.
            model.eval()
            test_accs.append(evaluate(model, ds, tid, dom_id, args.test_batch_size))
    print_line()
    print(('|' + ' {}_test {:.4f} |' * n_trg).format(*sum([[tlang, acc] for tlang, acc in zip(args.trg, test_accs)], [])))
    print_line()

if __name__ == '__main__':
    main()
