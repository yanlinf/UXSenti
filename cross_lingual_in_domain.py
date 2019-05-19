import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
import json
from model import XLXDClassifier, Discriminator
from utils.vocab import *
from utils.data import *
from utils.utils import *
from utils.bdi import *
from utils.module import *

LINE_WIDTH = 137


def print_line():
    print('-' * LINE_WIDTH)


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
        if k not in ('resume', 'mode'):
            setattr(args, k, dic[k])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['en', 'fr', 'de', 'ja'], nargs='+', default=['en', 'fr', 'de', 'ja'], help='languages')
    parser.add_argument('--dom', choices=['books', 'dvd', 'music'], nargs='+', default=['books', 'dvd', 'music'], help='domains')
    parser.add_argument('-src', '--src', choices=['en', 'fr', 'de', 'ja'], default='en', help='l_s, source_language')
    parser.add_argument('-trg', '--trg', choices=['en', 'fr', 'de', 'ja'], nargs='+', default=['fr', 'de', 'ja'], help='l_t, target_language')
    parser.add_argument('--sup_dom', choices=['books', 'dvd', 'music'], default='books', help='d_s (= d_t), domain to perform supervised learning')
    parser.add_argument('--train', default='data/train.pth', help='traning and testing data')
    parser.add_argument('--test', default='data/test.pth', help='traning and testing data')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--val_size', type=int, default=600, help='validation set size')
    parser.add_argument('--early_stopping',  type=bool_flag, nargs='?', const=True, default=False, help='perform early stopping')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train', help='train or evaluate')

    # architecture
    parser.add_argument('--emb_dim', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--hid_dim', type=int, default=1150, help='number of hidden units per layer of the language model')
    parser.add_argument('--dis_hid_dim', type=int, default=400, help='number of hidden units per layer of the discriminator')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--nshare', type=int, default=1, help='number of rnn layers to share')
    parser.add_argument('--dis_nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--tie_softmax', type=bool_flag, nargs='?', const=True, default=True, help='tie sofmat layer to the embedding layer')

    # loss function
    parser.add_argument('--lambd_lm', type=float, default=1, help='coefficient of the language modeling')
    parser.add_argument('--lambd_dis', type=float, default=0.1, help='coefficient of the adversarial loss')
    parser.add_argument('--lambd_clf', type=float, default=0.01, help='coefficient of the classification loss')

    # regularization
    parser.add_argument('--dropouto', type=float, default=0.4, help='dropout applied to rnn outputs')
    parser.add_argument('--dropoutc', type=float, default=0.6, help='dropout applied to classifier')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers')
    parser.add_argument('--dropouti', type=float, default=0.4, help='dropout for input embedding layers')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer')
    parser.add_argument('--dropoutw', type=float, default=0.5, help='weight dropout applied to the RNN hidden to hidden matrix')
    parser.add_argument('--dropoutd', type=float, default=0.1, help='dropout applied to language discriminator')
    parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')

    # optimization
    parser.add_argument('--max_steps', type=int, default=50000, help='upper step limit')
    parser.add_argument('-bs', '--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('-cbs', '--clf_batch_size', type=int, default=20, help='classification batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=100, help='classification batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--optimizer', type=str,  default='adam', choices=['adam', 'sgd'], help='optimizer to use (sgd, adam)')
    parser.add_argument('--beta1', type=float, default=0.7, help='beta of adam')
    parser.add_argument('--dis_nsteps', type=int, help='n discriminator steps for each lm step')
    parser.add_argument('-lr', '--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--dis_lr', type=float, default=0.003, help='initial learning rate for the discriminators')
    parser.add_argument('--grad_clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--dis_clip', type=float, default=1, help='gradient clipping for discriminator')

    # device / logging settings
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--val_interval', type=int, default=1000, metavar='N', help='validation interval')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--export', type=str,  default='export/', help='dir to save the model')

    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(log_interval=20, val_interval=40)
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

    n_trg = len(args.trg)
    lang2id = {lang: i for i, lang in enumerate(args.lang)}
    dom2id = {dom: i for i, dom in enumerate(args.dom)}
    src_id, dom_id = lang2id[args.src], dom2id[args.sup_dom]
    trg_ids = [lang2id[t] for t in args.trg]

    train_set = torch.load(args.train)
    test_set = torch.load(args.test)
    vocabs = [train_set[lang]['vocab'] for lang in args.lang]
    unlabeled = to_device([[batchify(train_set[lang][dom]['unlabeled'], args.batch_size) for dom in args.dom] for lang in args.lang], args.cuda)
    train_x, train_y, train_l = to_device(train_set[args.src][args.sup_dom]['train'], args.cuda)
    val_ds = [sample(to_device(train_set[t][args.sup_dom]['train'], args.cuda), args.val_size) for t in args.trg]
    test_ds = [to_device(test_set[t][args.sup_dom]['test'], args.cuda) for t in args.trg]

    senti_train = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.clf_batch_size)
    train_iter = iter(senti_train)
    train_ds = DataLoader(SentiDataset(train_x, train_y, train_l), batch_size=args.test_batch_size)
    val_ds = [DataLoader(SentiDataset(*ds), batch_size=args.test_batch_size) for ds in val_ds]
    test_ds = [DataLoader(SentiDataset(*ds), batch_size=args.test_batch_size) for ds in test_ds]

    lexicons = []
    for tid, tlang in zip(trg_ids, args.trg):
        sv, tv = vocabs[src_id], vocabs[tid]
        lex, lexsz = load_lexicon('data/muse/{}-{}.0-5000.txt'.format(args.src, tlang), sv, tv)
        lexicons.append((lex, lexsz, tid))

    ###############################################################################
    # Build the model
    ###############################################################################
    if args.resume:
        model, dis, lm_opt, dis_opt = model_load(args.resume)

    else:
        model = XLXDClassifier(n_classes=2, clf_p=args.dropoutc, n_langs=len(args.lang), n_doms=len(args.dom),
                               vocab_sizes=list(map(len, vocabs)), emb_size=args.emb_dim, hidden_size=args.hid_dim,
                               num_layers=args.nlayers, num_share=args.nshare, tie_weights=args.tie_softmax,
                               output_p=args.dropouto, hidden_p=args.dropouth, input_p=args.dropouti, embed_p=args.dropoute, weight_p=args.dropoutw)

        dis = Discriminator(args.emb_dim, args.dis_hid_dim, len(args.lang), args.dis_nlayers, args.dropoutd)

        if args.optimizer == 'sgd':
            lm_opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            dis_opt = torch.optim.SGD(dis.parameters(), lr=args.dis_lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            lm_opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.beta1, 0.999))
            dis_opt = torch.optim.Adam(dis.parameters(), lr=args.dis_lr, weight_decay=args.wdecay, betas=(args.beta1, 0.999))

    crit = nn.CrossEntropyLoss()

    bs = args.batch_size
    n_doms = len(args.dom)
    n_langs = len(args.lang)
    dis_y = to_device(torch.arange(n_langs).unsqueeze(-1).expand(n_langs, bs).contiguous().view(-1), args.cuda)

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
    best_accs = {tlang: 0. for tlang in args.trg}
    print('Traning:')
    print_line()
    ptrs = np.zeros((len(args.lang), len(args.dom)), dtype=np.int64)  # pointers for reading unlabeled data, of shape (n_lang, n_dom)
    total_loss = np.zeros((len(args.lang), len(args.dom)))  # shape (n_lang, n_dom)
    total_clf_loss = 0
    total_dis_loss = 0
    start_time = time.time()
    model.train()
    model.reset()
    for step in range(args.max_steps):
        loss = 0
        lm_opt.zero_grad()
        dis_opt.zero_grad()

        # language modeling loss
        dis_x = []
        for lid, t in enumerate(unlabeled):
            for did, lm_x in enumerate(t):
                if ptrs[lid, did] + bptt + 1 > lm_x.size(0):
                    ptrs[lid, did] = 0
                    model.reset(lid=lid, did=did)
                p = ptrs[lid, did]
                xs = lm_x[p: p + bptt].t().contiguous()
                ys = lm_x[p + 1: p + 1 + bptt].t().contiguous()
                lm_loss, hid = model.lm_loss(xs, ys, lid=lid, did=did, return_h=True)
                loss = loss + lm_loss * args.lambd_lm
                total_loss[lid, did] += lm_loss.item()
                ptrs[lid, did] += bptt
                if did == dom_id:
                    dis_x.append(hid[-1].mean(1))

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
        clf_loss = crit(model(xs, ls, src_id, dom_id), ys)
        total_clf_loss += clf_loss.item()
        (args.lambd_clf * clf_loss).backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.dis_clip > 0:
            for x in dis.parameters():
                x.data.clamp_(-args.dis_clip, args.dis_clip)
        dis_opt.step()
        lm_opt.step()

        if (step + 1) % args.log_interval == 0:
            total_loss /= args.log_interval
            total_clf_loss /= args.log_interval
            total_dis_loss /= args.log_interval
            elapsed = time.time() - start_time
            print('| step {:4d} | lr {:05.5f} | ms/batch {:7.2f} | lm_loss {:7.4f} | avg_ppl {:7.2f} | clf {:7.4f} | dis {:7.4f} |'.format(
                step, lm_opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                total_loss.mean(), np.exp(total_loss).mean(), total_clf_loss, total_dis_loss))
            total_loss[:, :], total_clf_loss, total_dis_loss = 0, 0, 0
            start_time = time.time()

        if (step + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                train_acc = evaluate(model, train_ds, src_id, dom_id)
                val_accs = [evaluate(model, ds, tid, dom_id) for tid, ds in zip(trg_ids, val_ds)]
                test_accs = [evaluate(model, ds, tid, dom_id) for tid, ds in zip(trg_ids, test_ds)]
                bdi_accs = [compute_nn_accuracy(model.encoder_weight(src_id),
                                                model.encoder_weight(tid),
                                                lexicon, 10000, lexicon_size=lexsz) for lexicon, lexsz, tid in lexicons]
                print_line()
                print(('| step {:4d} | train {:.4f} |' +
                       ' val' + ' {} {:.4f}' * n_trg + ' |' +
                       ' test' + ' {} {:.4f}' * n_trg + ' |' +
                       ' bdi' + ' {} {:.4f}' * n_trg + ' |').format(step, train_acc,
                                                                    *sum([[tlang, acc] for tlang, acc in zip(args.trg, val_accs)], []),
                                                                    *sum([[tlang, acc] for tlang, acc in zip(args.trg, test_accs)], []),
                                                                    *sum([[tlang, acc] for tlang, acc in zip(args.trg, bdi_accs)], [])))
                print_line()
                print('saving model to {}'.format(model_path.replace('.pt', '_final.pt')))
                model_save(model, dis, lm_opt, dis_opt, model_path.replace('.pt', '_final.pt'))
                for tlang, val_acc in zip(args.trg, val_accs):
                    if val_acc > best_accs[tlang]:
                        save_path = model_path.replace('.pt', '_{}.pt'.format(tlang))
                        print('saving {} model to {}'.format(tlang, save_path))
                        model_save(model, dis, lm_opt, dis_opt, save_path)
                        best_accs[tlang] = val_acc
                print_line()

            model.train()
            start_time = time.time()

    print_line()
    print(('Training ends - best val acc:' + ' {} {:.4f}' * n_trg).format(*sum([[tlang, best_accs[tlang]] for tlang in args.trg], [])))


def eval(args):
    n_trg = len(args.trg)
    lang2id = {lang: i for i, lang in enumerate(args.lang)}
    dom2id = {dom: i for i, dom in enumerate(args.dom)}
    dom_id = dom2id[args.sup_dom]
    trg_ids = [lang2id[t] for t in args.trg]

    test_set = torch.load(args.test)
    test_ds = [to_device(test_set[t][args.sup_dom]['test'], args.cuda) for t in args.trg]
    test_ds = [DataLoader(SentiDataset(*ds), batch_size=args.test_batch_size) for ds in test_ds]

    with torch.no_grad():
        test_accs = []
        for tid, tlang, ds in zip(trg_ids, args.trg, test_ds):
            if args.early_stopping:
                save_path = os.path.join(args.export, 'model_{}.pt'.format(tlang))
            else:
                save_path = os.path.join(args.export, 'model_final.pt')
            model, dis, lm_opt, dis_opt = model_load(save_path)   # Load the best saved model.
            model.eval()
            test_accs.append(evaluate(model, ds, tid, dom_id))
    print_line()
    print(('|' + ' {}_test {:.4f} |' * n_trg).format(*sum([[tlang, acc] for tlang, acc in zip(args.trg, test_accs)], [])))
    print_line()

if __name__ == '__main__':
    main()
