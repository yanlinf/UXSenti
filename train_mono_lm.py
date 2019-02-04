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
from model import RNNModel
from utils.vocab import *
from utils.data import *

TEST_SENTS = [
    'i love',
    'this book is',
    'i likes the',
    'do you feel different from people',
]
PRED_NWORDS = 20


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def model_save(model, criterion, optimizer, path):
    with open(path, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(path):
    with open(path, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

    return model, criterion, optimizer


def evaluate(data_source, batch_size, model, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN':
        model.reset()

    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args.bptt, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(F.log_softmax(model.decoder(output), -1), targets).data
        hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)


def sample(x, model, n_words, args):
    model.eval()
    if args.model == 'QRNN':
        model.reset()

    x = x.view(-1, 1)
    hidden = model.init_hidden(1)
    res = []
    for i in range(n_words):
        output, hidden = model(x, hidden)
        output = model.decoder(output)
        _, pred = output[-1].max(-1)
        res.append(pred.item())
        x = pred.view(1, 1)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--src', default='en', help='source_language')
    parser.add_argument('-trg', '--trg', default='fr', help='target_language')
    parser.add_argument('--src_dom', default='books', help='source domain')
    parser.add_argument('--trg_dom', default='books', help='target domain')
    # parser.add_argument('--data', default='data/', help='directory of the vocab and corpus')

    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')

    parser.add_argument('--epochs', type=int, default=8000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N', help='batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    randomhash = str(time.time()).split('.')[0]
    parser.add_argument('--export', type=str,  default=randomhash, help='dir to save the model')

    parser.add_argument('-lr', '--lr', type=float, default=0.004, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--optimizer', type=str,  default='adam', help='optimizer to use (sgd, adam)')
    parser.add_argument('--adam_beta', type=float, default=0.7, help='beta of adam')

    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='', help='path of model to resume')
    parser.add_argument('--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    args = parser.parse_args()
    args.tied = True

    # Set the random seed manually for reproducibility.
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
    print('Statistics:')

    # load vocabulary
    src_vocab_file = os.path.join('data', 'vocab_{}.txt'.format(args.src))
    trg_vocab_file = os.path.join('data', 'vocab_{}.txt'.format(args.trg))
    src_vocab = Vocab(path=src_vocab_file)
    trg_vocab = Vocab(path=trg_vocab_file)
    print('\tsrc vocab size: {}'.format(len(src_vocab)))
    print('\ttrg vocab size: {}'.format(len(trg_vocab)))

    with open(os.path.join('pickle', 'en'), 'rb') as fin:
        src_x = pickle.load(fin)

    if args.cuda:
        src_x = src_x.cuda()

    print('\tcorpus size:    {}'.format(src_x.size(0)))
    size = src_x.size(0)
    train_set = src_x[:int(size * 0.7)]
    valid_set = src_x[int(size * 0.7):int(size * 0.8)]
    test_set = src_x[int(size * 0.8):]

    print('\ttrain size:     {}'.format(train_set.size(0)))
    print('\tval size:       {}'.format(valid_set.size(0)))
    print('\ttest size:      {}'.format(test_set.size(0)))

    val_batch_size = args.batch_size
    test_batch_size = 1
    train_data = batchify(train_set, args.batch_size)
    val_data = batchify(valid_set, val_batch_size)
    test_data = batchify(test_set, test_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(src_vocab)

    if args.resume:
        # print('Resuming model ...')
        model, criterion, optimizer = model_load(args.resume)
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid,
                         args.nlayers, args.dropout, args.dropouth,
                         args.dropouti, args.dropoute, args.wdrop, args.tied)
        criterion = nn.NLLLoss()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.adam_beta, 0.999))

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters() if x.size())
    print('\ttotal params:   {}'.format(total_params))
    print()

    ###############################################################################
    # Training code
    ###############################################################################

    # Loop over epochs.
    lr = args.lr
    best_val_loss = float('inf')

    print('Traning:')
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            if args.model == 'QRNN':
                model.reset()

            total_loss = 0
            start_time = time.time()
            hidden = model.init_hidden(args.batch_size)
            nbatch, i = 0, 0
            while i < train_data.size(0) - 1 - 1:
                # setting model to training model and clear gradients
                model.train()
                optimizer.zero_grad()

                # sample seq_len
                bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))

                # adjust lr according to lr
                lr2 = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt

                # fetch a batch of data
                data, targets = get_batch(train_data, i, args.bptt, seq_len=seq_len)

                # detatch hidden
                hidden = repackage_hidden(hidden)

                # forward pass and backward pass
                output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
                log_probs = F.log_softmax(model.decoder(output), -1)
                raw_loss = criterion(log_probs, targets)
                loss = raw_loss
                if args.alpha:  # AR regularization
                    loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                if args.beta:  # TAR regularization
                    loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss.backward()

                # clip gradients
                if args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                # take one optimizer step
                optimizer.step()

                total_loss += raw_loss.data

                optimizer.param_groups[0]['lr'] = lr2

                if (nbatch + 1) % args.log_interval == 0:
                    cur_loss = total_loss.item() / args.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                          'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                              epoch, nbatch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                    total_loss = 0
                    start_time = time.time()

                nbatch += 1
                i += seq_len

            val_loss = evaluate(val_data, val_batch_size, model, criterion, args)
            print('-' * 96)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                      epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 96)

            if val_loss < best_val_loss:
                model_save(model, criterion, optimizer, model_path)
                print('saving model to {}'.format(model_path))
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 96)
        print('Keyboard Interrupte - Exiting from training early')

    model_load(model_path)   # Load the best saved model.

    print('Generated sentences')
    for sent in TEST_SENTS:
        print([src_vocab.w2idx[w] for w in sent.split()])
        word_ids = torch.tensor([src_vocab.w2idx[w] for w in sent.split()])
        if args.cuda:
            word_ids = word_ids.cuda()
        pred = sample(word_ids, model, PRED_NWORDS, args)
        print(' '.join(sent.split() + [src_vocab.idx2w[i] for i in pred]))

    test_loss = evaluate(test_data, test_batch_size, model, criterion, args)
    print('-' * 96)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('-' * 96)


if __name__ == '__main__':
    main()
