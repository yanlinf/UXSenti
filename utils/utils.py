def print_ids(ids, vocab):
    print(' '.join(vocab.idx2w[i] for i in ids))


def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True
