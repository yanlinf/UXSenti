import torch


def print_ids(ids, vocab):
    print(' '.join(vocab.idx2w[i] for i in ids))


def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True


def to_device(obj, cuda):
    if isinstance(obj, (tuple, list)):
        return [to_device(x, cuda) for x in obj]
    else:
        return obj.cuda() if cuda else obj.cpu()


def sample(x_list, n_samples, shuffle=True):
    if shuffle:
        size = x_list[0].size(0)
        perm = torch.randperm(size)
        return [x[perm][:n_samples] for x in x_list]
    else:
        return [x[:n_samples] for x in x_list]
