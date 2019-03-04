import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.text.models.awd_lstm import get_language_model


def get_cross_lingual_language_model(src_ntok, trg_ntok, emb_sz, n_hid, n_layers, tie_weights,
                                     output_p, hidden_p, input_p, embed_p, weight_p, n_share=-1):

    src_lm = get_language_model(vocab_sz=src_ntok, emb_sz=emb_sz, n_hid=n_hid, n_layers=n_layers,
                                pad_token=None, tie_weights=True, qrnn=False, bias=True,
                                bidir=False, output_p=output_p, hidden_p=hidden_p, input_p=input_p,
                                embed_p=embed_p, weight_p=weight_p)

    trg_lm = get_language_model(vocab_sz=trg_ntok, emb_sz=emb_sz, n_hid=n_hid, n_layers=n_layers,
                                pad_token=None, tie_weights=True, qrnn=False, bias=True,
                                bidir=False, output_p=output_p, hidden_p=hidden_p, input_p=input_p,
                                embed_p=embed_p, weight_p=weight_p)

    if n_share == -1 or n_share == n_layers:
        list(trg_lm.children())[0].rnns = list(src_lm.children())[0].rnns  # share lstm parameters
    else:
        rnn_list = list(list(trg_lm.children())[0].rnns.children())
        rnn_list[:n_share] = list(next(src_lm.children()).rnns.children())[:n_share]
        next(trg_lm.children()).rnns = nn.ModuleList(rnn_list)

    return src_lm, trg_lm


class MultiLingualMultiDomainLM(nn.Module):

    def __init__(self, n_langs, n_doms, n_tok, emb_sz, n_hid, n_layers, n_share, tie_weights,
                 output_p, hidden_p, input_p, embed_p, weight_p, alpha, beta):
        super().__init__()
        self.n_langs = n_langs
        self.n_doms = n_doms
        self.n_tok = n_tok
        self.emb_sz = emb_sz
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_share = n_share
        self.tie_weights = tie_weights
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.input_p = input_p
        self.embed_p = embed_p
        self.weight_p = weight_p
        self.alpha = alpha
        self.beta = beta

        self.criterion = nn.NLLLoss()

        models = []
        encoders = []
        for lid in range(n_langs):
            lm = get_language_model(vocab_sz=n_tok[lid], emb_sz=emb_sz, n_hid=n_hid, n_layers=n_layers,
                                    pad_token=None, tie_weights=tie_weights, qrnn=False, bias=True,
                                    bidir=False, output_p=output_p, hidden_p=hidden_p, input_p=input_p,
                                    embed_p=embed_p, weight_p=weight_p)
            if lid == 0:
                all_shared_rnns = list(next(lm.children()).rnns.children())[:n_share]
                dom_shared_rnns = next(lm.children()).rnns
            else:
                next(lm.children()).rnns = dom_shared_rnns
            models.append(lm)
            encoders.append(next(lm.children()).encoder_dp)

        for did in range(n_doms - 1):
            for lid in range(n_langs):
                lm = get_language_model(vocab_sz=n_tok[lid], emb_sz=emb_sz, n_hid=n_hid, n_layers=n_layers,
                                        pad_token=None, tie_weights=tie_weights, qrnn=False, bias=True,
                                        bidir=False, output_p=output_p, hidden_p=hidden_p, input_p=input_p,
                                        embed_p=embed_p, weight_p=weight_p)
                # share rnn layers
                if lid == 0:
                    rnn_list = list(list(lm.children())[0].rnns.children())
                    rnn_list[:n_share] = all_shared_rnns
                    dom_shared_rnns = nn.ModuleList(rnn_list)
                    next(lm.children()).rnns = dom_shared_rnns
                else:
                    next(lm.children()).rnns = dom_shared_rnns

                # share language embeddings across domains
                next(lm.children()).encoder_dp = encoders[lid]
                next(lm.children()).encoder = encoders[lid].emb

                # tie decoder weight to language embeddings (optional)
                if tie_weights:
                    list(lm.children())[1].decoder.weight = encoders[lid].emb.weight
                else:
                    pass
                    # list(lm.children())[1].decoder.weight = list(models[lid].children())[1].decoder.weight

                models.append(lm)

        self.models = nn.ModuleList(models)
        self.encoders = [enc.emb for enc in encoders]

    def get_model_id(self, lid, did):
        return did * self.n_langs + lid

    def forward(self, X, lid, did):
        """
        X: Tensor of shape (batch, seq)
        """
        model_id = self.get_model_id(lid, did)
        return self.models[model_id](X)

    def reset(self, lid, did):
        model_id = self.get_model_id(lid, did)
        hidden = next(self.models[model_id].children()).hidden
        self.models[model_id].reset()
        return hidden

    def set_hidden(self, hidden, lid, did):
        model_id = self.get_model_id(lid, did)
        next(self.models[model_id].children()).hidden = hidden

    def reset_all(self):
        for model in self.models:
            model.reset()

    def single_loss(self, X, Y, lid, did, return_h=False, criterion=None, smooth_ids=None):
        """
        X: Tensor of shape (batch_size, seq_len)
        Y: Tensor of shape (batch_size, seq_len)
        lid: int
        did: int
        return_h: bool (optional)

        returns: float, float / float, float, List[Tensor]
        """
        bs, bptt = X.size()
        model_id = self.get_model_id(lid, did)
        output, hid, hid_drop = self.models[model_id](X)
        if smooth_ids is not None:
            loss = raw_loss = criterion(F.log_softmax(output.view(-1, output.size(-1)), -1), Y.view(-1), smooth_ids)
        else:
            loss = raw_loss = self.criterion(F.log_softmax(output.view(-1, output.size(-1)), -1), Y.view(-1))

        if self.alpha > 0 or self.beta > 0:
            loss = loss + sum(self.alpha * h.pow(2).mean() for h in hid_drop[-1:])
            loss = loss + sum(self.beta * (h[:, 1:] - h[:, :-1]).pow(2).mean() for h in hid[-1:])

        if return_h:
            return raw_loss, loss, hid
        else:
            return raw_loss, loss

    def encoder_weight(self, lid):
        """
        lid: int in [0, n_lang)

        returns: Tensor of shape (vocab_size, emb_dim)
        """
        return self.encoders[lid].weight.data

    def get_language_model(self, lid, did):
        """
        lid: int
        did: int

        returns: nn.Module
        """
        model_id = self.get_model_id(lid, did)
        return self.models[model_id]


class Discriminator(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, nlayers, dropout):
        """
        Discriminator initialization.
        """
        super(Discriminator, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.dropout = dropout

        layers = []
        for i in range(self.nlayers + 1):
            idim = self.in_dim if i == 0 else self.hid_dim
            odim = self.hid_dim if i < self.nlayers else self.out_dim
            layers.append(nn.Linear(idim, odim))
            if i < self.nlayers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class MeanPoolClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(in_dim, n_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, X, l):
        """
        X: Tensor of shape (batch_size, seq_len, in_dim)
        l: LongTensor of shape (batch_size)

        returns: Tensor of shape (batch_size, n_classes)
        """
        bs, sl, _ = X.size()
        idxes = torch.arange(0, sl).unsqueeze(0).to(X.device)
        mask = (idxes < l.unsqueeze(1)).float()
        pooled = (X * mask.unsqueeze(-1)).sum(1) / l.float().unsqueeze(-1)
        dropped = self.dp(pooled)
        return self.lin(dropped)


class MaxPoolClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(in_dim, n_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, X, l):
        """
        X: Tensor of shape (batch_size, seq_len, in_dim)
        l: LongTensor of shape (batch_size)

        returns: Tensor of shape (batch_size, n_classes)
        """
        bs, sl, _ = X.size()
        idxes = torch.arange(0, sl).unsqueeze(0).to(X.device)
        mask = (idxes >= l.unsqueeze(1)).unsqueeze(-1)
        X = X.clone().masked_fill_(mask, float('-inf'))
        pooled, _ = X.max(1)
        dropped = self.dp(pooled)
        return self.lin(dropped)


class MeanMaxPoolClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(in_dim * 2, n_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, X, l):
        """
        X: Tensor of shape (batch_size, seq_len, in_dim)
        l: LongTensor of shape (batch_size)

        returns: Tensor of shape (batch_size, n_classes)
        """
        bs, sl, _ = X.size()
        idxes = torch.arange(0, sl).unsqueeze(0).to(X.device)
        mask = (idxes < l.unsqueeze(1)).float()
        X_masked = X * mask.unsqueeze(-1)
        max_pooled, _ = X_masked.max(1)
        mean_pooled = X_masked.sum(1) / l.float().unsqueeze(-1)
        dropped = self.dp(torch.cat([max_pooled, mean_pooled], -1))
        return self.lin(dropped)


class MultiLingualMultiDomainClassifier(MultiLingualMultiDomainLM):

    def __init__(self, n_classes, pool_layer, clf_dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.pool_layer = pool_layer

        clf_in_dim = self.emb_sz

        if self.pool_layer == 'mean':
            self.clfs = [MeanPoolClassifier(clf_in_dim, n_classes, clf_dropout) for _ in range(self.n_doms)]
        elif self.pool_layer == 'max':
            self.clfs = [MaxPoolClassifier(clf_in_dim, n_classes, clf_dropout) for _ in range(self.n_doms)]
        elif self.pool_layer == 'meanmax':
            self.clfs = [MeanMaxPoolClassifier(clf_in_dim, n_classes, clf_dropout) for _ in range(self.n_doms)]
        self.clfs = nn.ModuleList(self.clfs)

    def forward(self, X, lengths, lid, did):
        prev_h = self.reset(lid, did)
        _, hidden, _ = super(MultiLingualMultiDomainClassifier, self).forward(X, lid, did)
        self.set_hidden(prev_h, lid, did)
        return self.clfs[did](hidden[-1], lengths)


class MaxPoolLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pooled, _ = x.max(1)
        return pooled


class MeanPoolLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(1)


class MeanMaxPoolLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_pooled, _ = x.max(1)
        mean_pooled = x.mean(1)
        return torch.cat([max_pooled, mean_pooled], -1)


def get_pooling_layer(pool_layer):
    return {
        'mean': MeanPoolLayer,
        'max': MaxPoolLayer,
        'meanmax': MeanMaxPoolLayer,
    }.get(pool_layer)()


class PoolDiscriminator(nn.Module):

    def __init__(self, pool_layer, in_dim, hid_dim, out_dim, nlayers, dropout):
        super().__init__()

        clf_in_dim = {
            'mean': in_dim,
            'max': in_dim,
            'meanmax': in_dim * 2,
        }.get(pool_layer)
        self.clf = Discriminator(clf_in_dim, hid_dim, out_dim, nlayers, dropout)
        self.pool = get_pooling_layer(pool_layer)

    def forward(self, x):
        return self.clf(self.pool(x))

if __name__ == '__main__':
    print('testing MeanPoolClassifier')
    for k in range(10):
        x = torch.randn(20, 100, 50)
        l = torch.randint(10, 40, (20,))
        clf = MeanPoolClassifier(50, 10, 0.2).eval()
        pred = clf(x, l)
        for i, ll in enumerate(l):
            x[i, ll:] = 0
        pooled_gold = x.sum(1) / l.float().unsqueeze(-1)
        pred_gold = clf.lin(clf.dp(pooled_gold))
        assert (pred == pred_gold).all()
        print('passed test {}'.format(k))
