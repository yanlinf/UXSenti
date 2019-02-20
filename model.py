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
        self.models[model_id].reset()
        return self

    def reset_all(self):
        for model in self.models:
            model.reset()
        return self

    def single_loss(self, X, Y, lid, did, return_h=False):
        """
        X: Tensor of shape (batch_size, seq_len)
        Y: Tensor of shape (batch_size, seq_len)
        lid: int
        did: int
        return_h: bool (optional)
        """
        bs, bptt = X.size()
        output, hid, hid_drop = self.forward(X, lid, did)
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
