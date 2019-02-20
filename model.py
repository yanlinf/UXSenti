import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.text.models.awd_lstm import *


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
