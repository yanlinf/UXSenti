import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import *


class LSTMLanguageModel(nn.Module):
    """
    An AWD-LSTM language model implementation adapapted from
    https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py
    """
    init_range = 0.1

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, tie_weights,
                 output_p, hidden_p, input_p, embed_p, weight_p):
        super().__init__()
        self.vocab_size, self.emb_size, self.hidden_size, self.num_layers = \
            vocab_size, emb_size, hidden_size, num_layers
        self.tie_weights = tie_weights
        self.output_p, self.hidden_p, self.input_p, self.embed_p, self.weight_p = \
            output_p, hidden_p, input_p, embed_p, weight_p
        self.bs = 1

        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.encoder.weight.data.uniform_(-self.init_range, self.init_range)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.input_dp = RNNDropout(input_p)
        self.rnn = MultiLayerLSTM(emb_size, hidden_size, num_layers, emb_size,
                                  dropout=hidden_p, weight_dropout=weight_p)
        self.ouput_dp = RNNDropout(output_p)
        self.decoder = nn.Linear(emb_size, vocab_size)

        if tie_weights:
            self.decoder.weight = self.encoder.weight
        self.reset()

    def forward(self, inputs):
        bs, sl = inputs.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()

        outputs = self.input_dp(self.encoder_dp(inputs))
        outputs, hidden, raw_outputs = self.rnn(outputs, self.hidden, True)
        decoded = self.decoder(self.ouput_dp(outputs))
        self.hidden = self._to_detach(hidden)
        return decoded, raw_outputs

    def _to_detach(self, x):
        if isinstance(x, (list, tuple)):
            return [self._to_detach(b) for b in x]
        else:
            return x.detach()

    def _init_h(self, l):
        n_hid = self.hidden_size if l != self.num_layers - 1 else self.emb_size
        return self.weights.new(1, self.bs, n_hid).zero_()

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self._init_h(l), self._init_h(l)) for l in range(self.num_layers)]


class XLXDLM(nn.Module):
    """
    Cross-lingual cross-domain language model with parameter sharing.

    Paramters
    ---------
    n_langs: number of languages
    n_doms: number of domains
    num_layers: number of lstm layers
    num_share: number of lstm layers that are shared across all langs/doms, the other
               lstm layers are still shared across languages but are domain-specific
    """

    def __init__(self, n_langs, n_doms, vocab_sizes, emb_size, hidden_size, num_layers, num_share, tie_weights,
                 output_p, hidden_p, input_p, embed_p, weight_p):
        super().__init__()
        self.n_langs = n_langs
        self.n_doms = n_doms
        self.vocab_sizes = vocab_sizes
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_share = num_share
        self.tie_weights = tie_weights
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.input_p = input_p
        self.embed_p = embed_p
        self.weight_p = weight_p

        encoders = []
        models = []
        for lid in range(n_langs):
            for did in range(n_doms):
                lm = LSTMLanguageModel(vocab_sizes[lid], emb_size, hidden_size, num_layers, tie_weights,
                                       output_p, hidden_p, input_p, embed_p, weight_p)
                if lid > 0 or did > 0:  # share rnn layers across all langs / doms
                    for i in range(num_share):
                        lm.rnn.rnns[i] = models[0].rnn.rnns[i]
                if lid > 0:  # share domain specific rnn layers across languages
                    for i in range(num_share, num_layers):
                        lm.rnn.rnns[i] = models[did].rnn.rnns[i]
                if did == 0:
                    encoders.append(lm.encoder)
                else:
                    lm.encoder = models[-1].encoder
                    lm.encoder_dp = models[-1].encoder_dp

                if tie_weights:
                    lm.decoder.weight = lm.encoder.weight

                models.append(lm)

        self.models = nn.ModuleList(models)
        self.encoders = nn.ModuleList(encoders)
        self.crit = nn.CrossEntropyLoss()

    def encoder_weight(self, lid):
        return self.encoders[lid].weight.data.cpu().numpy()

    def _get_model_id(self, lid, did):
        return lid * self.n_doms + did

    def forward(self, inputs, lid, did):
        return self.models[self._get_model_id(lid, did)](inputs)

    def lm_loss(self, inputs, target, lid, did, return_h=False):
        decoded, raw_outputs = self.models[self._get_model_id(lid, did)](inputs)
        loss = self.crit(decoded.view(-1, decoded.size(-1)), target.view(-1))
        if return_h:
            return loss, raw_outputs
        else:
            return loss

    def reset(self, lid=None, did=None):
        if lid is None or did is None:
            return [[self.reset(lid, did) for did in range(self.n_doms)] for lid in range(self.n_langs)]
        else:
            lm = self.models[self._get_model_id(lid, did)]
            hidden = lm.hidden
            lm.reset()
            return hidden

    def set_hidden(self, hidden, lid, did):
        mid = self._get_model_id(lid, did)
        self.models[mid].hidden = hidden


class MeanPoolClassifier(nn.Module):

    def __init__(self, input_size, n_classes, dropout):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.dropout = dropout

        self.linear = nn.Linear(input_size, n_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        bs, sl, _ = inputs.size()
        idxes = torch.arange(0, sl).unsqueeze(0).to(inputs.device)
        mask = (idxes < lengths.unsqueeze(1)).float()
        pooled = (inputs * mask.unsqueeze(-1)).sum(1) / lengths.float().unsqueeze(-1)
        dropped = self.dp(pooled)
        return self.linear(dropped)


class XLXDClassifier(XLXDLM):
    """
    A sentiment classifier that uses XLXDLM as the feature extractor.

    Paramters
    ---------
    n_classes: number of target labels
    clf_p: dropout applied to the input features
    """

    def __init__(self, n_classes, clf_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.clf_p = clf_p

        self.clfs = [MeanPoolClassifier(self.emb_size, n_classes, clf_p) for _ in range(self.n_doms)]
        self.clfs = nn.ModuleList(self.clfs)
        self.clf_crit = nn.CrossEntropyLoss()

    def forward(self, inputs, lengths, lid, did):
        prev_h = self.reset(lid, did)
        _, raw_outputs = super().forward(inputs, lid, did)
        self.set_hidden(prev_h, lid, did)
        return self.clfs[did](raw_outputs[-1], lengths)

    def clf_loss(self, inputs, lengths, label, lid, did):
        logits = self.forward(inputs, lengths, lid, did)
        loss = self.clf_crit(logits, label)
        return loss


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            layers.append(nn.Linear(n_in, n_out))
            if i < self.num_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
