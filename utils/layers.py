import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        module.register_parameter(name_w + '_raw', nn.Parameter(w.data))
        module._parameters[name_w] = F.dropout(w, p=dropout, training=False)

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            module._parameters[name_w] = F.dropout(raw_w, p=dropout, training=module.training)

        return original_module_forward(*args)

    setattr(module, 'forward', forward)


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    """
    Dropout with probability `p` that is consistent on the seq_len dimension.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class WeightDropout(nn.Module):
    """
    A module that warps another layer in which some weights will be replaced by 0 during training.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, module, weight_p, layer_names=['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            raw_w = nn.Parameter(w.data)
            self.register_parameter(f'{layer}_raw', raw_w)
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class MultiLayerLSTM(nn.Module):
    """
    A multi-layer LSTM with specified output dimension and weight dropout.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=None,
                 bidirectional=False, dropout=0, weight_dropout=0, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        if output_size is None:
            output_size = hidden_size
        self.rnns = [nn.LSTM(input_size if l == 0 else hidden_size,
                             hidden_size if l != num_layers - 1 else output_size,
                             1, bidirectional=bidirectional,
                             batch_first=batch_first) for l in range(num_layers)]
        self.rnns = [WeightDropout(rnn, weight_dropout) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dps = nn.ModuleList([RNNDropout(dropout) for l in range(num_layers)])
        # self._set_weights()

    def _set_weights(self):
        w_names = ['weight_ih_l{}', 'bias_ih_l{}', 'bias_hh_l{}']
        for l in range(self.num_layers):
            for w_name in w_names:
                w = getattr(self.rnns[l].module, w_name.format(0))
                setattr(self, w_name.format(l), w)
            self.weight_hh_l0_raw = self.rnns[l].weight_hh_l0_raw

    def forward(self, inputs, hx=None, return_raw_output=False):
        new_h = []
        raw_outputs = []
        outputs = inputs
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            outputs, hid = rnn(outputs, hx[l] if hx else None)
            new_h.append(hid)
            raw_outputs.append(outputs)
            if l != self.num_layers - 1:
                outputs = hid_dp(outputs)
        if return_raw_output:
            return outputs, new_h, raw_outputs
        else:
            return outputs, new_h
