import torch
import torch.nn as nn
import torch.nn.functional as F


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
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
