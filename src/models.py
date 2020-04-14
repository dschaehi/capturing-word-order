import math

import torch
import torch.nn as nn


class ExpandedLinear(nn.Module):
    def __init__(self, in_bigram_vec_dim, out_bigram_vec_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_bigram_vec_dim, out_bigram_vec_dim))
        self.bias = nn.Parameter(torch.Tensor(out_bigram_vec_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, bigram_vecs):
        """
        bigram_vecs: (batch_size, max_sent_len, in_bigram_vec_dim)
        """
        # W: (batch_size, max_sent_len, in_bigram_vec_dim, out_bigram_vec_dim)
        W = self.weight.expand(*bigram_vecs.size(), self.weight.size(1))
        # bigram_vecs: (batch_size, max_sent_len, 1, in_bigram_vec_dim)
        bigram_vecs = bigram_vecs.unsqueeze(2)
        # out: (batch_size, max_sent_len, 1, out_bigram_vec_dim)
        out = torch.matmul(bigram_vecs, W)
        # out: (batch_size, max_sent_len, out_bigram_vec_dim)
        out = out.squeeze(2)
        out += self.bias
        return out

    def extra_repr(self):
        return "in_bigram_vec_dim={}, out_bigram_vec_dim={}".format(*self.weight.size())


class Net(nn.Module):
    def __init__(self, word_vec_dim, bigram_fn, out_bigram_vec_dim):
        super().__init__()
        self.bigram_fn = bigram_fn
        # Compute the input bigram vector dimension for the linear layer.
        in_bigram_vec_dim = bigram_fn(torch.randn((1, 2, word_vec_dim))).size(2)
        self.out_bigram_vec_dim = out_bigram_vec_dim
        self.T = ExpandedLinear(in_bigram_vec_dim, self.out_bigram_vec_dim)

    def forward(self, vec_sents, aggregate=True):
        # in_bigram_vecs: (batch_size, max_sent_len - 1, in_bigram_vec_dim)
        in_bigram_vecs = self.bigram_fn(vec_sents)
        # out_bigram_vecs: (batch_size, max_sent_len - 1, out_bigram_vec_dim)
        out_bigram_vecs = (
            self.T(in_bigram_vecs)
            * (in_bigram_vecs != 0)[:, :, : self.out_bigram_vec_dim].float()
        )
        if aggregate:
            # output: (batch_size, out_bigram_vec_dim)
            output = torch.sum(torch.tanh(out_bigram_vecs), dim=1)
        else:
            # output: (batch_size, max_sent_len - 1, out_bigram_vec_dim)
            output = torch.tanh(out_bigram_vecs)
        return output
