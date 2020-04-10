import math
import pickle
from pathlib import Path

import git
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from analyze_bigram_encoders import gen_neg_bigram_ixs, gen_pos_bigram_ixs
from misc import (
    WV,
    BigramEncoder,
    Corpus,
    DistanceFunction,
    LossFunction,
    process_word_vecs,
)


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)
FAST_TEXT = ROOT / "data/raw/crawl-300d-2M.vec"
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "models"

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def train(
    ix_sents,
    sent_lengths,
    ix_sents_dev,
    sent_lengths_dev,
    model,
    word_vecs,
    dist_fn,
    loss_fn,
    lr,
    num_epochs,
    batch_size,
    test_model,
    test_freq,
    seed=0,
    device=None,
):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = num_epochs * ix_sents.size(0)
    with tqdm(total=total) as pbar:
        for epoch in range(num_epochs):
            perm = torch.randperm(ix_sents.shape[0])
            ix_sents = ix_sents[perm]
            sent_lengths = sent_lengths[perm]
            for i in range(0, ix_sents.size(0), batch_size):
                optimizer.zero_grad()
                ix_sents_batch = ix_sents[i : i + batch_size].to(device)
                pos_bigram_ixs = gen_pos_bigram_ixs(ix_sents_batch, device=device)
                neg_bigram_ixs = gen_neg_bigram_ixs(ix_sents_batch, device=device)

                pos_vbigrams = wv.vecs[pos_bigram_ixs].to(device)
                neg_vbigrams = wv.vecs[neg_bigram_ixs].to(device)
                vsents = wv.vecs[ix_sents_batch].to(device)
                pos_dists = dist_fn(model(vsents), model(pos_vbigrams))
                neg_dists = dist_fn(model(vsents), model(neg_vbigrams))
                loss = loss_fn(pos_dists, neg_dists)
                pbar.set_description("{:5.4f}".format(loss.item()))
                loss.backward()
                optimizer.step()
                pbar.update(len(ix_sents_batch))
            if test_model and (epoch + 1) % test_freq == 0:
                pbar.write(
                    "{:5.4f}".format(
                        test(
                            ix_sents_dev,
                            sent_lengths_dev,
                            model,
                            word_vecs,
                            dist_fn,
                            batch_size,
                            device=device,
                        )
                    )
                )


def test(ix_sents, sent_lengths, model, word_vecs, dist_fn, batch_size, device=None):
    ix_sents = ix_sents.to(device)
    correct = 0
    with torch.no_grad():
        for i in range(0, ix_sents.size(0), batch_size):
            ix_sents_batch = ix_sents[i : i + batch_size].to(device)
            pos_bigram_ixs = gen_pos_bigram_ixs(ix_sents_batch, device=device)
            neg_bigram_ixs = gen_neg_bigram_ixs(ix_sents_batch, device=device)

            pos_vbigrams = wv.vecs[pos_bigram_ixs].to(device)
            neg_vbigrams = wv.vecs[neg_bigram_ixs].to(device)
            vsents = wv.vecs[ix_sents_batch].to(device)
            pos_dists = dist_fn(model(vsents), model(pos_vbigrams))
            neg_dists = dist_fn(model(vsents), model(neg_vbigrams))

            correct += torch.sum(pos_dists < neg_dists).item()
        accuracy = correct / ix_sents.size(0)
    return accuracy


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
            * (in_bigram_vecs > 0)[:, :, : self.out_bigram_vec_dim].float()
        )
        if aggregate:
            # output: (batch_size, out_bigram_vec_dim)
            output = torch.sum(torch.tanh(out_bigram_vecs), dim=1)
        else:
            # output: (batch_size, max_sent_len - 1, out_bigram_vec_dim)
            output = torch.tanh(out_bigram_vecs)
        return output


def generate_bigram_nn_models():
    for i in range(2, 7):
        corpus_size = 10 ** i
        bigram_fn_name = "diff"
        out_bigram_dim = 300
        dist_fn_name = "cos_dist"
        loss_fn_name = "mrl"
        margin = 0.1
        lr = 0.1
        num_epochs = max(100000 // corpus_size, 1)
        batch_size = 300
        test_model = True
        test_freq = max(num_epochs // 10, 1)

        with open(PROCESSED / "train.{}.pkl".format(str(corpus_size)), "rb") as f:
            wiki_train = pickle.load(f)

        with open(PROCESSED / "valid.pkl", "rb") as f:
            wiki_valid = pickle.load(f)

        wiki_combined = wiki_train + wiki_valid
        corpus = Corpus("wiki", wiki_combined, wv)

        model = Net(wv.vecs.size(1), BigramEncoder(bigram_fn_name), out_bigram_dim)

        model.to(device)
        dist_fn = DistanceFunction(dist_fn_name)
        loss_fn = LossFunction(loss_fn_name, margin=margin)

        print("Traninig on Wikipedia corpus of size {}".format(corpus_size))

        train(
            corpus.ix_sents[: -len(wiki_valid)],
            corpus.sent_lengths[: -len(wiki_valid)],
            corpus.ix_sents[-len(wiki_valid) :],
            corpus.sent_lengths[-len(wiki_valid) :],
            model,
            wv.vecs,
            dist_fn,
            loss_fn,
            lr,
            num_epochs,
            batch_size,
            test_model,
            test_freq,
            device=device,
        )

        torch.save(
            model.state_dict(),
            MODELS / "bigram_nn_wiki_train_{}.pth".format(str(corpus_size)),
        )
        print("Done.")
        print("Start testing...")
        test_accuracy = test(
            corpus.ix_sents[-len(wiki_valid) :],
            corpus.sent_lengths[-len(wiki_valid) :],
            model,
            wv.vecs.to(device),
            dist_fn,
            batch_size,
            device,
        )
        print("Done.")
        print("test accuracy: {}".format(test_accuracy))


print("Loading word vectors...")
word2index, word_vecs = process_word_vecs(FAST_TEXT)
# Note that the word embeddings are normalized.
wv = WV(F.normalize(word_vecs), word2index)
print("Done.")
generate_bigram_nn_models()
