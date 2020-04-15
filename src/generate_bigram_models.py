import pickle
from pathlib import Path

import git
import torch
import torch.nn.functional as F
from ray import tune


from analyze_bigram_encoders import gen_neg_bigram_ixs, gen_pos_bigram_ixs
from misc import (
    WV,
    BigramEncoder,
    Corpus,
    DistanceFunction,
    LossFunction,
    process_word_vecs,
)
from models import Net


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)
FAST_TEXT = ROOT / "data/raw/crawl-300d-2M.vec"
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "models"

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def train(
    wv,
    ix_sents,
    sent_lengths,
    ix_sents_dev,
    sent_lengths_dev,
    model,
    word_vecs,
    dist_fn,
    loss_fn,
    optimizer,
    lr,
    num_epochs,
    batch_size,
    iteration,
    test_model,
    test_freq,
    device=None,
):
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
        loss.backward()
        optimizer.step()
    if test_model and (iteration + 1) % test_freq == 0:
        accuracy = test(
            wv,
            ix_sents_dev,
            sent_lengths_dev,
            model,
            word_vecs,
            dist_fn,
            batch_size,
            device=device,
        )
        return {"accuracy": accuracy}
    return {}


def test(
    wv, ix_sents, sent_lengths, model, word_vecs, dist_fn, batch_size, device=None
):
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


class TrainBigramNN(tune.Trainable):
    def _setup(self, config):
        print("Loading word vectors...")
        word2index, word_vecs = process_word_vecs(FAST_TEXT)
        # Note that the word embeddings are normalized.
        self.wv = WV(F.normalize(word_vecs), word2index)
        # wv = WV(word_vecs, word2index)
        print("Done.")
        self.corpus_size = config["corpus_size"]
        bigram_fn_name = "diff"
        out_bigram_dim = 300
        dist_fn_name = "cos_dist"
        loss_fn_name = "mrl"
        margin = config["margin"]
        self.lr = config["lr"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.test_model = True
        self.test_freq = config["test_freq"]
        with open(PROCESSED / "train.{}.pkl".format(str(self.corpus_size)), "rb") as f:
            wiki_train = pickle.load(f)
        with open(PROCESSED / "valid.pkl", "rb") as f:
            wiki_valid = pickle.load(f)
        wiki_combined = wiki_train + wiki_valid
        self.corpus = Corpus("wiki", wiki_combined, self.wv, filter_stopwords=True)
        self.model = Net(
            self.wv.vecs.size(1), BigramEncoder(bigram_fn_name), out_bigram_dim
        )
        self.model.to(device)
        self.dist_fn = DistanceFunction(dist_fn_name)
        self.loss_fn = LossFunction(loss_fn_name, margin=margin)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(config["seed"])
        print("Traninig on Wikipedia corpus of size {}".format(self.corpus_size))

    def _train(self):
        result = train(
            self.wv,
            self.corpus.ix_sents[: self.corpus_size],
            self.corpus.sent_lengths[: self.corpus_size],
            self.corpus.ix_sents[self.corpus_size :],
            self.corpus.sent_lengths[self.corpus_size :],
            self.model,
            self.wv.vecs,
            self.dist_fn,
            self.loss_fn,
            self.optimizer,
            self.lr,
            self.num_epochs,
            self.batch_size,
            self._iteration,
            self.test_model,
            self.test_freq,
            self.device,
        )
        return result

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = str(Path(tmp_checkpoint_dir) / "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = str(Path(tmp_checkpoint_dir) / "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
