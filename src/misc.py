import itertools
import pickle
from pathlib import Path

import git
import humanize
import numpy as np
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from tqdm.auto import tqdm


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)

PROCESSED = ROOT / "data" / "processed"


def load_wiki(wiki_dir=PROCESSED, n_sents=1000000):
    filename = "wiki_{}.pkl".format(
        "_".join(humanize.intword(n_sents, format="%.0f").split())
    )
    with open(PROCESSED / filename, "rb") as f:
        wiki_corpus = pickle.load(f)
    return wiki_corpus


class WV:
    def __init__(self, word_vecs, word_index, has_padding=True, padding_ix=0):
        d = word_vecs.size(1)
        self.vocab = set(word_index.keys())
        self.oovs = set([])
        self.has_padding = has_padding
        if self.has_padding:
            self.padding_ix = padding_ix
            if self.padding_ix == 0:
                if not word_vecs[0].equal(torch.zeros((d,))):
                    self.vecs = torch.cat((torch.zeros((1, d)), word_vecs), dim=0)
                    self.dict = {word: word_index[word] + 1 for word in self.vocab}
            elif self.padding_ix == -1:
                if not word_vecs[-1].equal(torch.zeros((d,))):
                    self.vecs = torch.cat((word_vecs, torch.zeros((1, d))), dim=0)
                    self.dict = word_index
        else:
            self.vecs = word_vecs
            self.dict = word_index

    def adjust(self, vocab):
        common_vocab = sorted(list(self.vocab & vocab))
        self.oovs = self.oovs | (vocab - self.vocab)
        if self.has_padding:
            if self.padding_ix == 0:
                ixs = [0] + [self.dict[word] for word in common_vocab]
                self.vecs = self.vecs[ixs]
                self.dict = dict(
                    zip(
                        common_vocab + list(self.oovs),
                        list(range(1, len(common_vocab) + 1)) + [0] * len(self.oovs),
                    )
                )
            elif self.padding_ix == -1:
                ixs = [self.dict[word] for word in common_vocab] + [-1]
                self.vecs = self.vecs[ixs]
                self.dict = dict(
                    zip(
                        common_vocab + list(self.oovs),
                        list(range(len(common_vocab))) + [-1] * len(self.oovs),
                    )
                )
        else:
            ixs = [self.dict[word] for word in common_vocab]
            self.vecs = self.vecs[ixs]
            self.dict = dict(zip(common_vocab, list(range(len(common_vocab)))))
        self.vocab = vocab

    def extend(self, vocab):
        oovs = vocab - self.vocab
        self.oovs.update(oovs)
        self.dict.update({word: 0 for word in oovs})

    def to_ix_sents(
        self,
        sents,
        cond_lower_case=False,
        filter_stopwords=False,
        return_sent_lengths=False,
        adjust=False,
        device=None,
    ):
        max_sent_len = max((len(sent) for sent in sents))
        if not self.has_padding:
            raise Exception("The instance is not initialized with a padding.")
        if filter_stopwords:
            st_words = set(stopwords.words("english"))
            sents = [
                [word for word in sent if word.lower() not in st_words]
                for sent in sents
            ]
        if cond_lower_case:
            sents = [
                [
                    word
                    if (word in self.dict) or (not word.lower() in self.dict)
                    else word.lower()
                    for word in sent
                ]
                for sent in sents
            ]
        sent_lengths = torch.tensor(
            [len(sent) for sent in sents], dtype=torch.int64, device=device
        )
        if adjust:
            vocab = set(itertools.chain.from_iterable(sents))
            self.adjust(vocab)
        ix_sents = torch.tensor(
            [
                [self.dict[word] for word in sent]
                + [self.padding_ix] * (max_sent_len - len(sent))
                for sent in sents
            ],
            dtype=torch.int64,
            device=device,
        )
        if return_sent_lengths:
            return ix_sents, sent_lengths
        else:
            return ix_sents

    @staticmethod
    def load(file_path, has_padding=True, padding_ix=0, device=None):
        with open(file_path, "rb") as f:
            word_index, word_vecs = pickle.load(f)
        return WV(
            torch.tensor(word_vecs, dtype=torch.float, device=device),
            word_index,
            has_padding,
            padding_ix,
        )


class Corpus:
    def __init__(
        self,
        name,
        sents,
        wv,
        cond_lower_case=False,
        filter_stopwords=False,
        device=None,
    ):
        """dataset: a list of sentences, which are list of words."""
        self.name = name
        self.ix_sents, self.sent_lengths = wv.to_ix_sents(
            sents,
            cond_lower_case=cond_lower_case,
            filter_stopwords=filter_stopwords,
            return_sent_lengths=True,
            adjust=True,
            device=device,
        )

    @staticmethod
    def load(
        name, file_path, wv, cond_lower_case=False, filter_stopwords=False, device=None
    ):
        with open(file_path, "rb") as f:
            sents = pickle.load(f)
        return Corpus(
            name,
            sents,
            wv,
            cond_lower_case=cond_lower_case,
            filter_stopwords=filter_stopwords,
            device=device,
        )


class BigramFunction:
    """
    "diff: Vector difference"
    """

    def __init__(self, fn_name):
        self.bigram_fn = getattr(self, fn_name)

    def diff(self, vsents):
        """
        vsents: (batch_size, max_sent_len, word_vec_dim)
        output: (batch_size, max_sent_len - 1, word_vec_dim)
        """
        # A word vector has a zero entry, only if it corresponds to a
        # padding. So set the bigram vectors to zero, if the minuend
        # of the diffrence is zero.
        return (vsents[:, 1:, :] - vsents[:, :-1, :]) * (vsents[:, 1:, :] > 0).float()

    def concat(self, vsents):
        """
        vsents: (batch_size, max_sent_len, word_vec_dim)
        output: (batch_size, max_sent_len - 1, 2 * word_vec_dim)
        """
        # A word vector has a zero entry, only if it corresponds to a
        # padding. So set the left part to zero, if the right
        # part is zero.
        return torch.cat(
            (vsents[:, :-1, :] * (vsents[:, 1:, :] > 0).float(), vsents[:, 1:, :]),
            dim=2,
        )

    def mult(self, vsents):
        """
        vsents: (batch_size, max_sent_len, word_vec_dim)
        output: (batch_size, max_sent_len - 1, word_vec_dim)
        """
        return vsents[:, 1:, :] * vsents[:, :-1, :]

    def __call__(self, vsents):
        return self.bigram_fn(vsents)


class OneSkipBigramFunction:
    """
    "diff: Vector difference"
    """

    def __init__(self, fn_name):
        self.one_skip_bigram_fn = getattr(self, fn_name)

    def diff(self, vsents):
        """
        vsents: (batch_size, max_sent_len, word_vec_dim)
        output: (batch_size, max_sent_len - 2, word_vec_dim)
        """
        # A word vector has a zero entry, only if it corresponds to a
        # padding. So set the one-skip-bigram vectors to zero, if the minuend
        # of the diffrence is zero.
        return (vsents[:, 2:, :] - vsents[:, :-2, :]) * (vsents[:, 2:, :] > 0).float()

    def __call__(self, vsents):
        return self.one_skip_bigram_fn(vsents)


class DistanceFunction:
    """
    "cos_dist": Cosine distance
    "euc_dist": Euclidean distance
    """

    def __init__(self, fn_name, p=2, eps=1e-8):
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=eps)
        self.euc_dist = nn.PairwiseDistance(p=p, eps=eps)
        self.distance_fn = getattr(self, fn_name)

    def cos_dist(self, v, w):
        return 1 - self.cos_sim(v, w)

    def __call__(self, v, w):
        return self.distance_fn(v, w)


class LossFunction:
    """
    "mrl": Margin ranking loss
    """

    def __init__(self, fn_name, margin=1.0, reduction="mean"):
        self.margin_ranking_loss = nn.MarginRankingLoss(
            margin=margin, reduction=reduction
        )
        self.loss_fn = getattr(self, fn_name)

    def mrl(self, pos_examples, neg_examples):
        return self.margin_ranking_loss(
            pos_examples, neg_examples, -torch.ones_like(pos_examples)
        )

    def __call__(self, pos_examples, neg_examples):
        return self.loss_fn(pos_examples, neg_examples)


def get_first_line(path_to_vec):
    with open(path_to_vec, "r", encoding="utf-8") as f:
        return next(f)


def get_word_vecs_size(path_to_vec):
    def get_wvec_dim(path_to_vec):
        with open(path_to_vec, "r", encoding="utf-8") as f:
            next(f)
            return len(np.fromstring(next(f).split(" ", 1)[1], sep=" "))

    if len(get_first_line(path_to_vec).split(" ")) == 2:
        num_lines, wvec_dim = [int(n) for n in get_first_line(path_to_vec).split(" ")]
    else:
        with open(path_to_vec, "r", encoding="utf-8") as f:
            wvec_dim = get_wvec_dim(path_to_vec)
            num_lines = 0
            for line in f:
                num_lines += 1
    return num_lines, wvec_dim


def process_word_vecs(path_to_vec, out_dir=PROCESSED):
    """Get word vectors from word2index (glove, word2vec, fasttext ..)"""
    path_to_pickled_vec = out_dir / (path_to_vec.stem + ".pkl")
    if path_to_pickled_vec.exists():
        with open(path_to_pickled_vec, "rb") as f:
            word2index, word_vecs = pickle.load(f)
    else:
        num_lines, wvec_dim = get_word_vecs_size(path_to_vec)
        word_vecs = np.empty([num_lines, wvec_dim])
        word2index = {}
        with open(path_to_vec, "r", encoding="utf-8") as f:
            if len(get_first_line(path_to_vec).split(" ")) == 2:
                next(f)
            i = 0
            with tqdm(total=num_lines) as pbar:
                for line in f:
                    word, vec = line.split(" ", 1)
                    word2index[word] = i
                    word_vecs[i] = np.fromstring(vec, sep=" ")
                    i += 1
                    pbar.update(1)
        word_vecs = torch.from_numpy(word_vecs).float()
        with open(path_to_pickled_vec, "wb") as f:
            pickle.dump([word2index, word_vecs], f, protocol=4)
    return word2index, word_vecs
