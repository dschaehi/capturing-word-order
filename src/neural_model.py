from collections import deque, namedtuple
from itertools import product
from pathlib import Path
from typing import Tuple

import git
import numpy as np
import torch
from multiset import Multiset

from .misc import BigramEncoder
from .models import Net
from .sentence_reconstruction import ngram_sent_vecs


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)

PartialSent = namedtuple("PartialSent", ["multiset", "sent"])
PartialList = namedtuple("PartialList", ["multiset", "trigrams"])

DATA = ROOT / "data"
MODELS = ROOT / "models"

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def gen_bvs_i2b(
    unigram_sents,
    bigram_vec_repr,
    word_vecs,
    word2index,
    markers=True,
    start="▷",
    end="◁",
):
    for unigram_sent in unigram_sents:
        pairs = sorted(
            set(
                product(
                    set(unigram_sent) | set([start, end])
                    if markers
                    else set(unigram_sent),
                    repeat=2,
                )
            )
        )
        # generate vectors that correspond to the candidate bigrams
        bigram_vecs = np.vstack(
            ngram_sent_vecs(
                [[pair] for pair in pairs], bigram_vec_repr, word_vecs, word2index, 2
            )
        )
        index2bigram = dict(enumerate(pairs))
        yield (bigram_vecs, index2bigram)
    return


class BigramNN:
    def __init__(self, bigram_fn_name):
        self.model = Net(300, BigramEncoder(bigram_fn_name), 300)
        # FIXME: Change the name of the pytorch model. 
        self.model.load_state_dict(
            torch.load(MODELS / ("bigram_nn_wiki_train_100.pth"))
        )
        self.model.to(device)

    def __call__(self, ngram_vecs):
        vec_sents = ngram_vecs.to(device)
        with torch.no_grad():
            output = self.model.forward(vec_sents, aggregate=False)
        return output.cpu().numpy()


def adjacent(partial_sent: namedtuple, bigram: Tuple, bigram_sent: Multiset):
    if partial_sent.multiset[bigram] < bigram_sent[bigram]:
        if partial_sent.sent[-1] == bigram[0]:
            return PartialSent(
                multiset=partial_sent.multiset + Multiset([bigram]),
                sent=partial_sent.sent + [bigram[1]],
            )
    return False


def get_candidate_sents(bigram_sent: Multiset, start_token="▷", end_token="◁"):
    def gen_candidates(start_partial_sents, bigram_sent):
        # contains lists of enumerated ordered bigrams that could
        # potentially form sentences.
        candidates = []
        max_len = 0  # length of the longest candidate
        for start_partial_sent in start_partial_sents:
            Q = deque([start_partial_sent])
            while len(Q) > 0:
                if len(Q) > 1000:
                    return []
                partial_sent = Q.popleft()
                if len(partial_sent.sent) > max_len:
                    max_len = len(partial_sent.sent)
                    candidates = [partial_sent]
                if len(partial_sent.sent) == max_len:
                    candidates.append(partial_sent)
                for bigram in bigram_sent:
                    partial_sent_new = adjacent(partial_sent, bigram, bigram_sent)
                    if partial_sent_new and (partial_sent_new not in Q):
                        Q.append(partial_sent_new)
                # list(
                #     map(
                #         print,
                #         [
                #             " ".join(q.sent) # + "\n"
                #             # + "\n".join(map(str, q.multiset.items()))
                #             for q in Q
                #         ],
                #     )
                # )
                # input()
        return candidates

    start_partial_sents = [
        PartialSent(multiset=Multiset([bigram]), sent=[bigram[1]])
        for bigram in bigram_sent
        if start_token == bigram[0]
    ]
    candidates = gen_candidates(start_partial_sents, bigram_sent)
    candidate_sents = []
    if candidates:
        for candidate in candidates:
            bigram_sent_rest = bigram_sent - candidate.multiset
            bigram_sent_rest_reversed = Multiset(
                [(bigram[1], bigram[0]) for bigram in bigram_sent_rest]
            )
            end_partial_sents = [
                PartialSent(multiset=Multiset([bigram]), sent=[bigram[1]])
                for bigram in bigram_sent_rest_reversed
                if end_token == bigram[0]
            ]
            if not end_partial_sents:
                candidate_sents.append(candidate.sent)
            else:
                candidates_reversed = gen_candidates(
                    end_partial_sents, bigram_sent_rest_reversed
                )
                candidate_sents.extend(
                    [
                        candidate.sent + candidate_reversed.sent[::-1]
                        for candidate_reversed in candidates_reversed
                    ]
                )
    else:
        bigram_sent_reversed = Multiset(
            [(bigram[1], bigram[0]) for bigram in bigram_sent]
        )
        end_partial_sents = [
            PartialSent(multiset=Multiset([bigram]), sent=[bigram[1]])
            for bigram in bigram_sent_reversed
            if end_token == bigram[0]
        ]
        if end_partial_sents:
            candidates_reversed = gen_candidates(
                end_partial_sents, bigram_sent_reversed
            )
            candidate_sents.extend(
                [
                    candidate_reversed.sent[::-1]
                    for candidate_reversed in candidates_reversed
                ]
            )
    # remove markers, and concatenate the words to form
    # sentences
    candidate_sents = list(
        set(
            tuple(
                [start_token]
                + [
                    word
                    for word in candidate_sent
                    if word not in {start_token, end_token}
                ]
                + [end_token]
            )
            for candidate_sent in candidate_sents
        )
    )
    return candidate_sents
