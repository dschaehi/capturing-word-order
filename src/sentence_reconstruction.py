import warnings
from itertools import chain, combinations, product

import numpy as np
from multiset import Multiset
from sklearn.linear_model import orthogonal_mp
from toolz import compose

# from .solvers import basis_pursuit
from .solvers_cupy import basis_pursuit
from .solvers import omp as orthogonal_mp_arora


def accuracy(ngram_sents1, ngram_sents2):
    output = np.array(
        [
            Multiset(ngram_sent1) == Multiset(ngram_sent2)
            for (ngram_sent1, ngram_sent2) in zip(ngram_sents1, ngram_sents2)
        ]
    ).mean()
    return output


# def disc(ngram_vecs):
#     if ngram_vecs.ndim == 2:  # n = 1
#         _, d = ngram_vecs.shape
#         output = ngram_vecs
#     else:
#         _, n, d = ngram_vecs.shape
#         C = 1 / n
#         # TODO: Check the effect of dtype conversions
#         output = (
#             C * d ** ((n - 1) / 2) * ngram_vecs.astype(np.float64).prod(axis=1)
#         ).astype(np.float32)
#     return output


def ngram_sents(sents, n, markers=False, start="▷", end="◁"):
    """
    sents: List[List[str]]
    output: List[Multiset[str]] or List[Multiset[Tuple[str]]]
    """
    output = []
    for sent in sents:
        if n == 1:
            output.append(Multiset(sent if not markers else [start] + sent + [end]))
        else:
            output.append(
                Multiset(
                    zip(
                        *[
                            sent[j:] if not markers else ([start] + sent + [end])[j:]
                            for j in range(n)
                        ]
                    )
                )
            )
    return output


def sorted_ngrams(ngram_sents):
    output = [Multiset(map(compose(tuple, sorted), sent)) for sent in ngram_sents]
    return output


def ngram_sent_vecs(ngram_sents, ngram_vec_repr, word_vecs, word2index, n):
    """
    ngram_sents = List[Multiset[str]] or List[Multiset[Tuple[str]]]
    output: List[np.array]
    """
    output = []
    for ngram_sent in ngram_sents:
        if n == 1:
            indices = [word2index[unigram] for unigram in ngram_sent]
        else:
            indices = np.array(
                [[word2index[word] for word in ngram] for ngram in ngram_sent]
            )
        # ngram_vecs.shape = (len(ngram_sent), n, d)
        ngram_vecs = word_vecs[indices]
        ngram_sent_vec = ngram_vec_repr(ngram_vecs).sum(axis=0)
        output.append(ngram_sent_vec)
    return output


def bigram_sent2trigrams(bigram_sent, markers=False, start="▷", end="◁"):
    out = set(
        map(
            compose(tuple, sorted),
            (
                (*bigram, unigram)
                for (bigram, unigram) in chain(
                    *(
                        chain(product([bigram1], bigram2), product([bigram2], bigram1))
                        for (bigram1, bigram2) in combinations(bigram_sent, 2)
                    ),
                    chain.from_iterable(
                        ((bigram, start), (bigram, end)) for bigram in bigram_sent
                    )
                    if markers
                    else ()
                )
            ),
        )
    )
    return out


def gen_trigrams1(reconstructed_bigram_sents, markers=False, start="▷", end="◁"):
    out = (
        set(
            map(
                compose(tuple, sorted),
                (
                    (*bigram, unigram)
                    for (bigram, unigram) in chain(
                        *(
                            chain(
                                product([bigram1], bigram2), product([bigram2], bigram1)
                            )
                            for (bigram1, bigram2) in combinations(
                                reconstructed_bigram_sent, 2
                            )
                        ),
                        chain.from_iterable(
                            ((bigram, start), (bigram, end))
                            for bigram in reconstructed_bigram_sent
                        )
                        if markers
                        else ()
                    )
                ),
            )
        )
        for reconstructed_bigram_sent in reconstructed_bigram_sents
    )
    return out


def gen_trigrams2(reconstructed_unigram_sents, markers=True, start="▷", end="◁"):
    output = (
        set(
            map(
                compose(tuple, sorted),
                chain(
                    combinations(reconstructed_unigram_sent, 3),
                    (
                        (*bigram, unigram)
                        for (bigram, unigram) in product(
                            combinations(reconstructed_unigram_sent, 2), [start, end]
                        )
                    ),
                ),
            )
        )
        for reconstructed_unigram_sent in reconstructed_unigram_sents
    )
    return output


def tvs_i2t(trigrams, trigram_vec_repr, word_vecs, word2index):
    trigram_vecs = np.vstack(
        ngram_sent_vecs(
            [[trigram] for trigram in trigrams],
            trigram_vec_repr,
            word_vecs,
            word2index,
            3,
        )
    )
    index2trigram = dict(enumerate(trigrams))
    return trigram_vecs, index2trigram


def gen_tvs_i2t(trigrams_seq, trigram_vec_repr, word_vecs, word2index):
    for trigrams in trigrams_seq:
        # generate vectors that correspond to the candidate trigrams
        trigram_vecs = np.vstack(
            ngram_sent_vecs(
                [[trigram] for trigram in trigrams],
                trigram_vec_repr,
                word_vecs,
                word2index,
                3,
            )
        )
        index2trigram = dict(enumerate(trigrams))
        yield (trigram_vecs, index2trigram)
    return


def reconstruct(ngram_sent_vec, ngram_vecs, index2ngram, solver="omp", nnz=70):
    """
    ngram_sent_vecs: List[np.array]
    output: Multiset[str] or Multiset[Tuple(str)]
    """
    nnz = min(len(ngram_vecs), nnz)
    with warnings.catch_warnings():  # ignore RuntimeWarning from orthogonal_mp
        warnings.simplefilter("ignore")
        if solver == "omp":
            count_vec = orthogonal_mp(
                ngram_vecs.T, ngram_sent_vec, n_nonzero_coefs=nnz
            ).round()
        elif solver == "omp_arora":
            count_vec = orthogonal_mp_arora(
                ngram_vecs.T, ngram_sent_vec, n_nonzero_coefs=nnz
            ).round()
        elif solver == "bp":
            count_vec = basis_pursuit(ngram_vecs.T, ngram_sent_vec).round()
        else:
            raise NotImplementedError
        indices = np.argwhere(count_vec > 0).reshape(-1).astype(int)
        if type(count_vec) is not np.ndarray:
            count_vec = np.array([count_vec])
        counts = count_vec[indices].astype(int)
        output = Multiset(
            {
                index2ngram[int(index)]: int(count)
                for (index, count) in zip(indices, counts)
            }
        )
    # breakpoint()
    return output


def make_file_name(n, sents):
    return (
        "reconstructed_"
        + {1: "uni", 2: "bi", 3: "tri"}[n]
        + "grams_from_"
        + str(len(sents))
        + "_sents.pkl"
    )
