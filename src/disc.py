import itertools
from collections import deque, namedtuple
from itertools import combinations
from typing import Tuple

import numpy as np
from multiset import Multiset
from pathlib import Path

from .sentence_reconstruction import ngram_sent_vecs

PartialSent = namedtuple("PartialSent", ["multiset", "sent"])
PartialList = namedtuple("PartialList", ["multiset", "trigrams"])

DATA = Path("../data/")
MODELS = Path("../models")


# def disc(ngram_vecs: np.array):
#     _, n, d = ngram_vecs.shape
#     C = 1 / n
#     output = C * d ** ((n - 1) / 2) * ngram_vecs.prod(axis=1).sum(axis=0, keepdims=True)
#     return output


def disc(ngram_vecs):
    if ngram_vecs.ndim == 2:  # n = 1
        _, d = ngram_vecs.shape
        output = ngram_vecs
    else:
        _, n, d = ngram_vecs.shape
        C = 1 / n
        # TODO: Check the effect of dtype conversions
        output = (
            C * d ** ((n - 1) / 2) * ngram_vecs.astype(np.float64).prod(axis=1)
        ).astype(np.float32)
    return output


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
                combinations(
                    set(unigram_sent) | set([start, end])
                    if markers
                    else set(unigram_sent),
                    2,
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


def adjacent(partial_sent: namedtuple, bigram: Tuple, bigram_sent: Multiset):
    if partial_sent.multiset[bigram] < bigram_sent[bigram]:
        if partial_sent.sent[-1] == bigram[0]:
            return PartialSent(
                multiset=partial_sent.multiset + Multiset([bigram]),
                sent=partial_sent.sent + [bigram[1]],
            )
        if partial_sent.sent[-1] == bigram[1]:
            return PartialSent(
                multiset=partial_sent.multiset + Multiset([bigram]),
                sent=partial_sent.sent + [bigram[0]],
            )
    return False


def adjacent_trigram(partial_list: namedtuple, trigram: Tuple, trigram_sent: Multiset):
    if partial_list.multiset[trigram] < trigram_sent[trigram]:
        if len(partial_list.trigrams) == 1:
            first_trigram = list(partial_list.trigrams)[0]
            common_words = Multiset(first_trigram) & Multiset(trigram)
            if len(common_words) == 2:
                first_word = list(Multiset(first_trigram) - common_words)[0]
                fourth_word = list(Multiset(trigram) - common_words)[0]
                return PartialList(
                    multiset=partial_list.multiset + Multiset([trigram]),
                    trigrams=[
                        (first_word, *list(common_words)),
                        (*list(common_words), fourth_word),
                    ],
                )
        elif len(partial_list.trigrams) == 2:
            first_trigram, second_trigram = partial_list.trigrams
            first_word = first_trigram[0]
            second_and_third_words = Multiset(first_trigram[1:])
            fourth_word = second_trigram[2]
            if fourth_word in trigram:
                third_words = list(
                    second_and_third_words
                    & (Multiset(trigram) - Multiset([fourth_word]))
                )
                if len(third_words) == 1:
                    third_word = third_words[0]
                    second_word = list(second_and_third_words - Multiset([third_word]))[
                        0
                    ]
                    fifth_word = list(
                        Multiset(trigram) - Multiset([third_word, fourth_word])
                    )[0]
                    return PartialList(
                        multiset=partial_list.multiset + Multiset([trigram]),
                        trigrams=[
                            (first_word, second_word, third_word),
                            (second_word, third_word, fourth_word),
                            (third_word, fourth_word, fifth_word),
                        ],
                    )
                elif len(third_words) == 2:
                    third_word1, third_word2 = third_words
                    second_word1 = list(
                        second_and_third_words - Multiset([third_word1])
                    )[0]
                    second_word2 = list(
                        second_and_third_words - Multiset([third_word2])
                    )[0]
                    fifth_word1 = list(
                        Multiset(trigram) - Multiset([third_word1, fourth_word])
                    )[0]
                    fifth_word2 = list(
                        Multiset(trigram) - Multiset([third_word2, fourth_word])
                    )[0]
                    partial_sent1 = PartialList(
                        multiset=partial_list.multiset + Multiset([trigram]),
                        trigrams=[
                            (first_word, second_word1, third_word1),
                            (second_word1, third_word1, fourth_word),
                            (third_word1, fourth_word, fifth_word1),
                        ],
                    )
                    partial_sent2 = PartialList(
                        multiset=partial_list.multiset + Multiset([trigram]),
                        trigrams=[
                            (first_word, second_word2, third_word2),
                            (second_word2, third_word2, fourth_word),
                            (third_word2, fourth_word, fifth_word2),
                        ],
                    )
                    return [partial_sent1, partial_sent2]
        else:
            last_trigram = partial_list.trigrams[-1]
            if len(Multiset(last_trigram[1:]) & Multiset(trigram)) >= 2:
                third_last_word, second_last_word = last_trigram[1:]
                last_word = list(Multiset(trigram) - Multiset(last_trigram[1:]))[0]
                return PartialList(
                    multiset=partial_list.multiset + Multiset([trigram]),
                    trigrams=partial_list.trigrams
                    + [(third_last_word, second_last_word, last_word)],
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
        PartialSent(
            multiset=Multiset([bigram]),
            sent=[bigram[1] if bigram[0] == start_token else bigram[0]],
        )
        for bigram in bigram_sent
        if start_token in bigram
    ]
    candidates = gen_candidates(start_partial_sents, bigram_sent)
    candidate_sents = []
    if candidates:
        for candidate in candidates:
            bigram_sent_rest = bigram_sent - candidate.multiset
            end_partial_sents = [
                PartialSent(
                    multiset=Multiset([bigram]),
                    sent=[bigram[1] if bigram[0] == end_token else bigram[0]],
                )
                for bigram in bigram_sent_rest
                if end_token in bigram
            ]
            if not end_partial_sents:
                candidate_sents.append(candidate.sent)
            else:
                candidates_reversed = gen_candidates(
                    end_partial_sents, bigram_sent_rest
                )
                candidate_sents.extend(
                    [
                        candidate.sent + candidate_reversed.sent[::-1]
                        for candidate_reversed in candidates_reversed
                    ]
                )
    else:
        end_partial_sents = [
            PartialSent(
                multiset=Multiset([bigram]),
                sent=[bigram[1] if bigram[0] == end_token else bigram[0]],
            )
            for bigram in bigram_sent
            if end_token in bigram
        ]
        if end_partial_sents:
            candidates_reversed = gen_candidates(end_partial_sents, bigram_sent)
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


def get_candidate_trigrams(candidate_sents):
    candidate_trigrams = sorted(
        set(
            itertools.chain.from_iterable(
                (
                    (candidate_sent[i], candidate_sent[i + 1], candidate_sent[i + 2])
                    for i in range(len(candidate_sent) - 2)
                )
                for candidate_sent in candidate_sents
            )
        )
    )
    return candidate_trigrams


def get_candidate_sents_trigrams(
    trigram_sent: Multiset, start_token="▷", end_token="◁"
):
    def gen_candidates(start_partial_lists, trigram_sent):
        # contains lists of PartialList objects that could
        # potentially form sentences.
        candidates = []
        max_len = 0  # length of the longest candidate
        for start_partial_list in start_partial_lists:
            Q = deque([start_partial_list])
            # with tqdm(total=len(trigram_sent)) as pbar:
            while len(Q) > 0:
                if len(Q) > 1000:
                    return []
                partial_list = Q.popleft()
                if len(partial_list.trigrams) > max_len:
                    # pbar.update(len(partial_list.trigrams) - max_len)
                    max_len = len(partial_list.trigrams)
                    candidates = [partial_list]
                if len(partial_list.trigrams) == max_len:
                    candidates.append(partial_list)
                for trigram in trigram_sent:
                    result = adjacent_trigram(partial_list, trigram, trigram_sent)
                    if type(result) is list:
                        for partial_list_new in result:
                            if partial_list_new not in Q:
                                Q.append(partial_list_new)
                                # print(Q)
                                # input()
                    elif type(result).__name__ == "PartialList" and (result not in Q):
                        Q.append(result)
                        # print(Q)
                        # input()
        return candidates

    start_partial_lists = [
        PartialList(multiset=Multiset([trigram]), trigrams=[trigram])
        for trigram in trigram_sent
        if start_token in trigram
    ]
    # A list of PartialList objects
    candidates = gen_candidates(start_partial_lists, trigram_sent)
    # contains a sentences. May include start or end tokens
    candidate_sents = []

    def gen_sent(trigrams):
        if trigrams:
            return list(trigrams[0]) + [trigram[2] for trigram in trigrams[1:]]
        else:
            return []

    if candidates:
        for candidate in candidates:
            partial_sent = gen_sent(candidate.trigrams)
            trigram_sent_rest = trigram_sent - candidate.multiset
            end_partial_lists = [
                PartialList(multiset=Multiset([trigram]), trigrams=[trigram])
                for trigram in trigram_sent_rest
                if end_token in trigram
            ]
            if not end_partial_lists:
                candidate_sents.append(partial_sent)
            else:
                candidates_reversed = gen_candidates(
                    end_partial_lists, trigram_sent_rest
                )
                candidate_sents.extend(
                    [
                        partial_sent
                        + gen_sent(
                            [
                                trigram[::-1]
                                for trigram in candidate_reversed.trigrams[::-1]
                            ]
                        )
                        if partial_sent[-1] != candidate_reversed.trigrams[-1][-1]
                        else partial_sent
                        + gen_sent(
                            [
                                trigram[::-1]
                                for trigram in candidate_reversed.trigrams[::-1][1:]
                            ]
                        )
                        for candidate_reversed in candidates_reversed
                    ]
                )
    else:
        end_partial_lists = [
            PartialList(multiset=Multiset([trigram]), trigrams=[trigram])
            for trigram in trigram_sent
            if end_token in trigram
        ]
        if end_partial_lists:
            candidates_reversed = gen_candidates(end_partial_lists, trigram_sent)
            candidate_sents.extend(
                [
                    gen_sent(
                        [trigram[::-1] for trigram in candidate_reversed.trigrams[::-1]]
                    )
                    for candidate_reversed in candidates_reversed
                ]
            )
    # remove markers, and concatenate the words to form
    # sentences
    candidate_sents = list(
        set(
            tuple(
                word for word in candidate_sent if word not in {start_token, end_token}
            )
            for candidate_sent in candidate_sents
        )
    )
    return candidate_sents


#########
# Tests #
#########

## adjacent_trigram
"""
>>> trigram_sent = Multiset(
    {
        ("▷", "2", "1"): 1,
        ("1", "2", "3"): 1,
        ("3", "2", "1"): 1,
        ("3", "1", "5"): 1,
        ("1", "5", "6"): 1,
        ("5", "6", "7"): 1,
        ("6", "7", "8"): 1,
        ("7", "8", "9"): 1,
        ("8", "9", "◁"): 1,
    }
)
>>> partial_list1 = PartialList(multiset=Multiset([("▷", "2", "1")]), trigrams=[("▷", "2", "1")])
>>> print(partial_list1)
PartialList(multiset=Multiset({('▷', '2', '1'): 1}), trigrams=[('▷', '2', '1')])
>>> partial_list2 = adjacent_trigram(partial_list1, ("1", "2", "3"), trigram_sent)
>>> print(partial_list2)
PartialList(multiset=Multiset({('▷', '2', '1'): 1, ('1', '2', '3'): 1}), trigrams=[('▷', '2', '1'), ('2', '1', '3')])
>>> partial_list3 = adjacent_trigram(partial_list2, ("3", "2", "1"), trigram_sent)
>>> print(partial_list3)
[PartialList(multiset=Multiset({('▷', '2', '1'): 1, ('1', '2', '3'): 1, ('3', '2', '1'): 1}), trigrams=[('▷', '1', '2'), ('1', '2', '3'), ('2', '3', '1')]),
 PartialList(multiset=Multiset({('▷', '2', '1'): 1, ('1', '2', '3'): 1, ('3', '2', '1'): 1}), trigrams=[('▷', '2', '1'), ('2', '1', '3'), ('1', '3', '2')])]
>>> partial_list4 = adjacent_trigram(partial_list3[0], ("3", "1", "5"), trigram_sent)
>>> print(partial_list4)
PartialList(multiset=Multiset({('▷', '2', '1'): 1, ('1', '2', '3'): 1, ('3', '2', '1'): 1, '3': 1, '1': 1, '5': 1}), trigrams=[('▷', '1', '2'), ('1', '2', '3'), ('2', '3', '1'), ('3', '1', '5')])
>>> partial_list4 = adjacent_trigram(partial_list3[1], ("3", "1", "5"), trigram_sent)
>>> print(partial_list4)
False
"""

## get_candidate_sents_trigrams
