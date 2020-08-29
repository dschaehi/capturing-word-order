# Test whether all possible positive and negative examples are generated correctly and their counts are distributed evely.

from collections import Counter
from itertools import product

import numpy as np
import torch
from src.analyze_bigram_encoders import gen_neg_bigram_ixs, gen_pos_bigram_ixs


def test_example_generators():
    n_rows = 1000000
    ix_sents = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]).repeat(n_rows // 2, 1)

    pos_bigram_ixs = gen_pos_bigram_ixs(ix_sents)
    neg_bigram_ixs = gen_neg_bigram_ixs(ix_sents)

    counter_pos_examples = Counter(tuple(pair) for pair in pos_bigram_ixs.numpy())
    counter_neg_examples = Counter(tuple(pair) for pair in neg_bigram_ixs.numpy())

    pos_examples_1 = {(1, 2), (2, 3)}
    neg_examples_1 = set(product([1, 2, 3], repeat=2)) - pos_examples_1
    pos_examples_2 = {(4, 5), (5, 6), (6, 7), (7, 8)}
    neg_examples_2 = set(product([4, 5, 6, 7, 8], repeat=2)) - pos_examples_2

    total_counts_pos_examples_1 = sum(
        [counter_pos_examples[pair] for pair in pos_examples_1]
    )
    total_counts_neg_examples_1 = sum(
        [counter_neg_examples[pair] for pair in neg_examples_1]
    )
    total_counts_pos_examples_2 = sum(
        [counter_pos_examples[pair] for pair in pos_examples_2]
    )
    total_counts_neg_examples_2 = sum(
        [counter_neg_examples[pair] for pair in neg_examples_2]
    )

    total_counts_pos_examples_1 == total_counts_neg_examples_1 == n_rows // 2
    total_counts_pos_examples_2 == total_counts_neg_examples_1 == n_rows // 2

    assert sum(counter_pos_examples.values()) == n_rows

    assert (
        np.std(
            [
                counter_pos_examples[pair] / total_counts_pos_examples_1
                for pair in pos_examples_1
            ]
        )
        < 0.01
    )

    assert (
        np.std(
            [
                    counter_pos_examples[pair] / total_counts_neg_examples_1
                for pair in neg_examples_1
            ]
        )
        < 0.01
    )

    assert (
        np.std(
            [
                counter_pos_examples[pair] / total_counts_pos_examples_2
                for pair in pos_examples_2
            ]
        )
        < 0.01
    )

    assert (
        np.std(
            [
                counter_neg_examples[pair] / total_counts_neg_examples_2
                for pair in pos_examples_2
            ]
        )
        < 0.01
    )
    return True
