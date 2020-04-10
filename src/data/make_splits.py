import pickle
from pathlib import Path

import git
import numpy as np
from ..misc import load_wiki


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)

PROCESSED = ROOT / "data" / "processed"

print("Loading the Wikipedia corpus of 2 million sentences...")
sentences = load_wiki(n_sents=2000000, max_len=25)
print("Done")

np.random.seed(0)
np.random.shuffle(sentences)

valid_size = 10000
test_size = 10000


print(
    "Splitting the data int training, validation and test sets with varying the training set size (from 100 to 1 million)."
)
with open(PROCESSED / "test.pkl", "wb") as f:
    pickle.dump(sentences[-test_size:], f, protocol=-1)

with open(PROCESSED / "valid.pkl", "wb") as f:
    pickle.dump(sentences[-(test_size + valid_size) : -test_size], f, protocol=-1)

for n_sents in [10 ** k for k in [2, 3, 4, 5, 6]]:
    train = sentences[:n_sents]
    with open(PROCESSED / ("train." + str(n_sents) + ".pkl"), "wb") as f:
        pickle.dump(train, f, protocol=-1)
print("Done.")
