import argparse
import itertools
import json
import pickle
from pathlib import Path

import git
import humanize
from nltk import sent_tokenize, word_tokenize
from tqdm.auto import tqdm


def wiki_text_generator(p: Path, n_sents):
    with tqdm(total=n_sents) as pbar:
        with open(p, "r") as f:
            i = 0
            for line in f:
                if i < n_sents:
                    text = json.loads(line)["text"]
                    if text:
                        sents = [word_tokenize(sent) for sent in sent_tokenize(text)]
                        k = n_sents - i if i + len(sents) > n_sents else len(sents)
                        yield sents[:k]
                        i += len(sents)
                        pbar.update(k)
                else:
                    return


if __name__ == "__main__":

    repo = git.Repo(Path(__file__).absolute(), search_parent_directories=True)
    ROOT = Path(repo.working_tree_dir)
    default_wiki_path = (
        ROOT / "data" / "raw" / "wiki.json"
    )  # Path to the Wikipedia Corpus
    default_outdir = ROOT / "data" / "processed"
    default_n_sents = 1000000

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wiki_path",
        default=default_wiki_path,
        type=str,
        help="the path to the wikipedia data",
    )
    parser.add_argument(
        "outdir",
        default=default_outdir,
        type=str,
        help="the path to the output directory",
    )
    parser.add_argument(
        "n_sents",
        default=default_n_sents,
        type=int,
        help="the number of sentences to be extracted",
    )

    args = parser.parse_args()

    wiki_corpus = list(
        itertools.chain.from_iterable(
            wiki_text_generator(p=args.wiki_path, n_sents=args.n_sents)
        )
    )

    out_file_name = "wiki_{}.pkl".format(
        "_".join(humanize.intword(args.n_sents, format="%.0f").split())
    )
    with open(Path(args.outdir) / out_file_name, "wb") as f:
        pickle.dump(wiki_corpus, f, protocol=-1)
