import argparse
import itertools
import json
import pickle
from pathlib import Path

import git
import humanize
from nltk import sent_tokenize, word_tokenize
from tqdm.auto import tqdm


def wiki_text_generator(p: Path, n_sents, max_len):
    with tqdm(total=n_sents) as pbar:
        with open(p, "r") as f:
            i = 0
            for line in f:
                if i < n_sents:
                    text = json.loads(line)["text"]
                    if text:
                        # sents = [word_tokenize(sent) for sent in sent_tokenize(text)]
                        sents = [
                            sent
                            for sent in (
                                word_tokenize(sent) for sent in sent_tokenize(text)
                            )
                            if 3 <= len(sent) <= (max_len if max_len > 0 else 999999)
                        ]
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
        ROOT / "data" / "interim" / "wiki.json"
    )  # Path to the Wikipedia Corpus
    default_outdir = ROOT / "data" / "processed"
    default_n_sents = 1000000
    default_max_len = -1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki_path",
        nargs="?",
        default=default_wiki_path,
        type=str,
        help="the path to the wikipedia data (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir",
        nargs="?",
        default=default_outdir,
        type=str,
        help="the path to the output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--n_sents",
        nargs="?",
        default=default_n_sents,
        type=int,
        help="the number of sentences to be extracted (default: %(default)s)",
    )

    parser.add_argument(
        "--max_len",
        nargs="?",
        default=default_max_len,
        type=int,
        help="the maximal length of a sentence, where -1 means no limit (default: %(default)s)",
    )

    args = parser.parse_args()

    wiki_corpus = list(
        itertools.chain.from_iterable(
            wiki_text_generator(
                p=args.wiki_path, n_sents=args.n_sents, max_len=args.max_len
            )
        )
    )

    out_filename = "wiki_{}{}.pkl".format(
        "_".join(humanize.intword(args.n_sents, format="%.0f").split()),
        f"_maxlen_{args.max_len}" if args.max_len > 0 else "",
    )
    with open(Path(args.outdir) / out_filename, "wb") as f:
        pickle.dump(wiki_corpus, f, protocol=-1)
