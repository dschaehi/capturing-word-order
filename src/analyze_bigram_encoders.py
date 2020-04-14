from pathlib import Path

import git
import matplotlib as mpl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical
from tqdm.auto import trange

from misc import BigramEncoder
from models import Net


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)
MODELS = ROOT / "models"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze_bigram_encoder(
    bigram_encoder_name, wv, ix_sents, batch_size, smoke_test=False, device=None
):
    ix_sents = ix_sents.to(device)
    if smoke_test:
        ix_sents = ix_sents[:1000]
    if bigram_encoder_name == "T":
        model = Net(wv.vecs.size(1), BigramEncoder("diff"), 300)
        model.load_state_dict(
            torch.load(MODELS / "bigram_nn_wiki_train_{}.pth".format(str(1000000))),
        )
        model.to(device)

        def bigram_encoder(vec_sents):
            with torch.no_grad():
                result = model(vec_sents, aggregate=False)
            return result

    else:
        bigram_encoder = BigramEncoder(bigram_encoder_name)
    cos = nn.CosineSimilarity(dim=2)
    result_comparison = torch.tensor([], device=device).bool()
    result_pos_dist = torch.tensor([], device=device).float()
    result_neg_dist = torch.tensor([], device=device).float()
    for i in trange(0, len(ix_sents), batch_size):
        vsents = wv.vecs[ix_sents[i : i + batch_size]].to(device)
        bigram_vsents = bigram_encoder(vsents)
        bigram_sentvecs = bigram_vsents.sum(1, keepdim=True)
        pos_bigram_ixs = gen_pos_bigram_ixs(ix_sents[i : i + batch_size], device=device)
        neg_bigram_ixs = gen_neg_bigram_ixs(ix_sents[i : i + batch_size], device=device)
        pos_bigram_vecs = bigram_encoder(wv.vecs[pos_bigram_ixs].to(device))
        neg_bigram_vecs = bigram_encoder(wv.vecs[neg_bigram_ixs].to(device))
        comparison = cos(bigram_vsents, bigram_sentvecs).min(dim=1, keepdim=True).values
        pos_dist = cos(pos_bigram_vecs, bigram_sentvecs)
        neg_dist = cos(neg_bigram_vecs, bigram_sentvecs)
        result_comparison = torch.cat(
            (result_comparison, comparison > neg_dist,), dim=0
        )
        result_pos_dist = torch.cat((result_pos_dist, pos_dist), dim=0)
        result_neg_dist = torch.cat((result_neg_dist, neg_dist), dim=0)
    return result_comparison, result_pos_dist, result_neg_dist


def plot_result(
    bigram_encoder_name,
    wv,
    ix_sents,
    batch_size,
    outdir=ROOT / "paper/img",
    seed=0,
    add_legend=True,
):
    bigram_encoder_name_latex = {
        "T": r"f_{T}",
        "sign": r"f_{\infty}",
        "tanh": r"f_{1}",
        "tanh10": r"f_{10}",
        "mult": r"f_\odot",
    }[bigram_encoder_name]
    torch.manual_seed(seed)
    result_comparison, result_pos_dist, result_neg_dist = analyze_bigram_encoder(
        bigram_encoder_name, wv, ix_sents, batch_size, device=device
    )

    print(
        "Accuracy of {}: {:3.2f}".format(
            bigram_encoder_name_latex, result_comparison.float().mean()
        )
    )

    mpl.rcParams["text.latex.preamble"] = r"\usepackage{times}"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=24)
    fig, ax = plt.subplots()
    ax.set_position([0.22, 0.22, 0.7, 0.7])

    ax.set_ylim(top=50000)
    ax.set_xlim(left=-1, right=1)
    params = {"alpha": 0.7, "bins": 200}
    ax.hist(
        result_pos_dist.cpu().numpy(),
        label=(r"$(w, w') \in B(S)$" if add_legend else None),
        **params,
        color="C2"
    )
    ax.hist(
        result_neg_dist.cpu().numpy(),
        label=(r"$(w, w') \notin B(S)$" if add_legend else None),
        **params,
        color="C3"
    )
    ax.set(
        xlabel=r"$\cos(" + bigram_encoder_name_latex + r"(w, w'), \mathbf{S^2})$",
        ylabel=r"\# sentences",
    )
    if add_legend:
        ax.legend()
    else:
        ax.legend(frameon=False)
    plt.savefig(Path(outdir) / "bigram_encoder_{}.pdf".format(bigram_encoder_name))
    plt.show()


def gen_pos_bigram_ixs(ix_sents, device=None):
    batch_size = ix_sents.shape[0]
    sent_lengths = ix_sents.sign().sum(dim=1)
    ixs = (
        (torch.rand((batch_size,), device=device) * (sent_lengths.float() - 1))
        .floor()
        .long()
        .view(-1, 1)
    )
    return ix_sents[
        torch.arange(batch_size).view(-1, 1), torch.cat((ixs, (ixs + 1)), dim=1),
    ]


def gen_neg_bigram_ixs(ix_sents, device=None):
    batch_size, chunk_size = ix_sents.shape
    sent_lengths = ix_sents.sign().sum(dim=1)
    # ``distr_ixs1`` determines from where to sample the first index.
    # All indices but the last one allows for combining with ``sent_lengths`` - 1
    # different second indices.
    distr_ixs1 = (sent_lengths.view(-1, 1) - 1) * ix_sents.sign().to(device)
    # The last index allows for combining with ``sent_lengths`` different second indices
    distr_ixs1[torch.arange(batch_size), sent_lengths - 1] = sent_lengths
    ixs1 = Categorical(distr_ixs1.float()).sample().view(-1, 1)
    # ``distr_ixs2`` determines from where to sample the second index.
    # The boundary case is resolved by initializing  ``distr_ixs2`` with
    # one extra column and then removing it later.
    distr_ixs2 = torch.zeros((batch_size, chunk_size + 1), device=device)
    distr_ixs2[:, :-1] = ix_sents.sign()
    # The indices that lead to positive bigrams are avoided by setting
    # the corresponding value in ``distribution`` to zero.
    distr_ixs2[torch.arange(batch_size).view(-1, 1), ixs1 + 1] = 0
    distr_ixs2 = distr_ixs2[:, :-1]
    ixs2 = Categorical(distr_ixs2).sample().view(-1, 1)
    return ix_sents[
        torch.arange(batch_size).view(-1, 1), torch.cat((ixs1, ixs2), dim=1)
    ]
