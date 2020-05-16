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
    bigram_encoder_name,
    wv,
    ix_sents,
    batch_size,
    average_comparison=False,
    model_path=None,
    smoke_test=False,
    device=None,
):
    ix_sents = ix_sents.to(device)
    if smoke_test:
        ix_sents = ix_sents[:1000]
    if bigram_encoder_name == "T":
        model = Net(wv.vecs.size(1), BigramEncoder("diff"), 300)
        if model_path:
            model.load_state_dict(torch.load(model_path))
        else:
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
        pos_dist = cos(pos_bigram_vecs, bigram_sentvecs)
        neg_dist = cos(neg_bigram_vecs, bigram_sentvecs)
        if average_comparison:
            comparison = pos_dist
        else:
            comparison = (
                cos(bigram_vsents, bigram_sentvecs).min(dim=1, keepdim=True).values
            )
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
    average_comparison=False,
    outdir=ROOT / "paper/img",
    model_path=None,
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
        bigram_encoder_name,
        wv,
        ix_sents,
        batch_size,
        average_comparison=average_comparison,
        model_path=model_path,
        device=device,
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
        result_pos_dist[result_pos_dist != 0].cpu().numpy(),
        label=(r"$(w, w') \in B(S)$" if add_legend else None),
        **params,
        color="C2"
    )
    ax.hist(
        result_neg_dist[result_neg_dist != 0].cpu().numpy(),
        label=(r"$(w, w') \notin B(S)$" if add_legend else None),
        **params,
        color="C3"
    )
    ax.set(
        xlabel=r"$\cos(" + bigram_encoder_name_latex + r"(\mathbf{w}, \mathbf{w'}), \mathbf{S^2})$",
        ylabel=r"\# sentences",
    )
    if add_legend:
        ax.legend()
    else:
        ax.legend(frameon=False)
    plt.savefig(Path(outdir) / "bigram_encoder_{}.pdf".format(bigram_encoder_name))
    plt.show()


def plot_uniformity(
    word_pair,
    wv,
    ix_sents,
    batch_size,
    outdir=ROOT / "paper/img",
    model_path=None,
    seed=0,
    add_legend=True,
):
    torch.manual_seed(seed)

    bigram_encoder1 = BigramEncoder("diff")

    model = Net(wv.vecs.size(1), BigramEncoder("diff"), 300)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(
            torch.load(MODELS / "bigram_nn_wiki_train_{}.pth".format(str(1000000))),
        )
    model.to(device)

    def bigram_encoder2(vec_sents):
        with torch.no_grad():
            result = model(vec_sents, aggregate=False)
        return result

    ix_sents = ix_sents.to(device)
    cos = nn.CosineSimilarity(dim=2)
    result1 = torch.tensor([], device=device).float()
    result2 = torch.tensor([], device=device).float()
    for i in trange(0, len(ix_sents), batch_size):
        if word_pair == "random":
            bigram_ixs1 = torch.randint(
                low=2, high=wv.vecs.shape[0], size=(batch_size, 2)
            )
            bigram_ixs2 = torch.randint(
                low=2, high=wv.vecs.shape[0], size=(batch_size, 2)
            )
        elif word_pair == "bigram":
            bigram_ixs1 = gen_pos_bigram_ixs(
                ix_sents[i : i + batch_size], device=device
            )
            bigram_ixs2 = gen_pos_bigram_ixs(
                ix_sents[i : i + batch_size], device=device
            )
        else:
            raise NotImplementedError
        ixs_different = bigram_ixs1[:, 0] != bigram_ixs2[:, 0]
        bigram_ixs1 = bigram_ixs1[ixs_different]
        bigram_ixs2 = bigram_ixs2[ixs_different]
        bigram_vecs1 = bigram_encoder1(wv.vecs[bigram_ixs1].to(device))
        bigram_vecs2 = bigram_encoder1(wv.vecs[bigram_ixs2].to(device))
        bigram_vecs3 = bigram_encoder2(wv.vecs[bigram_ixs1].to(device))
        bigram_vecs4 = bigram_encoder2(wv.vecs[bigram_ixs2].to(device))
        dist1 = cos(bigram_vecs1, bigram_vecs2)
        dist2 = cos(bigram_vecs3, bigram_vecs4)
        result1 = torch.cat((result1, dist1), dim=0)
        result2 = torch.cat((result2, dist2), dim=0)

    mpl.rcParams["text.latex.preamble"] = r"\usepackage{times}"
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=24)
    fig, ax = plt.subplots()
    ax.set_position([0.22, 0.22, 0.7, 0.7])

    ax.set_ylim(top=40000)
    ax.set_xlim(left=-1, right=1)
    params = {"alpha": 0.7, "bins": 200}
    ax.hist(
        result1[result1 != 0].cpu().numpy(),
        label=(r"$f_\text{diff}$" if add_legend else None),
        **params,
        color="C1"
    )
    params = {"alpha": 0.7, "bins": 200}
    ax.hist(
        result2[result2 != 0].cpu().numpy(),
        label=(r"$f_{T}$" if add_legend else None),
        **params,
        color="C0"
    )
    ax.set(
        xlabel=r"$\cos(f(\mathbf{w_1}, \mathbf{w_2}), f(\mathbf{w_3}, \mathbf{w_4}))$",
        ylabel=(r"\# word pairs" if word_pair == "random" else r"\# bigrams"),
    )
    if add_legend:
        ax.legend()
    else:
        ax.legend(frameon=False)
    plt.savefig(Path(outdir) / "bigram_uniformity_{}.pdf".format(word_pair))
    plt.show()


def plot_bigram_norm(
    wv,
    ix_sents,
    batch_size,
    outdir=ROOT / "paper/img",
    model_path=None,
    seed=0,
    add_legend=True,
):
    torch.manual_seed(seed)

    model = Net(wv.vecs.size(1), BigramEncoder("diff"), 300)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    def bigram_encoder(vec_sents):
        with torch.no_grad():
            result = model(vec_sents, aggregate=False)
        return result

    ix_sents = ix_sents.to(device)
    result1 = torch.tensor([], device=device).float()
    # result2 = torch.tensor([], device=device).float()
    for i in trange(0, len(ix_sents), batch_size):
        bigram_ixs1 = gen_pos_bigram_ixs(ix_sents[i : i + batch_size], device=device)
        # bigram_ixs2 = gen_pos_bigram_ixs(ix_sents[i : i + batch_size], device=device)
        # bigram_ixs3 = gen_pos_bigram_ixs(ix_sents[i : i + batch_size], device=device)
        bigram_vecs1 = bigram_encoder(wv.vecs[bigram_ixs1].to(device))
        # bigram_vecs2 = bigram_encoder(wv.vecs[bigram_ixs2].to(device))
        # bigram_vecs3 = bigram_encoder(wv.vecs[bigram_ixs3].to(device))
        norm = bigram_vecs1.norm(dim=-1)
        # prod = (bigram_vecs2 * bigram_vecs3).sum(dim=-1)
        result1 = torch.cat((result1, norm), dim=0)
        # result2 = torch.cat((result2, prod), dim=0)

    mpl.rcParams["text.latex.preamble"] = r"\usepackage{times}"
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=24)
    fig, ax = plt.subplots()
    ax.set_position([0.22, 0.22, 0.7, 0.7])

    # ax.set_ylim(top=40000)
    # ax.set_xlim(left=-1, right=1)
    params = {"alpha": 0.7, "bins": 200}
    ax.hist(
        result1[result1 != 0].cpu().numpy(),
        # label=(r"$\lVert f(\mathbf{w}, \mathbf{w'})\rVert$" if add_legend else None),
        **params,
        color="C8"
    )
    # params = {"alpha": 0.7, "bins": 200}
    # ax.hist(
    #     result2[result2 != 0].cpu().numpy(),
    #     label=(r"$\mathbf{b} \cdot \mathbf{b'}$" if add_legend else None),
    #     **params,
    #     color="C9"
    # )
    ax.set(
        xlabel=r"$\lVert f(\mathbf{w}, \mathbf{w'})\rVert_2$", ylabel=r"\# bigrams",
    )
    plt.xticks([16, 17, 17.3, 18])
    if add_legend:
        ax.legend()
    else:
        ax.legend(frameon=False)
    plt.savefig(Path(outdir) / "bigram_norm.pdf")
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


# def gen_neg_bigram_ixs(ix_sents, device=None):
#     batch_size, chunk_size = ix_sents.shape
#     sent_lengths = ix_sents.sign().sum(dim=1)
#     # ``distr_ixs1`` determines from where to sample the first index.
#     # All indices but the last one allows for combining with ``sent_lengths`` - 1
#     # different second indices.
#     distr_ixs1 = (sent_lengths.view(-1, 1) - 1) * ix_sents.sign().to(device)
#     # The last index allows for combining with ``sent_lengths`` different second indices
#     distr_ixs1[torch.arange(batch_size), sent_lengths - 1] = sent_lengths
#     ixs1 = Categorical(distr_ixs1.float()).sample().view(-1, 1)
#     # ``distr_ixs2`` determines from where to sample the second index.
#     # The boundary case is resolved by initializing  ``distr_ixs2`` with
#     # one extra column and then removing it later.
#     distr_ixs2 = torch.zeros((batch_size, chunk_size + 1), device=device)
#     distr_ixs2[:, :-1] = ix_sents.sign()
#     # The indices that lead to positive bigrams are avoided by setting
#     # the corresponding value in ``distribution`` to zero.
#     distr_ixs2[torch.arange(batch_size).view(-1, 1), ixs1 + 1] = 0
#     distr_ixs2 = distr_ixs2[:, :-1]
#     ixs2 = Categorical(distr_ixs2).sample().view(-1, 1)
#     return ix_sents[
#         torch.arange(batch_size).view(-1, 1), torch.cat((ixs1, ixs2), dim=1)
#     ]


def gen_neg_bigram_ixs(ix_sents, device=None):
    batch_size, chunk_size = ix_sents.shape
    sent_lengths = ix_sents.sign().sum(dim=1)
    # ``distr_ixs1`` determines from where to sample the first index.
    # All indices but the last one allows for combining with ``sent_lengths`` - 2
    # different second indices.
    distr_ixs1 = (sent_lengths.view(-1, 1) - 2) * ix_sents.sign().to(device)
    # The last index allows for combining with ``sent_lengths`` - 1 different second indices
    distr_ixs1[torch.arange(batch_size), sent_lengths - 1] = sent_lengths - 1
    ixs1 = Categorical(distr_ixs1.float()).sample().view(-1, 1)
    # ``distr_ixs2`` determines from where to sample the second index.
    # The boundary case is resolved by initializing  ``distr_ixs2`` with
    # one extra column and then removing it later.
    distr_ixs2 = torch.zeros((batch_size, chunk_size + 1), device=device)
    distr_ixs2[:, :-1] = ix_sents.sign()
    # The indices that lead to positive bigrams are avoided by setting
    # the corresponding value in ``distribution`` to zero.
    distr_ixs2[torch.arange(batch_size).view(-1, 1), ixs1 + 1] = 0
    # Also avoid bigram with same words.
    distr_ixs2[torch.arange(batch_size).view(-1, 1), ixs1] = 0
    distr_ixs2 = distr_ixs2[:, :-1]
    ixs2 = Categorical(distr_ixs2).sample().view(-1, 1)
    return ix_sents[
        torch.arange(batch_size).view(-1, 1), torch.cat((ixs1, ixs2), dim=1)
    ]
