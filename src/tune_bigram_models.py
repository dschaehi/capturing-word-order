import argparse
import warnings
from pathlib import Path

import git
from git.exc import RepositoryDirtyError
from ray import tune

from generate_bigram_models import TrainBigramNN


warnings.filterwarnings("ignore")

# Force to commit changes
repo = git.Repo(".", search_parent_directories=True)
sha = repo.head.object.hexsha

current_path = Path(repo.working_dir).as_posix()
experiment_name = current_path.replace("/", ".")[1:] + "_" + sha[:8]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )

    args, _ = parser.parse_known_args()

    if not args.smoke_test and repo.is_dirty():
        raise RepositoryDirtyError(repo, "Have you forgotten to commit the changes?")

    config = {
        # A trick to log the SHA of the git HEAD.
        "SHA": tune.grid_search([sha]),
        "corpus_size": tune.grid_search([100000]),
        "margin": tune.grid_search([0.3]),
        "lr": tune.grid_search([0.1]),
        "batch_size": tune.grid_search([300]),
        "num_epochs": tune.grid_search([3]),
        "test_freq": 1,
    }

    analysis = tune.run(
        TrainBigramNN,
        name=experiment_name,
        config=config,
        num_samples=1 if args.smoke_test else 100,
        # trial_name_creator=trial_str_creator,
        resources_per_trial={"cpu": 8, "gpu": 1},
        stop={
            "training_iteration": 1
            if args.smoke_test
            else config["num_epochs"]
        },
        checkpoint_at_end=True,
        verbose=1,
    )
