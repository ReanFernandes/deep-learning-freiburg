import neps
import torch
import random
import numpy as np
import re

from neps.plot.read_results import process_seed


def get_results(results) -> dict:
    """ Get the results in an easy to read dictionary

      Args:
        results: Path of the results folder.

      Returns:
          Dictionary containing the final losses, hyperparameter configurations, validation errors
          across epochs and incumbents
    """

    summary = neps.get_summary_dict(results)
    losses_and_configs = {"Losses": [], "Config": [], "Val_errors": [], "Incumbents": [], "Epochs": [],
                          "Val_loss": []}
    config_results, _ = neps.status(results)
    config_results = dict(sorted(config_results.items()))
    lcs = {str(i): [] for i in range(summary["num_evaluated_configs"])}
    for config, result in config_results.items():
        losses_and_configs["Losses"].append(result.result['loss'])
        losses_and_configs["Config"].append(result.config)
        losses_and_configs["Val_errors"].append(result.result["info_dict"]["val_errors"])
        losses_and_configs["Val_loss"].append(result.result["info_dict"]["val_loss"])
    for config_id, vals in lcs.items():
        if len(vals) != 0:
            losses_and_configs["Val_errors"].append(lcs[config_id])

    incumbent, costs, _ = process_seed(
        path=results,
        seed=None,
        key_to_extract="cost",
        consider_continuations=False,
        n_workers=1,
    )
    losses_and_configs["Incumbents"].extend(incumbent)
    losses_and_configs["Epochs"].extend([np.sum(costs[:i]) for i in range(1, len(costs)+1)])
    return losses_and_configs


def convert_identifier_to_str(identifier: str, terminals_to_nb201: dict) -> str:
    """
    Converts identifier to string representation.
    """
    start_indices = [m.start() for m in re.finditer("(OPS*)", identifier)]
    op_edge_list = []
    counter = 0
    for i, _ in enumerate(start_indices):
        start_idx = start_indices[i]
        end_idx = start_indices[i + 1] if i < len(start_indices) - 1 else -1
        substring = identifier[start_idx:end_idx]
        for k in terminals_to_nb201.keys():
            if k in substring:
                op_edge_list.append(f"{terminals_to_nb201[k]}~{counter}")
                break
        if i == 0 or i == 2:
            counter = 0
        else:
            counter += 1

    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)


def set_seed(seed=123):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
