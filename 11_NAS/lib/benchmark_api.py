"""Benchmark APIs."""

import pickle
from enum import Enum
from typing import List, Union


class Metric(Enum):
    TRAIN_ACCURACY = "train_acc1es"
    VAL_ACCURACY = "eval_acc1es"

    TRAIN_LOSS = "train_losses"
    VAL_LOSS = "eval_losses"

    TRAIN_TIME = "train_time"
    PARAMS = "params"


class Dataset(Enum):
    CIFAR10 = "cifar10-valid"
    CIFAR100 = ""
    IMAGENET = ""


class BenchmarkAPI(object):
    """Abstract superclass for all BenchmarkAPI classes"""

    def __init__(self, dataset):
        self.dataset = dataset


class NB201BenchmarkAPI(BenchmarkAPI):
    """API class used to query metrics for architectures in the NB201 search space."""

    def __init__(self, data_file_path: str, dataset: Dataset):
        """Initializes the NB201BenchmarkAPI object.

        Args:
            data_file_path  : Path to the pickle file containing the given dataset.
            dataset         : The dataset
        """
        super(NB201BenchmarkAPI, self).__init__(dataset)
        self.data_file_path = data_file_path
        self.data = _load_pickle_file(self.data_file_path)

    def query(self, arch: str, metric: Metric, last_epoch_only: bool = False) -> Union[float, List[float]]:
        """Query the API for a given metric of a given architecture.

        Args:
            arch            : String representation of the cell architecture.
            metric          : Metric to query.
            last_epoch_only : If True, returns only the metric value for the final epoch of training.
                              Else, returns a list containing the metric values for each of the epochs.
                              If the metric is either Metric.PARAMS or Metric.TRAIN_TIME, this parameter is ignored.

        Returns:
            The value (or list of values) of the given metric for the given architecture.
        """

        # START TODO #################

        if metric.value == "train_time" or metric.value == "params":
            return self.data[arch][self.dataset.value]["cost_info"][metric.value]
        if last_epoch_only:
            return self.data[arch][self.dataset.value][metric.value][-1]
        return self.data[arch][self.dataset.value][metric.value]
        # END TODO #################


def _load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
