import os
import shutil
import neps
import logging

from lib.benchmark_api import NB201BenchmarkAPI, Dataset, Metric
from lib.utilities import convert_identifier_to_str, set_seed
from run_nas import primitives, structure


repetitive_kwargs = {
    "fixed_macro_grammar": True,
    "terminal_to_sublanguage_map": {
        "C": "C",
    },
}


def set_recursive_attribute(op_name, predecessor_values):
    in_channels = 16 if predecessor_values is None else predecessor_values["out_channels"]
    out_channels = in_channels * 2 if op_name == "DownSampleBlock" else in_channels
    return dict(in_channels=in_channels, out_channels=out_channels)


# Pipeline space used by NePS
# Note: Since we are going to us the same search space as in run_nas,
# we are going to reuse the primitives and structure variables
pipeline_space = dict(
    architecture=neps.ArchitectureParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        zero_op=None,
        **repetitive_kwargs,

    )
)

terminals_to_nb201 = {
        "AvgPool1x1": "avg_pool_3x3",
        "ReLUConvBN1x1": "nor_conv_1x1",
        "ReLUConvBN3x3": "nor_conv_3x3",
        "Identity": "skip_connect",
        "Zero": "none",
    }


def run_pipeline(architecture):
    set_seed(124)
    api = NB201BenchmarkAPI("./benchmark/nb201_cifar10_full_training.pickle", Dataset.CIFAR10)
    arch = convert_identifier_to_str(architecture.id, terminals_to_nb201)
    # Query train accuracy, train loss, validation accuracy, validation loss, training time and number of parameters
    # Query those metrics only for the last epoch
    # START TODO #################
    train_acc = api.query(arch, Metric.TRAIN_ACCURACY, last_epoch_only=True)
    train_loss = api.query(arch, Metric.TRAIN_LOSS, last_epoch_only=True)
    val_acc = api.query(arch, Metric.VAL_ACCURACY, last_epoch_only=True)
    val_loss = api.query(arch, Metric.VAL_LOSS, last_epoch_only=True)
    train_time = api.query(arch, Metric.TRAIN_TIME)
    params = api.query(arch, Metric.PARAMS)
    # END TODO #################

    return {
        "loss": 1 - val_acc / 100,
        "info_dict": {
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_time": train_time,
            "params": params
        }

    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_seed(124)
    if os.path.exists("results/nas_api"):
        shutil.rmtree("results/nas_api")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/nas_api",
        max_evaluations_total=50,
        searcher="bayesian_optimization",
        loss_value_on_error=0.7
    )
