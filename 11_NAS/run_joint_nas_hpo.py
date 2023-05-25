import os
import shutil
import neps
import torch.nn as nn
import logging

import torch.optim
from run_nas import primitives, structure
from lib.training import train
from lib.utilities import set_seed

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
        **repetitive_kwargs
    ),
    # START TODO #################
    # Define the optimizer, learning rate and batch size as NePS parameters as follows:
    #         optimizer       "sgd" or "adam"   (categorical)
    #         learning_rate   from 1e-7 to 1e-3 (log, float)
    #         batch_size      from   16 to   64 (int)
    # optimizer = ...
    # learning_rate = ...
    # batch_size = ...
    optimizer=neps.CategoricalParameter(choices=["sgd", "adam"]),
    learning_rate=neps.FloatParameter(lower=10e-7, upper=10e-3, log=True),
    batch_size=neps.IntegerParameter(lower=16, upper=64)
    # END TODO #################
)


def run_pipeline(architecture, optimizer, learning_rate, batch_size):
    set_seed(124)
    in_channels = 3
    base_channels = 16
    n_classes = 10
    out_channels_factor = 4

    # Specify number of epochs, and criterion (loss function)
    # START TODO #################
    num_epochs = 5
    criterion = torch.nn.CrossEntropyLoss()
    # END TODO #################

    # Convert NePS architecture to PyTorch model
    model = architecture.to_pytorch()

    # START TODO #################
    # Create a Sequential model as follows:
    # 1. Add STEM layers(Conv2d, BatchNorm2d from PyTorch) before the model suggested by NePS
    #   - The convolution layer should have a 3x3 kernel with padding equal to 1 and no bias
    # 2. Add the model suggested by NePS
    # 3. Add a batch normalization (BatchNorm2d) layer
    # 4. Add a ReLU activation layer
    # 5. Add an adaptive average pooling layer with output size 1
    # 6. Flatten the output and add a final linear layer.
    # The SGD optimizer should have the learning rate specified on the NePS pipeline
    # If the optimizer is SGD, use momentum=0.9
    model = nn.Sequential(
        nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(base_channels),
        model,
        nn.BatchNorm2d(base_channels * out_channels_factor),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(base_channels * out_channels_factor, n_classes)
    )
    if optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    results = train(model, num_epochs, batch_size, criterion, optim)
    # END TODO #################
    return {
        "loss": results["val_error"][-1],
        "info_dict": {
            "train_acc": results["train_acc"],
            "train_loss": results["train_loss"],
            "val_loss": results["val_loss"],
            "val_errors": results["val_error"],
            "train_time": results["train_time"],
            "params": results["params"],
            "cost": num_epochs
        }

    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_seed(124)
    if os.path.exists("results/nas_hpo"):
        shutil.rmtree("results/nas_hpo")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/nas_hpo",
        max_evaluations_total=10,
        searcher="bayesian_optimization",
        loss_value_on_error=0.7
    )
