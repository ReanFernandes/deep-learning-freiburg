import os
import shutil
import neps
import torch.nn as nn
import logging
import torch.optim

from neps.search_spaces.architecture import topologies as topos
from lib.primitives import AvgPool, ReLUConvBN, Identity, DownSampleBlock
from lib.training import train
from lib.utilities import set_seed

primitives = {
    "Sequential4": topos.get_sequential_n_edge(4),  # total number of cells in the network (not counting downsampling)
    "DenseCell": topos.get_dense_n_node_dag(3),  # number of nodes in each cell
    "down": {"op": DownSampleBlock},  # downsample block
    "AvgPool1x1": {"op": AvgPool},  # average pooling
    "Identity": {"op": Identity},  # identity operation
    "ReLUConvBN3x3": {"op": ReLUConvBN, "kernel_size": 3, "stride": 1, "padding": 1},  # 3x3 convolution
}
# Structure of the model
structure = [
    {"S": ["Sequential4(C, C, down, C, C)"]},  # Cell x 2 -> downsample -> Cell x 2
    {"C": ["DenseCell(OPS, OPS, OPS)"],  # Operations for each edge of the cell
     "OPS": ["Identity", "ReLUConvBN3x3", "AvgPool1x1"], }  # Possible operations to choose from
]

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
pipeline_space = dict(
    architecture=neps.ArchitectureParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        zero_op=None,
        **repetitive_kwargs,
    )
)


def run_pipeline(architecture):
    set_seed(124)
    # Input channels
    in_channels = 3
    # Base channels of the first convolutional layer
    base_channels = 16
    # Number of classes
    n_classes = 10

    # Specify batch size, learning rate, number of epochs, and criterion (loss function)
    # START TODO #################
    batch_size = 64
    lr = 1e-3
    num_epochs = 5
    criterion = torch.nn.CrossEntropyLoss()
    # END TODO #################

    # Convert NePS architecture to PyTorch model
    model = architecture.to_pytorch()

    # START TODO #################
    # Create a Sequential model as follows:
    # 1. First add a 3x3 Convolution layer with 3 input and 16 output channels, padding 1 and no bias
    # 2. Then add the model suggested by NePS
    # 3. Then add a Flatten layer
    # 4. Finally add a Linear layer
    # Use SGD optimizer to train the model and save the outcome of the training to a dictionary named "results"
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
        model,
        nn.Flatten(),
        nn.Linear(2048, 10)
    )
    optim = torch.optim.SGD(model.parameters(), lr=lr)
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
    # Run the NAS pipeline and save the results
    if os.path.exists("results/nas_simple"):
        shutil.rmtree("results/nas_simple")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/nas_simple",
        max_evaluations_total=10,
        searcher="bayesian_optimization",
        loss_value_on_error=0.7
    )
