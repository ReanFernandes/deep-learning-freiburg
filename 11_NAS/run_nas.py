import neps
import torch.nn as nn
import logging
import torch.optim

from neps.search_spaces.architecture import topologies as topos
from lib.primitives import DownSampleBlock, AvgPool, ReLUConvBN, Identity, Zero
from lib.training import train
from lib.utilities import set_seed

primitives = {
    # START TODO #################
    # Specify the primitives dictionary based on the nb201 search space (you can use the primitives dictionary from
    # run_nas_simple.py as a guideline). Use the primitives defined in lib/primitives.py where necessary.
    # Note: ReLUConvBN1x1 should have a 1x1 kernel, stride of 1 and 0 padding.
    # The Zero operation should have a stride of 1.

    "Sequential15": topos.get_sequential_n_edge(15),  # total number of cells in the network (not counting downsampling)
    "DenseCell": topos.get_dense_n_node_dag(4),  # number of nodes in each cell
    "down": {"op": DownSampleBlock},  # downsample block
    "AvgPool1x1": {"op": AvgPool},  # average pooling
    "Identity": {"op": Identity},  # identity operation
    "ReLUConvBN1x1": {"op": ReLUConvBN, "kernel_size": 1, "stride": 1, "padding": 0},  # 1x1 convolution
    "ReLUConvBN3x3": {"op": ReLUConvBN, "kernel_size": 3, "stride": 1, "padding": 1},  # 3x3 convolution
    "Zero": {"op": Zero, "stride": 1}

    # END TODO #################
}
# START TODO #################
# Specify the structure list based on the nb201 search space (you can use the structure list from
# run_nas_simple.py as a guideline). Use N=5.
# Keep in mind that you need to use DownSampleBlock after the search cells
# structure = ...
structure = [
    {"S": ["Sequential15(C, C, C, C, C, down, C, C, C, C, C, down, C, C, C, C, C)"]},
    {"C": ["DenseCell(OPS, OPS, OPS, OPS, OPS, OPS)"],  # Operations for each edge of the cell
     "OPS": ["Identity", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1", "Zero"]}
    # Possible operations to choose from
]

# END TODO #################

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
    out_channels_factor = 4

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
    # 1. Add STEM layers(Conv2d, BatchNorm2d from PyTorch) before the model suggested by NePS
    #   - The convolution layer should have a 3x3 kernel with padding equal to 1 and no bias
    # 2. Add the model suggested by NePS
    # 3. Add a batch normalization (BatchNorm2d) layer
    # 4. Add a ReLU activation layer
    # 5. Add an adaptive average pooling layer with output size 1
    # 6. Flatten the output and add a final linear layer.
    # Use SGD optimizer to train the model and save the outcome of the training to a dictionary named "results"
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
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/nas",
        max_evaluations_total=10,
        searcher="bayesian_optimization",
        loss_value_on_error=0.7
    )
