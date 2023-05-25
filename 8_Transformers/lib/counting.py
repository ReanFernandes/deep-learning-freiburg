import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from lib.attention import attention_function
from lib.dataset import CountingDataset


class CountingModel(nn.Module):
    """Explicit attention model for counting."""

    def __init__(self, vocab_size, sequence_length, hidden):
        """
        Args:
            vocab_size: Number of different digits in a sequence
            sequence_length: Number of digits in the sequence
            hidden: Intermediate encoding size
        """
        super().__init__()

        self.hidden = hidden
        self.norm = nn.LayerNorm(hidden)
        # Define query, key, value and output linear layer. For
        # query, use nn.Parameter(tensor, requires_grad=True) and
        # initialize it using torch.rand.
        # Shape of query:
        #   - self.query.shape (1, vocab_size, hidden)
        #
        # Shapes of linear layer weight matrices:
        #   - self.key.weight.shape (hidden, vocab_size)
        #   - self.value.weight.shape (hidden, vocab_size)
        #   - self.out.weight.shape (sequence_length + 1, hidden)
        # We are adding 1 to the sequence_length because a digit can
        # appear from 0 to n times in a sequence of length n.
        # START TODO #############
        # Important: The initialization order of the parameter and the
        # layers matters in order to pass the tests.
        # Please, initialize them in the same order as the one mentioned above.
        self.query = nn.Parameter(torch.rand(1, vocab_size, hidden), requires_grad=True)
        self.key = nn.Linear(vocab_size, hidden)
        self.value = nn.Linear(vocab_size, hidden)
        self.out = nn.Linear(hidden, sequence_length + 1)
        # END TODO #############

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: The input tensor with shape (batch_size, sequence_length, vocab_size)

        Returns:
            Tuple of:
                A torch.Tensor of the outputs with shape (batch_size, vocab_size, sequence_length + 1)
                A torch.Tensor of the attention weights with shape (batch_size, vocab_size, sequence_length)
        """

        # Calculate outputs and attention weights.
        # Use torch.Tensor.repeat https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
        # function to repeat query parameter batch times to pass to
        # attention_function.
        # START TODO #############

        q = self.query.repeat(x.shape[0], 1, 1)
        k = self.key(x)
        v = self.value(x)
        attention, attention_weights = attention_function(q, k, v, self.hidden)
        attention = self.norm(attention+q)
        outputs = self.out(attention)
        
        # END TODO #############
        return outputs.cpu(), attention_weights.cpu()


def train():
    torch.manual_seed(0)
    np.random.seed(0)
    sequence_length = 10
    vocab_size = 3
    hidden = 8
    batchsize = 256
    print_every = 50
    steps = 2500

    attention = CountingModel(vocab_size, sequence_length, hidden)

    optimizer = torch.optim.Adam(attention.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=250)
    train_dataset = CountingDataset(vocab_size, sequence_length, batchsize * steps)
    train_loader = DataLoader(train_dataset, batch_size=batchsize)
    test_dataset = CountingDataset(vocab_size, sequence_length, 32 * 10)

    attention.train()
    for step, (x, y) in enumerate(train_loader):
        out, _ = attention(x)
        loss = F.cross_entropy(out.transpose(1, 2), y)
        acc = torch.sum(out.argmax(dim=-1) == y) / (batchsize * vocab_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step(loss)
        if (step + 1) % print_every == 0:
            print(f"Step: {step+1}, Loss: {loss}, Accuracy: {acc}")

    save_path = Path("results")
    os.makedirs(save_path, exist_ok=True)
    torch.save(attention.state_dict(), os.path.join(save_path, "model.pth"))

    attention.eval()
    x, y = test_dataset[:]
    out, _ = attention(x)
    loss = F.cross_entropy(out.transpose(1, 2), y)
    acc = torch.sum(out.argmax(dim=-1) == y) / (len(y) * vocab_size)
    print(f"Test Loss: {loss}, Test Accuracy: {acc}")
