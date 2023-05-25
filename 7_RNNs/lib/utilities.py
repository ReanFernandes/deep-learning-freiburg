"""Helper functions for data generation"""

from typing import List, Tuple

import numpy as np
import torch


NUM_FUNCTIONS_TRAIN = 200
NUM_FUNCTIONS_VAL = 50
SEQUENCE_LENGTH = 80
NOISE_RATIO = 0.05
x = np.linspace(0, 5, SEQUENCE_LENGTH)


class RandomSineFunction:
    """Represents a function f(x) which is sum of a random number of random sine functions"""

    def __init__(self):
        num_sines = np.random.randint(1, 4)
        self.amplitude = np.random.uniform(0, 2, num_sines)
        self.offsets = np.random.uniform(-np.pi, np.pi, num_sines)
        self.frequency = np.random.uniform(0.1, 1, num_sines)

    def __call__(self, x: float) -> np.ndarray:
        """Query a point in the in the composite function y = f(x)

        Args:
            x: The points to query of shape (SEQUENCE_LENGTH,)

        Returns:
            Value of the function at the given points, i.e., f(x) of shape (SEQUENCE_LENGTH,)
        """

        return np.array([a * np.sin(np.pi * f * x + o)
                         for a, f, o in zip(self.amplitude, self.frequency, self.offsets)]).sum(axis=0)


def sample_sine_functions(num_functions: int) -> List[RandomSineFunction]:
    """Create a list of random sine functions

    Args:
        num_functions: Number of RandomSineFunctions to create.

    Returns:
        List of RandomSineFunctions
    """
    return [RandomSineFunction() for _ in range(num_functions)]


def prepare_sequences(functions: List[RandomSineFunction]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert to tensor and create noisy sequence
    Args:
        functions: List of RandomSineFunctions

    Returns:
        Tuple of sequences and same sequences with noise of shapes (SEQUENCE_LENGTH, number of sine functions, 1)
    """
    sequences = np.array([f(x).reshape(-1, 1) for f in functions])
    # the function is in dimension 0, the sequence in dimension 1 and the value in dimension 2 with length 1
    # add some noise, scale noise individually over each function by excluding the function axis (0)
    noisy_sequences = noisy(sequences, NOISE_RATIO, axes=(1, 2))
    return torch.Tensor(sequences), torch.Tensor(noisy_sequences)


def percentage_noise_removed(ground_truth: np.ndarray, noisy_sequence: np.ndarray, model_output: np.ndarray) -> float:
    """Compute the percentage of noise the model removed.

    Args:
        ground_truth: Original signal without the noise of shape (number of sine functions, SEQUENCE_LENGTH, 1).
        noisy_sequence: Signal with noise of shape (number of sine functions, SEQUENCE_LENGTH, 1).
        model_output: Denoised signal from the model of shape (number of sine functions, SEQUENCE_LENGTH, 1).

    Returns:
        Percentage of noise removed by the model (scalar value)
    """
    percentage_removed = 100 * (1 - (np.abs(ground_truth - model_output).sum() /
                                     np.abs(ground_truth - noisy_sequence).sum()))
    return percentage_removed if percentage_removed >= 0 else 0


def noisy(y: np.ndarray, noise_ratio=0.05, axes=None) -> np.ndarray:
    """Add Gaussian noise to the given input signal

    Args:
        y: Original signal of shape (SEQUENCE_LENGTH, number of functions, 1)
        noise_ratio: Noise ratio to use to corrupt original signal.
        axes: Over which axis to compute the noise range.

    Returns:
        Noisy signal.
    """
    # np.ptp gives the range from "peak to peak" (maximum to minimum) over the axis.
    # this way, the noise is automatically scaled to the value range of y.
    noise_range = np.ptp(y, axis=axes, keepdims=True) * noise_ratio
    return y + np.random.normal(0, noise_range, size=y.shape)
