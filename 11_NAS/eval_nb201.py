from lib.utilities import get_results
from lib.plot import plot_error_curves, plot_incumbents


def plot_results(results) -> None:
    """Plot the results from different runs.

      Args:
          results: Structure containing results of the runs

      Returns:
          None

    """
    # Get structured results
    losses_and_config = get_results(results)

    # Plot the validation errors over time (epochs)
    plot_error_curves(losses_and_config)

    # Plot the incumbents
    plot_incumbents(losses_and_config)


if __name__ == '__main__':
    plot_results('results/nas')
