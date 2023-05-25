from lib.utilities import get_results
from lib.plot import plot_comparison


def main():
    losses_and_config_nas = get_results("results/nas")
    losses_and_config_nas_hpo = get_results("results/nas_hpo")
    plot_comparison(losses_and_config_nas, losses_and_config_nas_hpo)


if __name__ == "__main__":
    main()
