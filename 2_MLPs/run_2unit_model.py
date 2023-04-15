"""Script to run the 2 unit model."""

from lib.models import create_2unit_net, run_model_on_xor


def main():
    # create the 2 layer network and run it
    model = create_2unit_net()
    run_model_on_xor(model)


if __name__ == '__main__':
    main()
