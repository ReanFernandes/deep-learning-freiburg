"""Script to run the 3 unit model."""

from lib.models import create_3unit_net, run_model_on_xor


def main():
    # create the 3 layer network and run it
    model = create_3unit_net()
    run_model_on_xor(model)


if __name__ == '__main__':
    main()
