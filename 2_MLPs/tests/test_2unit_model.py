from lib.models import create_2unit_net, run_test_model


def test_2unit_model():
    """Create the 2 unit 2 layer network and test it"""
    model = create_2unit_net()
    run_test_model(model)


if __name__ == '__main__':
    test_2unit_model()
    print("Test complete.")
