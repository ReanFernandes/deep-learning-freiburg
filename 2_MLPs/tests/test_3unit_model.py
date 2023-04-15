from lib.models import create_3unit_net, run_test_model


def test_3unit_model():
    """Create the 3 unit 2 layer network and test it"""
    model = create_3unit_net()
    run_test_model(model)


if __name__ == '__main__':
    test_3unit_model()
    print("Test complete.")
