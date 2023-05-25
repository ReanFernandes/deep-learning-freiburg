from lib.models import ConvNet3
from tests.convnet_tester import convnet_test
from tests.results import convnet3_result


def test_convnet3():
    convnet_test(ConvNet3, convnet3_result)


if __name__ == '__main__':
    test_convnet3()
    print('Test complete.')
