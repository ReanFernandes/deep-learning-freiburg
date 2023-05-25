from lib.models import ConvNet1
from tests.convnet_tester import convnet_test
from tests.results import convnet1_result


def test_convnet1():
    convnet_test(ConvNet1, convnet1_result)


if __name__ == '__main__':
    test_convnet1()
    print('Test complete.')
