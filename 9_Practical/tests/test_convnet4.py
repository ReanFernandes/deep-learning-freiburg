from lib.models import ConvNet4
from tests.convnet_tester import convnet_test
from tests.results import convnet4_result


def test_convnet4():
    convnet_test(ConvNet4, convnet4_result)


if __name__ == '__main__':
    test_convnet4()
    print('Test complete.')
