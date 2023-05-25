from lib.models import ConvNet2
from tests.convnet_tester import convnet_test
from tests.results import convnet2_result


def test_convnet2():
    convnet_test(ConvNet2, convnet2_result)


if __name__ == '__main__':
    test_convnet2()
    print('Test complete.')
