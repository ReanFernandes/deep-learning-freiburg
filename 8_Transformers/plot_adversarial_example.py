from lib.resnet import ResNet
from lib.plot import show_adversarial_example


if __name__ == "__main__":
    show_adversarial_example(ResNet())
