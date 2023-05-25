from lib.resnet import ResNet
from lib.plot import show_attention_on_input_image


if __name__ == "__main__":
    show_attention_on_input_image(ResNet())
