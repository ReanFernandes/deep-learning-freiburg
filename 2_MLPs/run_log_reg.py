"""Script to run the logistic regression."""

from lib.dataset import X, y
from lib.log_reg import logistic_regression


def main():
    # calculate the logistic regression given the dataset
    p, s = logistic_regression(X, y)

    # print the prediction and score
    print(f"Input:\n{X}\n")
    print(f"Labels:{y}\n")
    print(f"Predicted Labels: {p}\n")
    print(f"Score (Accuracy): {s:.3f}")


if __name__ == '__main__':
    main()
