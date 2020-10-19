import math
import sys


def calc_loss(model_preds_file, input_file):
    model_preds = open(model_preds_file, 'rt')
    input = open(input_file, 'rt')

    loss = 0.
    i = 0
    for y_hat in model_preds:
        i += 1
        y = next(input).split("|")[0].strip()
        loss += cross_entropy(float(y_hat), float(y))

    return loss / float(i)


def cross_entropy(y_hat, y):
    try:
        return -math.log(y_hat) if y == 1 else -math.log(1 - y_hat)
    except ValueError:
        return cross_entropy(1e-15, y)


if __name__ == "__main__":
    loss = calc_loss(sys.argv[1], sys.argv[2])
    print(f"loss: {loss}")