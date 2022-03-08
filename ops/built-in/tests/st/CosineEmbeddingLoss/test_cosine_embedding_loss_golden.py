import numpy as np


def calc_expect_func(x1, x2, target, y,
                     margin=0, reduction='mean', epsilon=1e-5):
    x1_val = x1["value"]
    x2_val = x2["value"]
    target = target["value"]

    prod_num = np.sum(x1_val * x2_val, axis=1)
    mag_square1 = np.sum(x1_val ** 2, axis=1) + epsilon
    mag_square2 = np.sum(x2_val ** 2, axis=1) + epsilon
    denom = np.sqrt(mag_square1 * mag_square2)
    cos = prod_num / denom

    zeros = np.zeros_like(target)
    pos = 1 - cos
    neg = np.maximum(cos - margin, 0)

    output_pos = np.where(target == 1, pos, zeros)
    output_neg = np.where(target == -1, neg, zeros)
    output = output_pos + output_neg

    if reduction == 'mean':
        output = np.mean(output, keepdims=True)

    if reduction == 'sum':
        output = np.sum(output, keepdims=True)

    return output