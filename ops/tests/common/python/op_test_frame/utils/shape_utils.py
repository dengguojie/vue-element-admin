from functools import reduce


def calc_shape_size(shape):
    if not shape:
        return 0
    reduce(lambda x, y: x * y, shape)
