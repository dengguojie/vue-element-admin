import os
import sys
import unittest

import numpy as np

from mindspore.ops import operations as P
import mindspore.context as context
from mindspore import Tensor

np.seterr(all='ignore')
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


RESOLUTION = np.finfo(np.float64).resolution
def calculate_cosine_similarity(a, b):
    a = a.astype(np.float64) + RESOLUTION
    b = b.astype(np.float64) + RESOLUTION
    if a.shape != b.shape:
        raise RuntimeError('a shape {} != b shape {}'.format(a.shape, b.shape))
    if a.max() == a.min() and b.max() == b.min():
        return 1
    numerator = np.sum(a * b)
    denominator = ((np.sum(a * a)) ** 0.5) * ((np.sum(b * b)) ** 0.5)
    return numerator / denominator


def cal_wts_arq(w, data_min, data_max, num_bits=8, offset_flag=False):
    eps = 1.192092896e-07
    if offset_flag:
        scale_upper_bound = (data_max * np.array(1 / (2**num_bits - 1)).astype(w.dtype))
        scale_low_bound = (data_min * np.array(1 / (2**num_bits - 1)).astype(w.dtype))
        data_scale = scale_upper_bound - scale_low_bound
        data_scale[data_scale < eps] = 1.0
        data_offset = -1.0 * np.round(data_min / data_scale) - 2**(num_bits - 1)
    else:
        data_scale = np.maximum(data_max / (2**(num_bits - 1) - 1), data_min * -1 / (2**(num_bits - 1)))
        data_scale[data_scale < eps] = 1.0
        data_offset = data_scale * 0.0

    data_y = np.round(w / data_scale) + data_offset
    data_y = np.clip(data_y, -1 * 2**(num_bits - 1), 2**(num_bits - 1) - 1)
    data_y = (data_y - data_offset) * data_scale
    data_y[np.isneginf(data_y)] = np.finfo(w.dtype).min
    data_y[np.isposinf(data_y)] = np.finfo(w.dtype).max

    return data_y


def test_tbe_arq(w, w_min, w_max, num_bits=8, offset_flag=False):
    y = P.WtsARQ(num_bits, offset_flag)(Tensor(w), Tensor(w_min), Tensor(w_max))
    return y


def print_featuremap(origin_data, np_data, tbe_data):
    print('original feature map is:\n {}'.format(origin_data))
    print('numpy output feature map is:\n {}'.format(np_data))
    print('tbe output feature map is:\n {}'.format(tbe_data.asnumpy()))


def compare_data(w, axes=[0], num_bits=8, offset_flag=False, show_data=False):
    w_shape = w.shape
    axis = []
    for item in range(len(w_shape)):
        if item not in axes:
            axis.append(item)
    axis = tuple(axis)

    data_min = w.min(axis=axis, keepdims=True, initial=0)
    data_max = w.max(axis=axis, keepdims=True, initial=0)

    data_y = cal_wts_arq(w, data_min, data_max, num_bits, offset_flag)
    y = test_tbe_arq(w, data_min, data_max, num_bits, offset_flag)
    y_sim = calculate_cosine_similarity(data_y, y.asnumpy())

    wrong_sim_flag = False
    for item in [y_sim]:
        if abs(item - 1.0) > 1e-6:
            wrong_sim_flag = True
            show_data = True

    if True:
    # if show_data:
        # print_featuremap(w, data_y, y)

        # print('original && numpy feature map similarity is', calculate_cosine_similarity(w, data_y))
        print('feature map similarity is', y_sim)

    if wrong_sim_flag:
        raise RuntimeError('similarity is wrong.')


class TestWtsARQ(unittest.TestCase):
    def setUp(self):
        os.system('rm kernel_meta/ -rf')

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_fp32_offset_false(self):
        w = np.random.randn(2,3,4).astype(np.float32)
        compare_data(w)

    def test_fp32_offset_true(self):
        w = np.random.randn(2,3,4).astype(np.float32)
        compare_data(w, offset_flag=True)

    def test_fp16_offset_false(self):
        w = np.random.randn(2,3,4).astype(np.float16)
        compare_data(w)

    def test_fp16_offset_true(self):
        w = np.random.randn(2,3,4).astype(np.float16)
        compare_data(w, offset_flag=True)

    def test_fp16_axes_null_offset_false(self):
        w = np.random.randn(2,3,4).astype(np.float16)
        compare_data(w, axes=[])

    def test_fp16_axes_null_offset_true(self):
        w = np.random.randn(2,3,4).astype(np.float16)
        compare_data(w, axes=[], offset_flag=True)

    def test_fp32_offset_false_axes_full(self):
        w = np.random.randn(2,3,4).astype(np.float32)
        compare_data(w, axes=[0,1,2], show_data=False)

    def test_fp32_offset_true_axes_full(self):
        w = np.random.randn(2,3,4).astype(np.float32)
        compare_data(w, axes=[0, 1, 2], offset_flag=True, show_data=False)

    def test_fp32_offset_true_zero_input(self):
        w = np.zeros((2,3,4)).astype(np.float32)
        compare_data(w, offset_flag=True, show_data=False)

    def test_fp32_offset_false_zero_input(self):
        w = np.zeros((2,3,4)).astype(np.float32)
        compare_data(w, offset_flag=False, show_data=False)

    def test_fp16_offset_true_zero_input(self):
        w = np.zeros((2,3,4)).astype(np.float16)
        compare_data(w, offset_flag=True, show_data=False)

    def test_fp16_offset_false_zero_input(self):
        w = np.zeros((2,3,4)).astype(np.float16)
        compare_data(w, offset_flag=False, show_data=False)

    def test_fp32_offset_true_random_0_0001(self):
        w = np.random.uniform(-0.0001, 0.0001, (2,3,4)).astype(np.float32)
        compare_data(w, offset_flag=True, show_data=False)

    def test_fp16_offset_true_random_0_0001(self):
        w = np.random.uniform(-0.0001, 0.0001, (2,3,4)).astype(np.float16)
        compare_data(w, offset_flag=True, show_data=False)

    def test_fp16_offset_false_random_0_0001(self):
        w = np.random.uniform(-0.0001, 0.0001, (2,3,4)).astype(np.float16)
        compare_data(w, offset_flag=False, show_data=False)

    def test_fp16_offset_true_random_max(self):
        w = np.random.randn(2,3,4).astype(np.float16)
        w[w <= 0] = np.finfo(np.float16).min
        w[w > 0] = np.finfo(np.float16).max
        compare_data(w, offset_flag=True, show_data=False)

    def test_fp16_offset_false_random_max(self):
        w = np.random.randn(2,3,4).astype(np.float16)
        w[w <= 0] = np.finfo(np.float16).min
        w[w > 0] = np.finfo(np.float16).max
        compare_data(w, offset_flag=False, show_data=False)

    def test_fp32_offset_true_random_max(self):
        w = np.random.randn(2,3,4).astype(np.float32)
        w[w <= 0] = np.finfo(np.float32).min
        w[w > 0] = np.finfo(np.float32).max
        compare_data(w, offset_flag=True, show_data=False)

    def test_fp32_offset_false_random_max(self):
        w = np.random.randn(2,3,4).astype(np.float32)
        w[w <= 0] = np.finfo(np.float32).min
        w[w > 0] = np.finfo(np.float32).max
        compare_data(w, offset_flag=False, show_data=False)


if __name__ == '__main__':
    # os.environ['DUMP_GE_GRAPH'] = '1'
    # os.environ['SLOG_PRINT_TO_STDOUT'] = '1'
    unittest.main()
