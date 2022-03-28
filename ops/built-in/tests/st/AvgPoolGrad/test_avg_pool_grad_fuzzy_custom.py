#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for avgPoolGrad fuzzy compile
from impl.dynamic.avg_pool_grad import avg_pool_grad_generalization
from impl.dynamic.avg_pool_grad import UNSUPPORTED_FUZZ_RES


LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": ["lower_limit"]}}]
UPPER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": ["upper_limit"]}}]

def test_avg_pool_grad_fuzzy_compile_head_node_1():
    # input_size, dedy, filter, dedx, ksize, strides, padding
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 26, 28), 'shape': (1, 1, 26, 28, 16) ,'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (16, 16), (26, 26), (28, 28)),
         'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        {'ori_shape': (16, 1, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
         'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        (16, 1, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_1',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 26, 28, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (16, 16), [16, 31], [16, 31]], 'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        {'ori_shape': (16, 1, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        {'strides': (1, 1, 1, 1)}, {'padding': 'VALID'}, {'ksize': (16, 1, 3, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_head_node_1'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_head_node_2():
    # N dim exceeds
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (21474836548, 32, 26, 28), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (32, 32), (26, 26), (28, 28))},
        {'ori_shape': (32, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_2',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_3():
    # H/W dim exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 32, 4900, 28), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (32, 32), (26, 26), (28, 28))},
        {'ori_shape': (32, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_3',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_4():
    # [-2]
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (-2,), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': None},
        {'ori_shape': (32, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_4',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_5():
    # dedy exceeds l1 size limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 32, 60, 1000), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (32, 32), (60, 60), (1000, 1000))},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 90, 1002), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_5',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_6():
    # dedy exceeds l1 size limit, correct w range
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 60, 900), 'shape': (1, 1, 60, 900, 16) ,'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (16, 16), (60, 60), (900, 900)),
         'range': ((1, 1), (1, 1), (60, 60), (900, 900), (16, 16))},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 90, 902), 'shape': (1, 1, 90, 902, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
         'range': ((1, 1), (1, 1), (90, 90), (902, 902), (16, 16))},
        (16, 1, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_6',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 60, 900, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (16, 16), [32, 63], [768, 977]],
        'range': ((1, 1), (1, 1), (60, 60), (900, 900), (16, 16))},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 90, 902, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'range': ((1, 1), (1, 1), (90, 90), (902, 902), (16, 16))},
        {'strides': (1, 1, 1, 1)}, {'padding': 'VALID'}, {'ksize': (16, 1, 31, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_head_node_6'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_head_node_7():
    # kernel_matrix is None
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 26, 28), 'shape': (1, 1, 26, 28, 16) ,'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (16, 16), (26, 26), (28, 28)),
         'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        None,
        {'ori_shape': (1, 16, 28, 30), 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
         'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        (16, 1, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_7',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 26, 28, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (16, 16), [16, 31], [16, 31]], 'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        {'ori_shape': [16, 1, 3, 3], 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        {'strides': (1, 1, 1, 1)}, {'padding': 'VALID'}, {'ksize': (16, 1, 3, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_head_node_7'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_body_node_1():
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape':(-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, 977]],
        'range': [[1, 1], (1, 1), [32, 63], [768, 977], [16, 16]]},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape':(-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)],
        'range': [(1, 1), (1, 1), (62, 93), (770, 979), (16, 16)]},
        (16, 1, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_1',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, 977]],
        'range': [[1, 1], (1, 1), [32, 63], [768, 977], [16, 16]]},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)],
        'range': [(1, 1), (1, 1), (62, 93), (770, 979), (16, 16)]}, {'strides': (1, 1, 1, 1)},
        {'padding': 'VALID'}, {'ksize': (16, 1, 31, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_body_node_1'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_body_node_2():
    # N dim lower bound exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[21474836548, 1], (32, 32), [32, 63], [768, 977]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_2',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == LOWER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_3():
    # H dim upper bound exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 4100], [768, 977]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_3',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UPPER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_4():
    # W dim lower bound exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [4100, 977]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_4',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == LOWER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_5():
    # W dim upper bound is -1
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, -1]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_5',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UPPER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_6():
    # dedy exceeds l1 size limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, 1000]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 1000)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_6',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UPPER_STR


if __name__ == "__main__":
    test_avg_pool_grad_fuzzy_compile_head_node_1()
    test_avg_pool_grad_fuzzy_compile_head_node_2()
    test_avg_pool_grad_fuzzy_compile_head_node_3()
    test_avg_pool_grad_fuzzy_compile_head_node_4()
    test_avg_pool_grad_fuzzy_compile_head_node_5()
    test_avg_pool_grad_fuzzy_compile_head_node_6()
    test_avg_pool_grad_fuzzy_compile_head_node_7()
    test_avg_pool_grad_fuzzy_compile_body_node_1()
    test_avg_pool_grad_fuzzy_compile_body_node_2()
    test_avg_pool_grad_fuzzy_compile_body_node_3()
    test_avg_pool_grad_fuzzy_compile_body_node_4()
    test_avg_pool_grad_fuzzy_compile_body_node_5()
    test_avg_pool_grad_fuzzy_compile_body_node_6()