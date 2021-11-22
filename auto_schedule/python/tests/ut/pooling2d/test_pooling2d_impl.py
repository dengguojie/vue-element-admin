# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_pooling2d(x, _, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode, ceil_mode,
                  fusion_params, impl_mode, kernel_name='dsl_pooling2d'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.pooling2d(data1, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode, ceil_mode,
                        fusion_params, impl_mode)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("pooling2d", "pooling2d.test_pooling2d_impl", "dsl_pooling2d")


def test_pooling2d_para_check_tensor_in(_):
    """
    @return: Ture
    """
    window = (8, 8)
    stride = (1, 1)
    pooling_mode = "AVG"
    padding_mode = "SAME"

    # check instance
    try:
        input1 = tvm.const(1, dtype="float16")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check length of shape
    try:
        input1 = tvm.placeholder((1, 1, 8, 8, 16, 16), name="input1", dtype="float16")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check length of shape
    try:
        input1 = tvm.placeholder((1, 1, -1, -1, 16), name="input1", dtype="float16")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check dtype of shape
    try:
        input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float32")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check batch_size
    try:
        input1 = tvm.placeholder((0, 1, 8, 8, 16), name="input1", dtype="float16")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check c1
    try:
        input1 = tvm.placeholder((1, 0, 8, 8, 16), name="input1", dtype="float16")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check c0
    try:
        input1 = tvm.placeholder((1, 1, 8, 8, 15), name="input1", dtype="float16")
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check avg window * windowh > 255
    try:
        input1 = tvm.placeholder((1, 1, 28, 28, 16), name="input1", dtype="float16")
        tbe.pooling2d(input1, (16, 16), stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_pooling2d_para_check_window(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    stride = (1, 1)
    pooling_mode = "AVG"
    padding_mode = "SAME"

    # check instance
    try:
        window = {}
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check length of window
    try:
        window = (1, 1, 8, 8)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check value range of window
    try:
        window = (1, 0)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    try:
        window = (32769, 1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_pooling2d_para_check_stride(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    pooling_mode = "MAX"
    padding_mode = "SAME"

    # check instance
    try:
        stride = {}
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check length of stride
    try:
        stride = (1, 1, 1, 1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check value range of window
    try:
        stride = (64, 1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    try:
        stride = (1, -1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_pooling2d_para_check_pooling_mode(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    stride = (1, 1)
    padding_mode = "SAME"

    # check mode
    try:
        pooling_mode = "ABC"
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_pooling2d_para_check_tf_padding_mode(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    stride = (1, 1)
    pooling_mode = "MAX"
    data_mode = 1

    # check instance
    try:
        padding_mode = 5
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check value of padding_mode
    try:
        padding_mode = "5"
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_pooling2d_para_check_pad(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    stride = (1, 1)
    padding_mode = "SAME"
    pooling_mode = "MAX"
    data_mode = 1

    # check instance
    try:
        pad = {}
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, pad=pad, data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check len of pad
    try:
        pad = (1, 2, 3, 4, 5)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, pad=pad, data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check value of pad
    try:
        pad = (-1, 1, 1, 1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, pad=pad, data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_pooling2d_para_check_dilation(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    stride = (1, 1)
    padding_mode = "SAME"
    pooling_mode = "MAX"
    data_mode = 1

    # check instance
    try:
        dilation = {}
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, dilation=dilation,
                      data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check len of dilation
    try:
        dilation = (1, 2, 3)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, dilation=dilation,
                      data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    # check value of dilation
    try:
        dilation = (256, 1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, dilation=dilation,
                      data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    try:
        dilation = (1, -1)
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, dilation=dilation,
                      data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_pooling2d_para_check_data_mode(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    stride = (1, 1)
    padding_mode = "SAME"
    pooling_mode = "MAX"

    # check instance
    try:
        data_mode = 3
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, data_mode=data_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_pooling2d_para_check_caffe_ceil_mode(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (2, 2)
    stride = (1, 1)
    padding_mode = "SAME"
    pooling_mode = "MAX"
    data_mode = 0

    # check value of ceil_mode
    try:
        ceil_mode = 3
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode, data_mode=data_mode, ceil_mode=ceil_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_pooling2d_check_pooling_mode_with_padding_mode(_):
    """
    @return: Ture
    """
    input1 = tvm.placeholder((1, 1, 8, 8, 16), name="input1", dtype="float16")
    window = (10, 10)
    stride = (1, 1)
    padding_mode = "VALID"
    pooling_mode = "MAX"
    try:
        tbe.pooling2d(input1, window, stride, pooling_mode, padding_mode)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_max_pool2d_not_support_kernel_para(_):
    """
    @return: Ture
    """
    shape = (1, 1, 21, 23, 16)
    windows = (21, 21)
    stride = (1, 1)
    input1 = tvm.placeholder(shape, name="input1", dtype="float16")
    try:
        tbe.pooling2d(input1, windows, stride, "MAX", "VALID")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_check_outsize_pad_with_ceil(_):
    """
    @return: Ture
    """
    shape = (1, 1, 64, 64, 16)
    windows = (1, 1)
    stride = (1, 1)
    input1 = tvm.placeholder(shape, name="input1", dtype="float16")

    try:
        tbe.pooling2d(input1, windows, stride, "MAX", "SAME", (1, 2, 1, 1), (1, 1), 0, 1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    try:
        tbe.pooling2d(input1, windows, stride, "MAX", "SAME", (1, 1, 1, 2), (1, 1), 0, 1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


def test_check_ub_tiling(_):
    """
    check raise error msg for function _check_ub_tiling
    @return: Ture
    """

    data_mode = 0
    pooling_mode = "AVG"
    out_size_w, window_h, window_w = (1, 3, 3)
    padding_mode = "SAME"
    from tbe.dsl.compute.pooling2d import _check_ub_tiling

    try:
        _check_ub_tiling(data_mode, pooling_mode, padding_mode, out_size_w, window_h, window_w, 16, 0)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))

    return True


test_func_list = [
    test_pooling2d_para_check_tensor_in,
    test_pooling2d_para_check_window,
    test_pooling2d_para_check_stride,
    test_pooling2d_para_check_pooling_mode,
    test_pooling2d_para_check_tf_padding_mode,
    test_pooling2d_para_check_pad,
    test_pooling2d_para_check_dilation,
    test_pooling2d_para_check_data_mode,
    test_pooling2d_para_check_caffe_ceil_mode,
    test_pooling2d_check_pooling_mode_with_padding_mode,
    test_max_pool2d_not_support_kernel_para,
    test_check_outsize_pad_with_ceil,
    test_check_ub_tiling,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (1, 31, 18, 174, 16), "dtype": "float16"},
               {"shape": (1, 31, 18, 174, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_GMP_VALID",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (18, 174), "stride": (10, 23), "pooling_mode": "GMP", "padding_mode": "VALID",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case2 = {
    "params": [{"shape": (20, 180, 416, 416, 16), "dtype": "float16"},
               {"shape": (20, 180, 416, 416, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_GAP_VALID",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (416, 416), "stride": (2, 2), "pooling_mode": "GAP", "padding_mode": "VALID",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case3 = {
    "params": [{"shape": (1, 1, 64, 64, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_AVG_SAME",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (1, 1), "pooling_mode": "AVG", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case4 = {
    "params": [{"shape": (2, 32, 64, 64, 16), "dtype": "float16"},
               {"shape": (2, 32, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_SAME",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (1, 1), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case5 = {
    "params": [{"shape": (1, 1, 8, 8, 16), "dtype": "float16"},
               {"shape": (1, 1, 8, 8, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_GMP_VALID_run_in_MAX_mode",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (8, 8), "stride": (1, 1), "pooling_mode": "GMP", "padding_mode": "VALID",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case6 = {
    "params": [{"shape": (1, 31, 18, 18, 16), "dtype": "float16"},
               {"shape": (1, 31, 18, 18, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_GMP_SAME_run_in_MAX",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (18, 18), "stride": (10, 23), "pooling_mode": "GMP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case7 = {
    "params": [{"shape": (2, 128, 16, 16, 16), "dtype": "float16"},
               {"shape": (2, 128, 16, 16, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_AVG_SAME_run_in_GAP",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (16, 16), "pooling_mode": "AVG", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case8 = {
    "params": [{"shape": (2, 128, 16, 16, 16), "dtype": "float16"},
               {"shape": (2, 128, 16, 16, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_VALID_run_in_GMP",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (16, 16), "pooling_mode": "MAX", "padding_mode": "VALID",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case9 = {
    "params": [{"shape": (2, 32, 64, 64, 16), "dtype": "float16"},
               {"shape": (2, 32, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_CEIL_IS_ONE",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (1, 1), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 1,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case10 = {
    "params": [{"shape": (2, 32, 64, 64, 16), "dtype": "float16"},
               {"shape": (2, 32, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_KW_IS_ONE",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (1, 1), "stride": (1, 1), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 1,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case11 = {
    "params": [{"shape": (1, 1, 64, 64, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_AVG_DATA_MODE_ZERO_CEIL_MODE_ZERO",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (1, 1), "pooling_mode": "AVG", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case12 = {
    "params": [{"shape": (1, 1, 64, 64, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_AVG_H_NOT_EQ_WINH_AND_W_EQ_WINW_DATA_MODE_ZERO",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 64), "stride": (1, 1), "pooling_mode": "AVG", "padding_mode": "SAME",
                        "pad": (2, 0, 2, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case13 = {
    "params": [{"shape": (1, 1, 128, 128, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_AVG_STRIDEH_LARGE_THAN_63_DATA_MODE_ONE",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (64, 64), "pooling_mode": "AVG", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case14 = {
    "params": [{"shape": (1, 1, 10, 10, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_GAP_TO_AVG",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (14, 14), "stride": (64, 64), "pooling_mode": "GAP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case15 = {
    "params": [{"shape": (1, 1, 64, 64, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_DATA_MODE_ZERO_CEIL_MODE_ZERO",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (1, 1), "stride": (1, 1), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (1, 1, 1, 1), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 1,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case16 = {
    "params": [{"shape": (1, 1, 64, 64, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_L1_FUSION_TYPE_ZERO_SPLIT_INDEX_3",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (1, 1), "stride": (1, 1), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 1,
                        "fusion_params": {"out_shape": (1, 1, 64, 64, 16), "l1_fusion_type": 0,
                                          "in_split_index": 3},
                        "impl_mode": "high_performance"}
}

case17 = {
    "params": [{"shape": (1, 1, 64, 64, 16), "dtype": "float16"},
               {"shape": (1, 1, 64, 64, 16), "dtype": "float16"}
               ],
    "case_name": "test_pooling2d_MAX_L1_FUSION_TYPE_ZERO_SPLIT_INDEX_1",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (1, 1), "stride": (1, 1), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 1, "ceil_mode": 1,
                        "fusion_params": {"out_shape": (1, 1, 64, 64, 16), "l1_fusion_type": 0,
                                          "in_split_index": 1},
                        "impl_mode": "high_performance"}
}

case18 = {
    "params": [{"shape": (1, 4, 16, 16, 16), "dtype": "float16"},
               {"shape": (1, 1, 1, 1, 1), "dtype": "float16"},
               ],
    "case_name": "test_pooling2d_gmp_18",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (2, 2), "pooling_mode": "GAP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

compile_case_list = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8,
    case9,
    case10,
    case11,
    case12,
    case13,
    case14,
    case15,
    case16,
    case17,
    case18
]
for item in compile_case_list:
    ut_case.add_case(case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
