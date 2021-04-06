# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe


warnings.filterwarnings("ignore")


def dsl_unsorted_segment_min(x, _, segment_ids, num_segments, init_value=0, kernel_name='dsl_unsorted_segment_min'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    if not isinstance(segment_ids, list):
        shape = segment_ids.get("value")
        data2 = tvm.placeholder((len(shape), ), name='data2', dtype="int32")
        tensor_list = [data1, data2]
    else:
        data2 = segment_ids
        tensor_list = [data1]
    res = tbe.unsorted_segment_min(data1, data2, num_segments, init_value)

    tensor_list.append(res)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("unsorted_segment_min", "unsorted_segment_min.test_unsorted_segment_min_impl",
               "dsl_unsorted_segment_min")


def test_num_segments_is_not_int(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        tbe.unsorted_segment_min(input1, [1, 1], 2.0)
    except RuntimeError as e:
        print(e)
    return True


def test_init_value_is_not_int_float(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        tbe.unsorted_segment_min(input1, [1, 1], 2, {})
    except RuntimeError as e:
        print(e)
    return True


def test_len_segment_ids_is_not_equal_shape_zero(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        tbe.unsorted_segment_min(input1, [1, 1, 3], 2)
    except RuntimeError as e:
        print(e)
    return True


def test_max_segment_ids_plus_1_large_than_num_segments(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        tbe.unsorted_segment_min(input1, [1], 0)
    except RuntimeError as e:
        print(e)
    return True


def test_max_segment_ids_is_tensor_dtype(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        segment_ids = tvm.placeholder((1,), name="input2", dtype="int8")
        tbe.unsorted_segment_min(input1, segment_ids, 0)
    except RuntimeError as e:
        print(e)
    return True


def test_max_segment_ids_is_tensor_num_segments_is_not_int(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        segment_ids = tvm.placeholder((1,), name="input2", dtype="int32")
        tbe.unsorted_segment_min(input1, segment_ids, 2.0)
    except RuntimeError as e:
        print(e)
    return True


def test_max_segment_ids_is_tensor_shape_zero_not_equal(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        segment_ids = tvm.placeholder((2,), name="input2", dtype="int32")
        tbe.unsorted_segment_min(input1, segment_ids, 1)
    except RuntimeError as e:
        print(e)
    return True


def test_max_segment_ids_is_tensor_len_segment_ids_shape_not_equal_one(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        segment_ids = tvm.placeholder((1, 1), name="input2", dtype="int32")
        tbe.unsorted_segment_min(input1, segment_ids, 1)
    except RuntimeError as e:
        print(e)
    return True


def test_max_segment_ids_is_tensor_input_not_tensor(_):
    try:
        input1 = tvm.const(1, dtype="float16")
        segment_ids = tvm.placeholder((1, 1), name="input2", dtype="int32")
        tbe.unsorted_segment_min(input1, segment_ids, 1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_num_segments_is_not_int,
    test_init_value_is_not_int_float,
    test_len_segment_ids_is_not_equal_shape_zero,
    test_max_segment_ids_plus_1_large_than_num_segments,
    test_max_segment_ids_is_tensor_dtype,
    test_max_segment_ids_is_tensor_num_segments_is_not_int,
    test_max_segment_ids_is_tensor_shape_zero_not_equal,
    test_max_segment_ids_is_tensor_len_segment_ids_shape_not_equal_one,
    test_max_segment_ids_is_tensor_input_not_tensor,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
               {"shape": (6, 1024), "dtype": "float16", "format": "ND"},
               [1, 3, 5, 3, 1],
               8
               ],
    "case_name": "test_unsorted_segment_min_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
               {"shape": (6, 1024), "dtype": "float16", "format": "ND"},
               [-1, -1, -4, -5, -5],
               6
               ],
    "case_name": "test_unsorted_segment_min_max_segment_less_than_zero",
    "expect": "success",
    "support_expect": True
}


case3 = {
    "params": [{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
               {"shape": (6, 1024), "dtype": "float16", "format": "ND"},
               {"value": [1, 3, 5, 3, 1]},
               6
               ],
    "case_name": "test_unsorted_segment_min_segment_is_tensor",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,
    case2,
    case3,
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, _, segment_ids, num_segment, init_value):
    x_value = x.get("value")
    x_shape = x.get("shape")
    x_data_type = x.get("dtype")
    out_shape = list(x_shape)
    out_shape[0] = num_segment
    result = np.full(out_shape, init_value, dtype=x_data_type)
    last_index = segment_ids[0]
    result[last_index] = x_value[0]
    for in_index, out_index in enumerate(segment_ids[1:], start=1):
        if last_index != out_index:
            last_index = out_index
            result[last_index] = 65504
        result[last_index] = np.minimum(result[last_index], x_value[in_index])
    return (result, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (5, 100), "dtype": "float16", "param_type": "input"},
                   {"shape": (6, 100), "dtype": "float16", "param_type": "output"},
                   [1, 1, 4, 5, 5],
                   6,
                   0
                   ],
        "case_name": "test_unsorted_segment_min_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (2, 80), "dtype": "int32", "param_type": "input"},
                   {"shape": (16, 80), "dtype": "int32", "param_type": "output"},
                   [4, 13],
                   16,
                   1
                   ],
        "case_name": "test_unsorted_segment_min_precision_2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 17, 19, 64), "dtype": "float16", "param_type": "input"},
                   {"shape": (16, 17, 19, 64), "dtype": "float16", "param_type": "output"},
                   [11, 11, 10, 9],
                   16,
                   -1
                   ],
        "case_name": "test_unsorted_segment_min_precision_3",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    # ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
