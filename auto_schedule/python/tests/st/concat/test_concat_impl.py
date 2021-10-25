# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_concat(x, _, axis, kernel_name='dsl_concat'):
    input1_shape = x[0].get("shape")
    input1_dtype = x[0].get("dtype")
    data1 = tvm.placeholder(input1_shape, name='data1', dtype=input1_dtype)
    raw_tensor = [data1]
    tensor_list = [data1]
    if len(x) == 2:
        input2_shape = x[1].get("shape")
        input2_dtype = x[1].get("dtype")
        data2 = tvm.placeholder(input2_shape, name='data2', dtype=input2_dtype)
        raw_tensor.append(data2)
        tensor_list.append(data2)
    res = tbe.concat(raw_tensor, axis)

    tensor_list.append(res)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("concat", "concat.test_concat_impl", "dsl_concat")


def test_axis_is_not_int(_):
    try:
        input1 = tvm.placeholder((16, ), dtype="float16", name="input1")
        input2 = tvm.placeholder((16, ), dtype="float16", name="input2")
        tbe.concat([input1, input2], 1.0)
    except RuntimeError as e:
        print(e)
    return True


def test_axis_large_than_len_tensor0_shape(_):
    try:
        input1 = tvm.placeholder((16, ), dtype="float16", name="input1")
        input2 = tvm.placeholder((16, ), dtype="float16", name="input2")
        tbe.concat([input1, input2], 3)
    except RuntimeError as e:
        print(e)
    return True


def test_tensor0_is_not_tensor(_):
    try:
        input1 = tvm.const(16, dtype="float16")
        input2 = tvm.placeholder((16, ), dtype="float16", name="input2")
        tbe.concat([input1, input2], 3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor1_is_not_tensor(_):
    try:
        input1 = tvm.placeholder((16, ), dtype="float16", name="input2")
        input2 = tvm.const(16, dtype="float16")
        tbe.concat([input1, input2], 0)
    except RuntimeError as e:
        print(e)
    return True


def test_tensor0_shape_value_less_than_zero(_):
    try:
        input1 = tvm.placeholder((-1, ), dtype="float16", name="input1")
        input2 = tvm.placeholder((16, ), dtype="float16", name="input2")
        tbe.concat([input1, input2], 0)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor0_dim_is_not_same_tensor1_dim(_):
    try:
        input1 = tvm.placeholder((16, ), dtype="float16", name="input1")
        input2 = tvm.placeholder((16, 16), dtype="float32", name="input2")
        tbe.concat([input1, input2], 0)
    except RuntimeError as e:
        print(e)
    return True


def test_tensor0_shape_value_is_not_same_tensor1_shape_value(_):
    try:
        input1 = tvm.placeholder((16, 16), dtype="float16", name="input1")
        input2 = tvm.placeholder((16, 17), dtype="float16", name="input2")
        tbe.concat([input1, input2], 0)
    except RuntimeError as e:
        print(e)
    return True


test_func_list = [
    test_axis_is_not_int,
    test_axis_large_than_len_tensor0_shape,
    test_tensor0_is_not_tensor,
    test_tensor1_is_not_tensor,
    test_tensor0_shape_value_less_than_zero,
    test_tensor0_dim_is_not_same_tensor1_dim,
    test_tensor0_shape_value_is_not_same_tensor1_shape_value
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {"params": [[{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
                     {"shape": (5, 1024), "dtype": "float16", "format": "ND"}
                     ],
                    {"shape": (10, 1024), "dtype": "float16", "format": "ND"},
                    0
                    ],
         "case_name": "test_concat_1",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [[{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
                     {"shape": (5, 1024), "dtype": "float16", "format": "ND"}
                     ],
                    {"shape": (10, 1024), "dtype": "float16", "format": "ND"},
                    -1
                    ],
         "case_name": "test_concat_axis_less_than_zero",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [[{"shape": (5, 1024), "dtype": "float16", "format": "ND"}
                     ],
                    {"shape": (10, 1024), "dtype": "float16", "format": "ND"},
                    1
                    ],
         "case_name": "test_concat_only_one_tensor",
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


def calc_expect_func(x, _, axis):
    x1_value = x[0].get("value")
    x2_value = x[1].get("value")
    output = np.concatenate((x1_value, x2_value), axis=axis)
    return output


ut_case.add_precision_case(
    "all", {
        "params": [[{"shape": (1, 7, 7, 512), "dtype": "float16", "param_type": "input"},
                    {"shape": (1, 7, 7, 32), "dtype": "float16", "param_type": "input"},
                    ],
                   {"shape": (1, 7, 7, 544), "dtype": "float16", "param_type": "output"},
                   -1
                   ],
        "case_name": "test_concat_precision_Densenet121",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [[{"shape": (24, 2), "dtype": "float16", "param_type": "input"},
                    {"shape": (24, 2), "dtype": "float16", "param_type": "input"},
                    ],
                   {"shape": (24, 4), "dtype": "float16", "param_type": "output"},
                   -1
                   ],
        "case_name": "test_concat_precision_frozen_inference",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [[{"shape": (1, 13, 13, 192), "dtype": "float16", "param_type": "input"},
                    {"shape": (1, 13, 13, 192), "dtype": "float16", "param_type": "input"},
                    ],
                   {"shape": (1, 26, 13, 192), "dtype": "float16", "param_type": "output"},
                   1
                   ],
        "case_name": "test_concat_precision_squeeze",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
