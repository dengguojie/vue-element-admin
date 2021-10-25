# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_reduce_max(x, _, axis, keep_dim, kernel_name='dsl_reduce_max'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.reduce_max(data1, axis, keep_dim)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("reduce_max", "reduce_max.test_reduce_max_impl", "dsl_reduce_max")


def test_axis_in_none(_):
    try:
        input1 = tvm.placeholder((128, 128), name="input1", dtype="float16")
        tbe.reduce_max(input1, [None])
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_axis_in_none
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
               {"shape": (5, ), "dtype": "float16", "format": "ND"},
               [-1],
               False
               ],
    "case_name": "test_reduce_max_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (256, 256, 256), "dtype": "float16", "format": "ND"},
               {"shape": (256, ), "dtype": "float16", "format": "ND"},
               [0, 2],
               False
               ],
    "case_name": "test_reduce_max_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{"shape": (32, 32, 256000), "dtype": "float16", "format": "ND"},
               {"shape": (32, 256000), "dtype": "float16", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_max_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [{"shape": (32, 32, 2, 128000), "dtype": "float16", "format": "ND"},
               {"shape": (32, 2, 128000), "dtype": "float16", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_max_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [{"shape": (32, 32, 256000, 128), "dtype": "float16", "format": "ND"},
               {"shape": (32, 256000, 128), "dtype": "float16", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_max_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [{"shape": (11, 11, 9991, 33331), "dtype": "float16", "format": "ND"},
               {"shape": (11, 9991, 33331), "dtype": "float16", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_max_6",
    "expect": "success",
    "support_expect": True
}

case7 = {
    "params": [{"shape": (11, 11, 141, 33331), "dtype": "float16", "format": "ND"},
               {"shape": (11, 141, 33331), "dtype": "float16", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_max_7",
    "expect": "success",
    "support_expect": True
}

case8 = {
    "params": [{"shape": (32, 32, 2, 99991), "dtype": "float16", "format": "ND"},
               {"shape": (32, 2, 99991), "dtype": "float16", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_max_8",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, _, axis, keep_dims):
    x_value = x.get("value")
    res = np.maximum.reduce(x_value, axis=axis, keepdims=keep_dims)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (7, 5, 13, 8, 7), "dtype": "float16", "param_type": "input"},
                   {"shape": (7, 5, 8, 7), "dtype": "float16", "param_type": "output"},
                   (2, ),
                   False
                   ],
        "case_name": "test_reduce_max_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (16, 128), "dtype": "int32", "param_type": "input"},
                   {"shape": (16,), "dtype": "int32", "param_type": "output"},
                   (1, ),
                   False
                   ],
        "case_name": "test_reduce_max_precision_2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (7, 5, 13, 8, 7), "dtype": "int8", "param_type": "input"},
                   {"shape": (7, 5, 13, 8), "dtype": "int8", "param_type": "output"},
                   (4, ),
                   False
                   ],
        "case_name": "test_reduce_max_precision_3",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (11, 13, 8, 13), "dtype": "float16", "param_type": "input"},
                   {"shape": (11, 13, 1, 13), "dtype": "float16", "param_type": "output"},
                   (2, ),
                   True
                   ],
        "case_name": "test_reduce_max_precision_4",
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
