# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe


warnings.filterwarnings("ignore")


def dsl_vsub(x, y, _, kernel_name='dsl_vsub'):
    input_shape1 = x.get("shape")
    input_dtype1 = x.get("dtype")
    input_shape2 = y.get("shape")
    input_dtype2 = y.get("dtype")
    data1 = tvm.placeholder(input_shape1, name='data1', dtype=input_dtype1)
    data2 = tvm.placeholder(input_shape2, name='data2', dtype=input_dtype2)
    res = tbe.vsub(data1, data2)

    tensor_list = [data1, data2, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vsub", "vsub.test_vsub_impl", "dsl_vsub")


def test_rhs_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vsub(input1, 5)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_rhs_in_not_tensor
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (1, 1, 1, 256), "dtype": "float16", "format": "ND"},
               {"shape": (1, 1, 1, 256), "dtype": "float16", "format": "ND"},
               {"shape": (1, 1, 1, 256), "dtype": "float16", "format": "ND"}
               ],
    "case_name": "test_vsub_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (1, 1, 1056, 304), "dtype": "float32", "format": "ND"},
               {"shape": (1, 1, 1056, 304), "dtype": "float32", "format": "ND"},
               {"shape": (1, 1, 1056, 304), "dtype": "float32", "format": "ND"}],
    "case_name": "test_vsub_2",
    "expect": "success",
    "support_expect": True
}

compile_case = [
    case1,
    case2
]
for item in compile_case:
    ut_case.add_case(case=item)


def calc_expect_func(x, y, _):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.subtract(x_value, y_value)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vsub_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 128, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 128, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 128, 128), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vsub_precision_2",
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
