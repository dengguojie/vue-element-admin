# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vaddrelu(x, y, _, kernel_name='dsl_vaddrelu'):
    input1_shape = x.get("shape")
    input1_dtype = x.get("dtype")
    input2_shape = y.get("shape")
    input2_dtype = y.get("dtype")
    data1 = tvm.placeholder(input1_shape, name='data1', dtype=input1_dtype)
    data2 = tvm.placeholder(input2_shape, name='data2', dtype=input2_dtype)
    res = tbe.vaddrelu(data1, data2)

    tensor_list = [data1, data2, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vaddrelu", "vaddrelu.test_vaddrelu_impl", "dsl_vaddrelu")


def test_lhs_is_not_tensor(_):
    try:
        input1 = tvm.const(0, dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        tbe.vaddrelu(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_rhs_is_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.const(0, dtype="float16")
        tbe.vaddrelu(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_lhs_dtype_is_not_same_rhs_dtype(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float32")
        tbe.vaddrelu(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_lhs_is_not_tensor, None],
    "2": [test_rhs_is_not_tensor, None],
    "3": [test_lhs_dtype_is_not_same_rhs_dtype, None]
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (10000, ), "dtype": "float16", "format": "ND"},
                    {"shape": (10000, ), "dtype": "float16", "format": "ND"},
                    {"shape": (10000, ), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_vaddrelu_not_support_vaddrelu",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (128, ), "dtype": "float16", "format": "ND"},
                    {"shape": (128, ), "dtype": "float16", "format": "ND"},
                    {"shape": (128, ), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_vaddrelu_support_vaddrelu",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case2, "Ascend710"]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, y, _):
    x_value = x.get("value")
    y_value = y.get("value")
    output = np.maximum(0, x_value + y_value)
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (256, 1, 1, 1), "dtype": "float16", "param_type": "input"},
                   {"shape": (256, 1, 1, 1), "dtype": "float16", "param_type": "input"},
                   {"shape": (256, 1, 1, 1), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vaddrelu_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend710", {
        "params": [{"shape": (64, 4096), "dtype": "float16", "param_type": "input"},
                   {"shape": (64, 4096), "dtype": "float16", "param_type": "input"},
                   {"shape": (64, 4096), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_vaddrelu_precision_2",
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
