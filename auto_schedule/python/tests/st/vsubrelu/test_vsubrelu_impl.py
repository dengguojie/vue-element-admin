# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from te.platform.cce_conf import te_set_version


warnings.filterwarnings("ignore")


def dsl_vsubrelu(x, y, _, kernel_name='dsl_vsubrelu'):
    input_shape1 = x.get("shape")
    input_dtype1 = x.get("dtype")
    input_shape2 = y.get("shape")
    input_dtype2 = y.get("dtype")
    data1 = tvm.placeholder(input_shape1, name='data1', dtype=input_dtype1)
    data2 = tvm.placeholder(input_shape2, name='data2', dtype=input_dtype2)
    res = tbe.vsubrelu(data1, data2)

    tensor_list = [data1, data2, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vsubrelu", "vsubrelu.test_vsubrelu_impl", "dsl_vsubrelu")


def test_rhs_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vsubrelu(input1, 5)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_lhs_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vsubrelu(5, input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_dtype_in_not_same(soc):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float32")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        te_set_version("Ascend710")
        tbe.vsubrelu(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
        te_set_version(soc)
    return True


test_func_list = [
    test_lhs_in_not_tensor,
    test_rhs_in_not_tensor,
    test_dtype_in_not_same
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (1, 1, 1, 152), "dtype": "float16", "format": "ND"},
               {"shape": (1, 1, 1, 152), "dtype": "float16", "format": "ND"},
               {"shape": (1, 1, 1, 152), "dtype": "float16", "format": "ND"}
               ],
    "case_name": "test_vsubrelu_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (1, 1, 1056, 152), "dtype": "float16", "format": "ND"},
               {"shape": (1, 1, 1056, 152), "dtype": "float16", "format": "ND"},
               {"shape": (1, 1, 1056, 152), "dtype": "float16", "format": "ND"}],
    "case_name": "test_vsubrelu_2",
    "expect": "success",
    "support_expect": True
}

compile_case = {
    "1": [case1, "Ascend710"],
    "2": [case2, None]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, y, _):
    x_value = x.get("value")
    y_value = y.get("value")
    res =  np.maximum(np.subtract(x_value, y_value), 0)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vsubrelu_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 128, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 128, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 128, 128), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vsubrelu_precision_2",
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
