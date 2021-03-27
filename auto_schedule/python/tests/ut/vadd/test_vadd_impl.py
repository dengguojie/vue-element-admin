# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

from te import tvm
import te.lang.cce as tbe


def dsl_vadd(x, y, z, kernel_name='dsl_vadd'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    data2 = tvm.placeholder(input_shape, name='data2', dtype=input_dtype)
    res = tbe.vadd(data1, data2)

    tensor_list = [data1, data2, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vadd", "vadd.test_vadd_impl", "dsl_vadd")

def test_rhs_in_not_tensor(soc):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vadd(input1, 5)
    except RuntimeError as e:
        print(e.args[0]["detailed_cause"])
    return True


test_func_list = [
    test_rhs_in_not_tensor
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"}
               ],
    "case_name": "test_vadd_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (30000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (30000, 1), "dtype": "float32", "format": "ND"}],
    "case_name": "test_vadd_2",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend710"], case2)


def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.add(x_value, y_value)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vadd_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vadd_precision_yolo_person_detect",
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

