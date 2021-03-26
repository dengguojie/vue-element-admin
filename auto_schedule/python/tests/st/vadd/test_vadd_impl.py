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

def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.add(x_value, y_value)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (1, 16),
                "dtype": "float16",
                "param_type": "input"
            },
            {
                "shape": (1, 16),
                "dtype": "float16",
                "param_type": "input"
            },
            {
                "shape": (1, 16),
                "dtype": "float16",
                "param_type": "output"
            },
        ],
        "calc_expect_func":
        calc_expect_func,
        "precision_standard":
        precision_info.PrecisionStandard(0.001, 0.001)
    })
