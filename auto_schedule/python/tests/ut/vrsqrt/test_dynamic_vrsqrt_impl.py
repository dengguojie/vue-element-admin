# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vrsqrt")
def dsl_dynamic_vrsqrt(x, y, kernel_name="dsl_dynamic_vrsqrt"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.vrsqrt(data1, "high_precision")

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vrsqrt", "vrsqrt.test_dynamic_vrsqrt_impl", "dsl_dynamic_vrsqrt")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vrsqrt_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (2, -1),
        "dtype": "float16",
        "range": [(2, 2), (1, None)]
    }, {
        "shape": (2, -1),
        "dtype": "float16",
        "range": [(2, 2), (1, None)]
    }],
    "case_name":
        "test_dync_vrsqrt_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)


def calc_expect_func(x, y):
    x_value = x.get("value")
    res = np.power(x_value, -0.5)
    return (res, )

# vrsqrt instruction has precision problem
# The max error of all dtypes in all soc reaches 0.004 
ut_case.add_precision_case(
    ["Ascend910A", "Ascend310"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.004, 0.004),
        "case_name": "test_dync_vrsqrt_prec_01"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.004, 0.004),
        "case_name": "test_dync_vrsqrt_prec_02"
    })
