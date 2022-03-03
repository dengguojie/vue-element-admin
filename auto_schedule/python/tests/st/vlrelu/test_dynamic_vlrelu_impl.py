# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vlrelu")
def dsl_dynamic_vlrelu(x, y, value, kernel_name="dsl_dynamic_vlrelu"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.vlrelu(data1, tvm.const(value,dtype=input_dtype))

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vlrelu", "vlrelu.test_dynamic_vlrelu_impl", "dsl_dynamic_vlrelu")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, 0],
    "case_name":
        "test_dync_vlrelu_1",
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
    }, 0],
    "case_name":
        "test_dync_vlrelu_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (-1, 32, 199),
        "dtype": "int32",
        "range": [(1, None), (32, 32), (199, 199)]
    }, {
        "shape": (-1, 32, 199),
        "dtype": "int32",
        "range": [(1, None), (32, 32), (199, 199)]
    }, 0],
    "case_name":
        "test_dync_vlrelu_3",
    "expect":
        "success",
    "support_expect":
        True
}


ut_case.add_case(["Ascend920A", "Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend920A", "Ascend910A", "Ascend710"], case2)
ut_case.add_case(["Ascend910A"], case3)


def calc_expect_func(x, y, value):
    x_value = x.get("value")
    res = np.where(x_value>0, x_value, value*x_value)
    return (res, )


ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
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
        0
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vlrelu_prec_01"
    })

ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        0
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_vlrelu_prec_02"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (46, -1, 33),
                "dtype": "int32",
                "range": [(46, 46), (1, 100), (33, 33)],
                "run_shape": (46, 25, 33),
                "param_type": "input"
            },
            {
                "shape": (46, -1, 33),
                "dtype": "int32",
                "range": [(46, 46), (1, 100), (33, 33)],
                "run_shape": (46, 25, 33),
                "param_type": "output"
            },
        0
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_vlrelu_prec_03"
    })
