# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("vmadd")
def dsl_dync_vmadd(x1, x2, x3, y, kernel_name="dsl_dync_vmadd"):
    input_dtype = x1.get("dtype")

    ins = classify([x1, x2, x3], "elewise")
    schedules, tensors = [], []

    for (x1, x2, x3) in ins:
        with tbe.dsl.compute():
            shape_x1, shape_x2, shape_x3 = shape_util.variable_shape([x1, x2, x3])
            data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_x2, name='data2', dtype=input_dtype)
            data3 = tvm.placeholder(shape_x3, name='data3', dtype=input_dtype)
            res = tbe.dsl.vmadd(data1, data2, data3)

            tensors.append((data1, data2, data3, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vmadd", "vmadd.test_dynamic_vmadd_impl", "dsl_dync_vmadd")

case1 = {
    "params": [{
        "shape": (5, -1, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16)]
    }, {
        "shape": (5, -1, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16)]
    }, {
        "shape": (5, -1, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16)]
    },{
        "shape": (5, -1, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16)]
    }],
    "case_name":
        "test_dync_vmadd_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    }, {
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    }, {
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    },{
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    }],
    "case_name":
        "test_dync_vmadd_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)


def calc_expect_func(x1, x2, x3, y):
    x1_value = x1.get("value")
    x2_value = x2.get("value")
    x3_value = x3.get("value")
    mul_res = np.multiply(x1_value, x3_value)
    res  = np.add(mul_res, x2_value)
    return res,


ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001),
        "case_name":
            "test_dync_vmadd_prec_01"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name":
            "test_dync_vmadd_prec_02"
    })

