# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("vaddrelu")
def dsl_dync_vaddrelu(x, y, z, kernel_name="dsl_dync_vaddrelu"):
    input_dtype = x.get("dtype")

    ins = classify([x, y], "elewise")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = tbe.dsl.vaddrelu(data1, data2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vaddrelu", "vaddrelu.test_dynamic_vaddrelu_impl", "dsl_dync_vaddrelu")

case1 = {
    "params": [{
        "shape": (5, -1, 16, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16), (16, 16)]
    }, {
        "shape": (5, -1, 16, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16), (16, 16)]
    }, {
        "shape": (5, -1, 16, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16), (16, 16)]
    }],
    "case_name":
        "test_dync_vaddrelu_1",
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
    }],
    "case_name":
        "test_dync_vaddrelu_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (-1, 6400),
        "dtype": "int16",
        "range": [(1, 256), (6400, 6400)]
    }, {
        "shape": (-1, 6400),
        "dtype": "int16",
        "range": [(1, 256), (6400, 6400)]
    }, {
        "shape": (-1, 6400),
        "dtype": "int16",
        "range": [(1, 256), (6400, 6400)]
    }],
    "case_name":
        "test_dync_vaddrelu_3",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case("all", case1)
ut_case.add_case(["Ascend710"], case2)
ut_case.add_case(["Ascend710"], case3)


def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_value = y.get("value")
    add_res = np.add(x_value, y_value)
    res = np.maximum(add_res, 0)
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
                "param_type": "output"
            },
        ],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001),
        "case_name":
            "test_dync_vaddrelu_prec_01"
    })
