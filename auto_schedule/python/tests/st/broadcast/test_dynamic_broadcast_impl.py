# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from copy import deepcopy

@register_operator("broadcast")
def dsl_dynamic_broadcast(x, y, z, kernel_name="dsl_dynamic_broadcast"):
    input_dtype = x.get("dtype")

    extra_params = {"disable_optimization":True}
    ins = tbe.dsl.classify([x, y], "broadcast", extra_params)
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = tbe.dsl.broadcast(data1, data2.shape)
            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("broadcast", "broadcast.test_dynamic_broadcast_impl", "dsl_dynamic_broadcast")

case1 = {
    "params": [{
        "shape": (-1,),
        "dtype": "float32",
        "range": [(1, None),]
    }, {
        "shape": (60, -1),
        "dtype": "float32",
        "range": [(60, 60), (1, None)]
    }, {
        "shape": (60, -1),
        "dtype": "float32",
        "range": [(60, 60), (1, None)]
    }],
    "case_name":
        "test_dync_broadcast_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (1, -1),
        "dtype": "float16",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (2, -1),
        "dtype": "float16",
        "range": [(2, 2), (1, None)]
    },  {
        "shape": (2, -1),
        "dtype": "float16",
        "range": [(2, 2), (1, None)]
    }],
    "case_name":
        "test_dync_broadcast_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)


def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_shape = y.get("run_shape")
    res = np.broadcast_to(x_value, y_shape)

    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (-1, 1),
                "dtype": "float32",
                "range": [(1, 200), (1, 1)],
                "run_shape": (1, 1),
                "param_type": "input"
            },
            {
                "shape": (-1, 34),
                "dtype": "float32",
                "range": [(1, 1000), (34, 34)],
                "run_shape": (512, 34),
                "param_type": "input"
            }, {
                "shape": (-1, 34),
                "dtype": "float32",
                "range": [(1, 1000), (34, 34)],
                "run_shape": (512, 34),
                "param_type": "output"
            }
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_broadcast_prec_01"
    })

ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (-1, 1),
                "dtype": "float16",
                "range": [(1, 200), (1, 1)],
                "run_shape": (1, 1),
                "param_type": "input"
            },
            {
                "shape": (-1, 100),
                "dtype": "float16",
                "range": [(1, 100), (100, 100)],
                "run_shape": (66, 100),
                "param_type": "input"
            },
            {
                "shape": (-1, 100),
                "dtype": "float16",
                "range": [(1, 100), (100, 100)],
                "run_shape": (66, 100),
                "param_type": "output"
            }
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_broadcast_prec_02"
    })

ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (1, -1),
                "dtype": "float16",
                "range": [(1, 1), (1, 200)],
                "run_shape": (1, 1),
                "param_type": "input"
            },
            {
                "shape": (101, -1),
                "dtype": "float16",
                "range": [(101, 101), (1, 200)],
                "run_shape": (101, 60),
                "param_type": "input"
            },
            {
                "shape": (101, -1),
                "dtype": "float16",
                "range": [(101, 101), (1, 200)],
                "run_shape": (101, 60),
                "param_type": "output"
            }
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_broadcast_prec_03"
    })
