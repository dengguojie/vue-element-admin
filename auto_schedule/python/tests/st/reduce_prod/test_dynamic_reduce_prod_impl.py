# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("reduce_prod")
def dsl_dync_reduce_prod(x, y, axis, keepdims, kernel_name="dsl_dync_reduce_prod"):
    input_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = 'before'
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})
    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.reduce_prod(data1, axis.get("value"), keepdims)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("reduce_prod", "reduce_prod.test_dynamic_reduce_prod_impl", "dsl_dync_reduce_prod")

case1 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [1, 3], False],
    "case_name":
    "test_dync_reduce_prod_1",
    "expect":
    "success",
    "support_expect":
    True
}

case2 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [0, 2], False],
    "case_name":
    "test_dync_reduce_prod_2",
    "expect":
    "success",
    "support_expect":
    True
}


case3 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, [0, 2], False],
    "case_name":
    "test_dync_reduce_prod_2",
    "expect":
    "success",
    "support_expect":
    True
}


compile_case_list = [
    case1,
    case2,
    case3
]


for item in compile_case_list:
    ut_case.add_case(["Ascend910A", "Ascend310"], case=item)


def calc_expect_func(x, y, axis, keepdims):
    x_value = x.get("value")
    res = np.prod(x_value, axis=axis, keepdims=keepdims)
    return (res, )


ut_case.add_precision_case(
    ["Ascend310"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, None), (1, None)],
                "run_shape": (16, 16),
                "param_type": "input"
            },
            {
                "shape": (-1, ),
                "dtype": "float32",
                "range": [(1, None)],
                "run_shape": (16, ),
                "param_type": "output"
            },
            (0, ),
            False
        ],
        "calc_expect_func":
        calc_expect_func,
        "precision_standard":
        precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name":
        "test_dync_reduce_prod_prec_float32"
    })

ut_case.add_precision_case(
    ["all"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "int32",
                "range": [(1, None), (1, None)],
                "run_shape": (16, 16),
                "param_type": "input"
            },
            {
                "shape": (-1, ),
                "dtype": "int32",
                "range": [(1, None)],
                "run_shape": (16, ),
                "param_type": "output"
            },
            (0, ),
            False
        ],
        "calc_expect_func":
        calc_expect_func,
        "precision_standard":
        precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name":
        "test_dync_reduce_prod_prec_int32"
    })
