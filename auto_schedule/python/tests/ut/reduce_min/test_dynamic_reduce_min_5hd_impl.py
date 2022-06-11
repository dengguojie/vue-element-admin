# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("reduce_min_5hd")
def dsl_dync_reduce_min_5hd(x, y, axis, keepdims=False, kernel_name="dsl_dync_reduce_min_5hd"):
    input_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = 'before'

    if axis[0] == -2:
        input_axis = {"shape": [-1, ], 'dtype': 'int32', "rel_pos_to_reduce": "axis"}
    elif axis[0] == -1:
        input_axis = {"shape": [2, ], 'dtype': 'int32', "rel_pos_to_reduce": "axis"}
    else:
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True, "ignore_fractal_format": False})
    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.reduce_min(data1, axis.get("value"), keepdims)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("reduce_min_5hd", "reduce_min.test_dynamic_reduce_min_5hd_impl", "dsl_dync_reduce_min_5hd")

case1 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, [1, 4]],
    "case_name":
        "test_dync_reduce_min_5hd_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, [-2]],
    "case_name":
        "test_dync_reduce_min_5hd_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (32, 2, 3, 42),
        "ori_format": "NHWC",
        "shape": (32, 3, 2, 3, 16),
        "format": "NC1HWC0",
        "range": [(32, 32), (3, 3), (2, 2), (3, 3), (16, 16)]
    }, {
        "ori_shape": (32, 2, 3, 42),
        "ori_format": "NHWC",
        "shape": (32, 2, 3),
        "format": "NC1HWC0",
        "range": [(32, 32), (2, 2), (3, 3)]
    }, [1, 4]],
    "case_name":
        "test_dync_reduce_min_5hd_3",
    "case_type":
        "static",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, [-1]],
    "case_name":
        "test_dync_reduce_min_5hd_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend910B2"], case1)
ut_case.add_case(["Ascend910A", "Ascend910B2"], case2)
ut_case.add_case(["Ascend910A", "Ascend910B2"], case3)
ut_case.add_case(["Ascend910A", "Ascend910B2"], case4)
