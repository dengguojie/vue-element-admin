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
    if isinstance(axis, dict):
        axis["rel_pos_to_reduce"] = 'axis'
        input_axis = axis
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

ut_case.add_case(["Ascend910", "Ascend910B2"], case1)
