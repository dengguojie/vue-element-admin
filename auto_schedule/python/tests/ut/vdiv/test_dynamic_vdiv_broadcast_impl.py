# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vdiv")
def dynamic_vdiv_broadcast(x, y, z, kernel_name="dynamic_vdiv_broadcast"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
            input_x = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            input_y = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            x_shape = shape_util.shape_to_list(input_x.shape)
            y_shape = shape_util.shape_to_list(input_y.shape)
            x_shape, y_shape, z_shape = shape_util.broadcast_shapes(x_shape, y_shape,
                                                                    param_name_input1="input_x",
                                                                    param_name_input2="input_y")
            broadcast_x = tbe.dsl.broadcast(input_x, z_shape)
            broadcast_y = tbe.dsl.broadcast(input_y, z_shape)
            res = tbe.dsl.vdiv(broadcast_x, broadcast_y)

            tensors.append((input_x, input_y, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vdiv", "vdiv.test_dynamic_vdiv_broadcast_impl", "dynamic_vdiv_broadcast")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vdiv_broadcast_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)