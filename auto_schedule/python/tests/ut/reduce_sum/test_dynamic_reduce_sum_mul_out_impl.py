# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("UT_reduce_sum_mul_out")
def dsl_dync_reduce_sum_mul_out(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        kernel_name = "dsl_dync_reduce_sum_mul_out"
        x = {
            "shape": (-1, -1, -1, -1),
            "dtype": "float16",
            "range": [(1, None), (1, None), (1, None), (1, None)]
        }

        y = {
            "shape": (-1, -1),
            "dtype": "float16",
            "range": [(1, None), (1, None)]
        }
        axis = [1, 3]
        keepdims = False

        input_dtype = x.get("dtype")
        x["rel_pos_to_reduce"] = 'before'
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
        ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})
        schedules, tensors = [], []

        try:
            for (x, axis) in ins:
                with tbe.dsl.compute():
                    shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
                    data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                    cast_res = tbe.dsl.cast_to(data1, "float32")
                    res = tbe.dsl.reduce_sum(data1, axis.get("value"), keepdims)
                    tensors.append([data1, cast_res, res])

                with tvm.target.cce():
                    sch = tbe.dsl.auto_schedule([cast_res, res])
                schedules.append(sch)

            config = {"name": kernel_name, "tensor_list": tensors}
            tbe.dsl.build(schedules, config)
        except RuntimeError as e:
            if e.args[0].get("errCode") == "E90003":
                return True
        return False


ut_case = OpUT("UT_reduce_sum_mul_out", "reduce_sum.test_dynamic_reduce_sum_mul_out_impl", "dsl_dync_reduce_sum_mul_out")

test_func_list = [
    dsl_dync_reduce_sum_mul_out
]

for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)