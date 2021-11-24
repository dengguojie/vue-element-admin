# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("gather_nd")
def dsl_dync_gather_nd(params, indices, output, batch_dims=0, kernel_name="dsl_dync_vadd"):

    with tbe.common.context.op_context.OpContext("dynamic"):

        ins = classify([params, indices, batch_dims], "gather_nd")
        schedules, tensors = [], []

        for params_input, indices_input, batch_dims_input in ins:
            with tbe.dsl.compute():
                params_shape, indices_shape, batch_dims = shape_util.variable_shape([params_input, indices_input, batch_dims_input], "gather")
                params_tensor = tvm.placeholder(params_shape, name='params', dtype=params_input["dtype"])
                indices_tesnor = tvm.placeholder(indices_shape, name='indices', dtype=indices_input["dtype"])
                res = tbe.dsl.gather_nd(params_tensor, indices_tesnor, batch_dims)
                tensors.append((params_tensor, indices_tesnor, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)

ut_case = OpUT("gather", "gather.test_gather_nd_impl", "dsl_dync_gather_nd")

case1 = {
    "params": [{
        "shape": (5, -1,),
        "dtype": "float16",
        "range": [(5, 5), (1, None)]
    }, {
        "shape": (-1, 1),
        "dtype": "int32",
        "range": [(1, None), (1, 1)]
    }, {
        "shape": (-1, -1,),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_gather_nd_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "int32",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }],
    "case_name":
        "test_dync_gather_nd_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["all",], case1)
ut_case.add_case(["all",], case2)
