import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("gather_v2")
def dsl_dynamic_gather_v2(params, indices, axis, output, batch_dims=0, kernel_name="dsl_dynamic_gather_v2"):

    with tbe.common.context.op_context.OpContext("dynamic"):

        ins = classify([params, indices, axis, batch_dims], "gather")
        schedules, tensors = [], []

        for params_input, indices_input, axis_input, batch_dims_input in ins:
            with tbe.dsl.compute():
                params_shape, indices_shape, axis_dim, batch_dims = shape_util.variable_shape([params_input, indices_input, axis_input, batch_dims_input], "gather")
                params_tensor = tvm.placeholder(params_shape, name='params', dtype=params_input["dtype"])
                indices_tesnor = tvm.placeholder(indices_shape, name='indices', dtype=indices_input["dtype"])
                axis_tensor = tvm.placeholder([1], name='axis', dtype=axis["dtype"])
                res = tbe.dsl.gather(params_tensor, indices_tesnor, axis_dim, batch_dims)
                tensors.append((params_tensor, indices_tesnor, axis_tensor, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)

ut_case = OpUT("gather", "gather.test_dynamic_gather_v2_impl", "dsl_dynamic_gather_v2")

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
        "shape": (1,),
        "dtype": "int32",
        "range": [(1, 1),]
    }, {
        "shape": (-1, -1,),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_gather_v2_1",
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
        "shape": (1,),
        "dtype": "int32",
        "range": [(1, 1),]
    }, {
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }],
    "case_name":
        "test_dync_gather_v2_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["all",], case1)
ut_case.add_case(["all",], case2)
