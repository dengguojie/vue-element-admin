# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_dync_btvadd(x, y, z, kernel_name="dsl_dync_btvadd"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)

            shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
            input1 = tbe.dsl.broadcast(data1, shape_max)
            input2 = tbe.dsl.broadcast(data2, shape_max)
            add = tbe.dsl.vadd(input1, input2)
            scalar = tbe.dsl.var("scalar", dtype=input_dtype)
            res = tbe.dsl.vadds(add, scalar)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("btvadd", "broadcast_tilingcase.test_dynamic_broadcast_tilingcase_op_impl", "dsl_dync_btvadd")
case1 = {
    "params": [{
        "shape": (-1,),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "range": [(1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_op_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}
ut_case.add_case(["Ascend910A", "Ascend710"], case1)
