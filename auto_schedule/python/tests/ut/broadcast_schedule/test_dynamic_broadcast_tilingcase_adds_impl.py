# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.platform import platform_info


def dsl_dync_vadds(x, y, z, kernel_name="dsl_dync_vadds"):
    dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=dtype)

            shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="input_x",
                                                                  param_name_input2="input_y")
            input_x_fp32 = data1
            input_y_fp32 = data2

            input_x_fp32 = tbe.dsl.broadcast(input_x_fp32, shape)
            input_y_fp32 = tbe.dsl.broadcast(input_y_fp32, shape)
            input_z = tbe.dsl.broadcast(tvm.const(10, dtype=dtype), shape)

            add = tbe.dsl.vadd(input_x_fp32, input_z)
            res = tbe.dsl.vmul(add, input_y_fp32)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vadds", "broadcast_schedule.test_dynamic_broadcast_tilingcase_adds_impl", "dsl_dync_vadds")
case1 = {
    "params": [{
        "shape": (1, -1),
        "dtype": "int32",
        "range": [(1, 1), (1, 1), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(2, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "int32",
        "range": [(2, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_adds_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)
