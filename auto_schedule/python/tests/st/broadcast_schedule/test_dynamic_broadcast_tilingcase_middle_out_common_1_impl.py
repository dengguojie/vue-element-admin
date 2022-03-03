# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_dync_vmulout(x, y, z, kernel_name="dsl_dync_vmulout"):
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
            input_x_fp32 = tbe.dsl.broadcast(data1, shape)
            input_y_fp32 = tbe.dsl.broadcast(data2, shape)

            sub = tbe.dsl.vsub(input_x_fp32, input_y_fp32)
            res1 = tbe.dsl.vmuls(data1, tvm.const(1, dtype=dtype))
            res = tbe.dsl.vadd(sub, input_y_fp32)

            tensors.append((data1, data2, res1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule([res1, res])
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vmiddleout", "broadcast_schedule.test_dynamic_broadcast_tilingcase_middle_out_common_1_impl", "dsl_dync_vmulout")
case1 = {
    "params": [{
        "shape": (-1, 1, 8192),
        "dtype": "float16",
        "range": [(1, None), (1, 1), (8192, 8192)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (2, None), (1, 8192)]
    }, {
        "shape": (-1, -1, 8192),
        "dtype": "float16",
        "range": [(2, None), (1, None), (8192, 8192)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_middle_out_common_1_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}
ut_case.add_case(["Ascend910A", "Ascend710"], case1)
