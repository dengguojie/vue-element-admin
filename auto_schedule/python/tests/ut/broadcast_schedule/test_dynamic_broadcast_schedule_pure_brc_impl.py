# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_pure_brc(x, y, z, kernel_name="dsl_pure_brc"):
    input_dtype = x.get("dtype")

    extra_params = {"pure_brc": True}
    ins = tbe.dsl.classify([x, y], "broadcast", extra_params)
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.broadcast(data1, shape_y)

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("pbroadcast", "broadcast_schedule.test_dynamic_broadcast_schedule_pure_brc_impl", "dsl_pure_brc")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_schedule_pure_brc_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310P3"], case1)
