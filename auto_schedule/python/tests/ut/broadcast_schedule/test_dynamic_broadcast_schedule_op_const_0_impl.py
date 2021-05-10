# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_dync_cbroadcast(x, y, z, kernel_name="dsl_dync_cbroadcast"):
    input_dtype = x.get("dtype")

    extra_params = {"disable_optimization": True}
    ins = tbe.dsl.classify([x, y], "broadcast", extra_params)
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            _, shape = shape_util.variable_shape([x, y])
            data1 = tvm.const(10, dtype=input_dtype)
            res = tbe.dsl.broadcast(data1, shape)

            tensors.append([res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("cbroadcast", "broadcast_schedule.test_dynamic_broadcast_schedule_op_const_0_impl", "dsl_dync_cbroadcast")

case1 = {
    "params": [{
        "shape": (1,),
        "dtype": "float16",
        "range": [(1, 1)]
    }, {
        "shape": (20, ),
        "dtype": "float16",
        "range": [(20, 20)]
    }, {
        "shape": (20, ),
        "dtype": "float16",
        "range": [(20, 20)]
    }],
    "case_name":
        "test_dynamic_broadcast_schedule_op_const_0_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (1,),
        "dtype": "float16",
        "range": [(1, 1)]
    }, {
        "shape": (0, ),
        "dtype": "float16",
        "range": [(0, 0)]
    }, {
        "shape": (0, ),
        "dtype": "float16",
        "range": [(0, 0)]
    }],
    "case_name":
        "test_dynamic_broadcast_schedule_op_const_0_impl_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case2)
