# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_dync_sbroadcast(x, y, z, kernel_name="dsl_dync_sbroadcast"):
    input_dtype = x.get("dtype")

    extra_params = {"disable_optimization": True}
    ins = tbe.dsl.classify([x, y], "broadcast", extra_params)
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape = shape_util.variable_shape([y])[0]
            shape_x = [1 if x["shape"][i] == 1 else shape[i] for i in range(len(shape))]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.broadcast(data1, shape)

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("sbroadcast", "broadcast_schedule.test_dynamic_broadcast_schedule_op_impl", "dsl_dync_sbroadcast")

case1 = {
    "params": [{
        "shape": (1,),
        "dtype": "float16",
        "range": [(1, 1)]
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
        "test_dynamic_broadcast_schedule_op_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)
