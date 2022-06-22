# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_multi_compute_brc(x, y, z, kernel_name="dsl_multi_compute_brc"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            cast0 = tbe.dsl.cast_to(data1, "float16")
            cast1 = tbe.dsl.cast_to(data2, "float16")
            shape_x = shape_util.shape_to_list(shape_x)
            shape_y = shape_util.shape_to_list(shape_y)
            _, _, shape_max = shape_util.unify_broadcast_shapes([shape_x, shape_y])
            brc1 = tbe.dsl.broadcast(cast0, shape_max)
            brc2 = tbe.dsl.broadcast(cast1, shape_max)
            add = tbe.dsl.vadd(brc1, brc2)
            res = tbe.dsl.cast_to(add, "int8")

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("mbroadcast", "broadcast_schedule.test_dynamic_broadcast_schedule_multi_compute_impl", "dsl_multi_compute_brc")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_schedule_multi_compute_brc_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (1, 16),
        "dtype": "int8",
        "range": [(1, 1), (16, 16)]
    }, {
        "shape": (1880, 16),
        "dtype": "int8",
        "range": [(1880, 1880), (16, 16)]
    }, {
        "shape": (1880, 16),
        "dtype": "int8",
        "range": [(1880, 1880), (16, 16)]
    }],
    "case_name":
        "test_dynamic_broadcast_schedule_multi_compute_brc_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310P3"], case1)
ut_case.add_case(["Ascend910A", "Ascend310P3"], case2)
