# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.platform import platform_info


def dsl_dync_vadd(x, y, z, kernel_name="dsl_dync_vadd"):
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

            res = tbe.dsl.vadd(input_x_fp32, input_y_fp32)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vadd", "broadcast_schedule.test_dynamic_broadcast_tilingcase_add_impl", "dsl_dync_vadd")
case1 = {
    "params": [{
        "shape": (64, 48985, 325),
        "dtype": "float16",
        "range": [(64, 64), (48985, 48985), (325, 325)]
    }, {
        "shape": (64, -1, 325),
        "dtype": "float16",
        "range": [(64, 64), (1, None), (325, 325)]
    }, {
        "shape": (64, 48985, 325),
        "dtype": "float16",
        "range": [(64, 64), (48985, 48985), (325, 325)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_add_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (3, 12, 5),
        "dtype": "float16",
        "range": [(3, 3), (12, 12), (5, 5)]
    }, {
        "shape": (3, -1, 5),
        "dtype": "float16",
        "range": [(3, 3), (1, None), (5, 5)]
    }, {
        "shape": (3, 12, 5),
        "dtype": "float16",
        "range": [(3, 3), (12, 12), (5, 5)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_add_impl_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (3, 12, 32880),
        "dtype": "float16",
        "range": [(3, 3), (12, 12), (32880, 32880)]
    }, {
        "shape": (3, -1, 32880),
        "dtype": "float16",
        "range": [(3, 3), (1, None), (32880, 32880)]
    }, {
        "shape": (3, 12, 32880),
        "dtype": "float16",
        "range": [(3, 3), (12, 12), (32880, 32880)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_add_impl_3",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (-1, 1, 30700),
        "dtype": "float16",
        "range": [(1, None), (1, 1), (30700, 30700)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (2, None), (1, 30700)]
    }, {
        "shape": (-1, -1, 30700),
        "dtype": "float16",
        "range": [(2, None), (1, None), (30700, 30700)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_add_impl_4",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {
    "params": [{
        "shape": (-1, 1, -1),
        "dtype": "float16",
        "range": [(2, None), (1, 1), (2, None)]
    }, {
        "shape": (1, -1, 1),
        "dtype": "float16",
        "range": [(1, 1), (2, None), (1, 1)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(2, None), (2, None), (2, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_add_impl_5",
    "expect":
        "success",
    "support_expect":
        True
}

case6 = {
    "params": [{
        "shape": (2, 325, 48985),
        "dtype": "float16",
        "range": [(2, 2), (325, 325), (48985, 48985)]
    }, {
        "shape": (2, -1, 48985),
        "dtype": "float16",
        "range": [(2, 2), (1, None), (48985, 48985)]
    }, {
        "shape": (2, 325, 48985),
        "dtype": "float16",
        "range": [(2, 2), (325, 325), (48985, 48985)]
    }],
    "case_name":
        "test_dynamic_broadcast_tilingcase_add_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case3)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case4)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case5)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case6)
