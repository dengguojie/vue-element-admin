# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.platform import platform_info


def dsl_dync_vfloormod(x, y, z, kernel_name="dsl_dync_vfloormod"):
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
            has_improve_precision = False
            input_x_fp32 = data1
            input_y_fp32 = data2
            if platform_info.api_check_support("te.lang.cce.vdiv",
                                               "float32"):
                input_x_fp32 = tbe.dsl.cast_to(data1, "float32")
                input_y_fp32 = tbe.dsl.cast_to(data2, "float32")
                has_improve_precision = True

            input_x_fp32 = tbe.dsl.broadcast(input_x_fp32, shape)
            input_y_fp32 = tbe.dsl.broadcast(input_y_fp32, shape)

            res = tbe.dsl.vdiv(input_x_fp32, input_y_fp32)

            if platform_info.api_check_support("te.lang.cce.floor",
                                               res.dtype):
                res = tbe.dsl.floor(res)
            else:
                res = tbe.dsl.cast_to(res, "float16")
                res = tbe.dsl.floor(res)

            if dtype != "int32":
                if has_improve_precision:
                    res = tbe.dsl.cast_to(res, "float32")
                else:
                    res = tbe.dsl.cast_to(res, "float16")
                res = tbe.dsl.vmul(res, input_y_fp32)
                res = tbe.dsl.vsub(input_x_fp32, res)
                if has_improve_precision:
                    res = tbe.dsl.cast_to(res, dtype)
            else:
                x2_broad = tbe.dsl.broadcast(data2, shape)
                x1_broad = tbe.dsl.broadcast(data1, shape)
                res = tbe.dsl.vmul(res, x2_broad)
                res = tbe.dsl.vsub(x1_broad, res)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vfloormod", "broadcast_schedule.test_dynamic_broadcast_tilingcase_floormod_impl", "dsl_dync_vfloormod")
case1 = {
    "params": [{
        "shape": (1, -1),
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
        "test_dynamic_broadcast_tilingcase_floormod_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)

