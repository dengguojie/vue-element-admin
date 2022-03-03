# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.common import context as tbe_context
from tbe.dsl import classify


@register_operator("multi_outputs")
def dsl_dync_multi_outputs(x, z, mask, kernel_name="dsl_multi_outputs"):

    """
    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    Algorithm: relu_v2

    Parameters:

    x: the dict of input data, support float16, float32, int8, int32, uint8

    y: the dict of output

    mask: the dict of mask_output

    kernel_name: cce kernel name, default value is "relu_v2".

    Returns
    -------
    None
    """

    dtype = x.get("dtype").lower()

    ins = classify([x], "elewise")
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([_x])

            input_data = tvm.placeholder(shape_x[0], dtype, "input_data")
            data_res = tbe.dsl.vrelu(input_data)

            res = tbe.dsl.cast_to(data_res, dtype)
            res_mask = tbe.dsl.vcmp(input_data, 0, "gt", "bit")

            tensors.append([input_data, res, res_mask])
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule([res, res_mask])
        schedules.append(sch)

    # tensor list in list
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)
    tbe_context.get_context().add_compile_info("ops_compile_info", "ops_compile_info")


ut_case = OpUT("multi_outputs", "multi_outputs.test_dynamic_multi_outputs_impl", "dsl_dync_multi_outputs")

case1 = {
    "params": [{
        "shape": (-1, 16),
        "ori_shape": (-1, 16),
        "dtype": "float16",
        "range": [(1, 10), (16, 16)],
        "format": "ND",
        "ori_format": "ND",
    }, {
        "shape": (-1, 16),
        "ori_shape": (-1, 16),
        "dtype": "float16",
        "range": [(1, 10), (16, 16)],
        "format": "ND",
        "ori_format": "ND",
    }, {
        "shape": (-1, 16),
        "ori_shape": (-1, 16),
        "dtype": "float16",
        "range": [(1, 10), (16, 16)],
        "format": "ND",
        "ori_format": "ND",
    }],
    "case_name":
        "test_dynamic_multi_outputs_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)


