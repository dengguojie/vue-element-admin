# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common import context as tbe_context
from tbe.dsl.base import operation

@register_operator("SoftmaxV2", pattern="Softmax")
def dsl_dync_softmax(input_x, y, axis=-1, kernel_name="dsl_dync_softmax"):
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    if not isinstance(axis, int):
        axis = list(axis)
    tbe_context.get_context().add_compile_info("ori_axis", axis)
    tbe_context.get_context().add_compile_info("kernel_name", "SoftmaxV2")
    if isinstance(axis, int):
        axis = [axis]
    
    with tbe.dsl.compute():
        new_shape = []
        a = operation.var("a")
        new_shape.append(a)
        b = operation.var("b")
        new_shape.append(b)
        axis = [1]
        
        data_input = tvm.placeholder(new_shape, dtype=dtype, name="data")

        a_shape = shape_util.shape_to_list(data_input.shape)
        a_dtype = data_input.dtype
        axis = list(axis)
        has_improve_precision = False
        if a_dtype == "float16":
            has_improve_precision = True
            data_input1 = tbe.dsl.cast_to(data_input, "float32")
            data_max = tbe.dsl.reduce_max(data_input1, axis=axis, keepdims=True)
        else:
            data_max = tbe.dsl.reduce_max(data_input1, axis=axis, keepdims=True)

        data_max = tbe.dsl.broadcast(data_max, a_shape)
        data_subtrac = tbe.dsl.vsub(data_input1, data_max)
        data_exp = tbe.dsl.vexp(data_subtrac)
        data_expsum = tbe.dsl.reduce_sum(data_exp, axis, keepdims=True)
        data_expsum = tbe.dsl.broadcast(data_expsum, a_shape)
        output = tbe.dsl.vdiv(data_exp, data_expsum)
        if has_improve_precision and a_dtype == "float16":
            output = tbe.dsl.cast_to(output, "float16")

    schedules = []
    with tvm.target.cce():
        sch = tbe.dsl.auto_schedule(output)
    schedules.append(sch)
    tensor_list = [data_input, output]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("SoftmaxV2", "softmax.test_dynamic_softmax_schedule_op_impl", "dsl_dync_softmax")

case1 = {
    "params": [{
        "shape": (-1,-1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, -1],
    "case_name":
        "test_dync_softmax_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)