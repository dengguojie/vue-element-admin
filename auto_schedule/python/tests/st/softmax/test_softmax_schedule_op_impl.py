# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util
from te import platform as tbe_platform

@tbe_platform.fusion_manager.fusion_manager.register("softmax_grad")
def dsl_softmaxgrad(softmax, grad_softmax, grad_x, axis=-1, kernel_name="dsl_softmaxgrad"):
    shape_softmax = softmax.get("shape")
    shape_grad_softmax = grad_softmax.get("shape")
    dtype_softmax = softmax.get("dtype")

    if not isinstance(axis, int):
        axis = list(axis)
    shape_util.compare_tensor_dict_key(softmax, grad_softmax, "dtype")

    axis = shape_util.axis_check(len(shape_softmax), axis)

    check_list = ("float16", "float32")
    input_dtype = dtype_softmax.lower()
    
    shape_softmax, axis = shape_util.shape_refine(list(shape_softmax), axis)
    shape_softmax, axis = shape_util.simplify_axis_shape(shape_softmax, axis)
    shape_grad_softmax = shape_softmax
    softmax = tvm.placeholder(shape_softmax, name="softmax", dtype=input_dtype)
    grad_softmax = tvm.placeholder(shape_grad_softmax,
                                       name="grad_softmaxgrad",
                                       dtype=input_dtype)

    dtype = dtype_softmax
    shape_input1 = shape_util.shape_to_list(shape_softmax)
    shape_input2 = shape_util.shape_to_list(shape_grad_softmax)
    has_improve_precision = False
    if list(shape_input1) != list(shape_input2):
        shape_input1, shape_input2, shape =\
            shape_util.broadcast_shapes(shape_input1, shape_input2,
                                        param_name_input1="softmax",
                                        param_name_input2="grad_softmax")
        softmax = tbe.dsl.broadcast(softmax, shape, dtype)
        grad_softmax = tbe.dsl.broadcast(grad_softmax, shape, dtype)

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        softmax = tbe.dsl.cast_to(softmax, "float32")
        grad_softmax = tbe.dsl.cast_to(grad_softmax, "float32")
        has_improve_precision = True
    data_vmul = tbe.dsl.vmul(softmax, grad_softmax)
    data_sum = tbe.dsl.reduce_sum(data_vmul, axis=axis, keepdims=True)
    if list(shape_input1) != list(shape_input2):
        data_sum_tmp = tbe.dsl.broadcast(data_sum, shape)
    else:
        data_sum_tmp = tbe.dsl.broadcast(data_sum, shape_input2)
    data_sub = tbe.dsl.vsub(grad_softmax, data_sum_tmp)
    res = tbe.dsl.vmul(softmax, data_sub)
    if has_improve_precision:
        res = tbe.dsl.cast_to(res, "float16")

    with tvm.target.cce():
        sch = tbe.dsl.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [softmax, grad_softmax, res]}
    tbe.dsl.build(sch, config)


ut_case = OpUT("SoftmaxGrad", "softmax.test_softmax_schedule_op_impl", "dsl_softmaxgrad")

case1 = {
    "params": [{
        "shape": (8,25000),
        "dtype": "float32",
    }, {
        "shape": (8,25000),
        "dtype": "float32",
    }, {
        "shape": (8,25000),
        "dtype": "float32",
    }, -1],
    "case_name":
        "test_softmaxgrad_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)