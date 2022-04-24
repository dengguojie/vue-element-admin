# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common import context as tbe_context
from tbe.dsl.base import operation


def test_dsl_tuple_sum_interface_0(_):
    reduce_axis = [0, 1]
    input_x = tvm.placeholder([160, 96, 512], dtype="float32", name="input_x")
    input_y = tvm.placeholder([160, 96, 512], dtype="float32", name="input_y")
    res_x, res_y = tbe.dsl.tuple_sum([input_x, input_y], reduce_axis, keepdims=False)

    shape_x = shape_util.shape_to_list(res_x.shape)
    shape_y = shape_util.shape_to_list(res_y.shape)
    if not shape_x == shape_y == [512]:
        return False
    return True


def test_dsl_tuple_sum_interface_1(_):
    reduce_axis = [0, 2, 3]
    input_x = tvm.placeholder([32, 3, 7, 7, 16], dtype="float32", name="input_x")
    input_y = tvm.placeholder([32, 3, 7, 7, 16], dtype="float32", name="input_y")
    res_x, res_y = tbe.dsl.tuple_sum([input_x, input_y], reduce_axis, keepdims=True)
    
    shape_x = shape_util.shape_to_list(res_x.shape)
    shape_y = shape_util.shape_to_list(res_y.shape)
    if not shape_x == shape_y == [1, 3, 1, 1, 16]:
        return False
    return True


def dsl_tuple_sum(_):
    """
    test dsl tuple_sum
    """
    input_x = tvm.placeholder([32, 3, 7, 7, 16], dtype="float32", name="input_x")
    input_y = tvm.placeholder([32, 3, 7, 7, 16], dtype="float32", name="input_y")
    reduce_axis = [0, 2, 3]
    with tbe.common.context.op_context.OpContext("dynamic"):
        res_x, res_y = tbe.dsl.tuple_sum([input_x, input_y], reduce_axis, keepdims=True)
    res_op = [res_x.op, res_y.op]
    sch = tvm.create_schedule(res_op)
    func = tvm.build(sch, [input_x, input_y, res_x, res_y], "c", "llvm", name="func")
    ctx = tvm.cpu(0)

    x = tvm.nd.array(np.random.uniform(size=(32, 3, 7, 7, 16)).astype(input_x.dtype), ctx)
    y = tvm.nd.array(np.random.uniform(size=(32, 3, 7, 7, 16)).astype(input_y.dtype), ctx)
    outs = [tvm.nd.array(np.random.uniform(size=(1, 3, 1, 1, 16)).astype(input_y.dtype), ctx),
            tvm.nd.array(np.random.uniform(size=(1, 3, 1, 1, 16)).astype(input_y.dtype), ctx)]

    func(x, y, *outs)
    x_np = x.asnumpy().reshape((32, 3, 7, 7, 16))
    y_np = y.asnumpy().reshape((32, 3, 7, 7, 16))
    ans_x = np.sum(x_np, axis=(0, 2, 3), keepdims=True)
    ans_y = np.sum(y_np, axis=(0, 2, 3), keepdims=True)

    tvm.testing.assert_allclose(outs[0].asnumpy(), ans_x, rtol=0.001, atol=0.001)
    tvm.testing.assert_allclose(outs[1].asnumpy(), ans_y, rtol=0.001, atol=0.001)
    return True


ut_case = OpUT("TupleReduce", "tuple_sum.test_dsl_tuple_sum_impl")
ut_case.add_cust_test_func(test_func=test_dsl_tuple_sum_interface_0)
ut_case.add_cust_test_func(test_func=test_dsl_tuple_sum_interface_1)
ut_case.add_cust_test_func(test_func=dsl_tuple_sum)

