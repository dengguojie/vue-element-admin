# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from tbe.dsl.compute import nn

import tbe
from tbe import tvm

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_op_context_impl", "dsl_vsub")


def test_vmaddrelu_instance_tensor_1(_):
    try:
        nn.vmaddrelu("11", "12", "13")
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vaddrelu_lhs_tensor(_):
    try:
        nn.vaddrelu("11", "12")
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsubrelu_lhs_tensor(_):
    try:
        nn.vsubrelu("11", "12")
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_broadcast_shape_tensor(_):
    try:
        nn.broadcast(11, "12")
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_tensor_broadcast_shape_length(_):
    try:
        shape_src = (16, 64, 1, 8)
        var = tvm.placeholder(shape_src, name="var_tensor", dtype="float16")
        shape_dst = (16, 64, 32, 8, 16)
        nn.broadcast(var, shape_dst)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_nn_vmaddrelu_second_tensor_type_exception(_):
    lhs = tvm.placeholder((1, 16), name="lhs", dtype="float16")
    mid_shape = (1, 16)
    rhs = tvm.placeholder((1, 16), name="rhs", dtype="float16")
    try:
        nn.vmaddrelu(lhs, mid_shape, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_nn_vmaddrelu_third_tensor_type_exception(_):
    lhs = tvm.placeholder((1, 16), name="lhs", dtype="float16")
    mid = tvm.placeholder((1, 16), name="mid", dtype="float16")
    rhs_shape = (1, 16)
    try:
        nn.vmaddrelu(lhs, mid, rhs_shape)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vaddrelu_rhs_type_exception(_):
    lhs = tvm.placeholder((1, 16), name="lhs", dtype="float16")
    rhs_shape = (1, 16)
    try:
        nn.vaddrelu(lhs, rhs_shape)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vaddrelu_input_dtype_exception(_):
    lhs = tvm.placeholder((1, 16), name="lhs", dtype="float16")
    rhs = tvm.placeholder((1, 16), name="rhs", dtype="float32")
    try:
        nn.vaddrelu(lhs, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsubrelu_rhs_type_exception(_):
    lhs = tvm.placeholder((1, 16), name="lhs", dtype="float16")
    rhs_shape = (1, 16)
    try:
        nn.vsubrelu(lhs, rhs_shape)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsubrelu_input_dtype_exception(_):
    lhs = tvm.placeholder((1, 16), name="lhs", dtype="float16")
    rhs = tvm.placeholder((1, 16), name="rhs", dtype="float32")
    try:
        nn.vsubrelu(lhs, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vlrelu_alpha_dtype_exception(_):
    input = tvm.placeholder((1, 16), name="input", dtype="float16")
    alpha = tvm.const(3.0, dtype="float32")
    try:
        nn.vlrelu(input, alpha)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vlrelu_alpha_type_exception(_):
    input = tvm.placeholder((1, 16), name="input", dtype="float16")
    alpha = tvm.var("a", dtype="float16")
    try:
        nn.vlrelu(input, alpha)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_broadcast_second_type_exception(_):
    value = tvm.const(2, dtype="int32")
    shape = 2
    try:
        nn.broadcast(value, shape)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_tensor_broadcast_shape_exception(_):
    input = tvm.placeholder((2, 3, 4), name="input", dtype="float16")
    shape = [2, 2]
    try:
        nn._tensor_broadcast(input, shape)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_tensor_broadcast_shape_illegal_exception(_):
    input = tvm.placeholder((2, 3), name="input", dtype="float16")
    shape = [3, 2, 2]
    try:
        nn._tensor_broadcast(input, shape)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_tensor_broadcast_shape_known_broadcast(_):
    input = tvm.placeholder((2, 3), name="input", dtype="float16")
    shape = [3, 2, 3]
    nn._tensor_broadcast(input, shape)
    return True


def test_tensor_broadcast_shape_unknown_broadcast(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        shape_var = tvm.var("dim_0_1")
        input = tvm.placeholder(shape_var, name="input", dtype="float16")
        shape = [-1]
        nn._tensor_broadcast(input, shape)
        return True


test_func_list = [
    test_vmaddrelu_instance_tensor_1,
    test_vaddrelu_lhs_tensor,
    test_vsubrelu_lhs_tensor,
    test_broadcast_shape_tensor,
    test_tensor_broadcast_shape_length,
    test_nn_vmaddrelu_second_tensor_type_exception,
    test_nn_vmaddrelu_third_tensor_type_exception,
    test_vaddrelu_rhs_type_exception,
    test_vaddrelu_input_dtype_exception,
    test_vsubrelu_rhs_type_exception,
    test_vsubrelu_input_dtype_exception,
    test_vlrelu_alpha_dtype_exception,
    test_vlrelu_alpha_type_exception,
    test_broadcast_second_type_exception,
    test_tensor_broadcast_shape_exception,
    test_tensor_broadcast_shape_illegal_exception,
    test_tensor_broadcast_shape_known_broadcast,
    test_tensor_broadcast_shape_unknown_broadcast
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
