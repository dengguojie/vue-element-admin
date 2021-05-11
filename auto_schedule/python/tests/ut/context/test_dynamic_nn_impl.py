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


test_func_list = [
    test_vmaddrelu_instance_tensor_1,
    test_vaddrelu_lhs_tensor,
    test_vsubrelu_lhs_tensor,
    test_broadcast_shape_tensor,
    test_tensor_broadcast_shape_length
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
