# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

import tbe
from tbe import tvm
from tbe.common.context import op_context
from tbe.common.context import op_info

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_op_info_impl", "op_info")


def test_op_info_name(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    op_name = op_info1.op_name
    if op_name == "op_name":
        return True
    return False


def test_op_info_type(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    op_type = op_info1.op_type
    if op_type == "op_type":
        return True
    return False


def test_op_info_pattern(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    op_info1.pattern = "pattern"
    pattern = op_info1.pattern
    if pattern == "pattern":
        return True
    return False


def test_op_info_inputs(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    inputInit = ["inputs1", "inputs2"]
    op_info1.inputs = inputInit
    inputs = op_info1.inputs
    if inputs == inputInit:
        return True
    return True


def test_op_info_outputs(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    outputInit = ["inputs1", "inputs2"]
    op_info1.outputs = outputInit
    outputs = op_info1.outputs
    if outputs == outputInit:
        return True
    return False


def test_op_info_attrs(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    attrs_init = {"attr1", "attr2", "attr3"}
    op_info1.attrs = attrs_init
    attrs = op_info1.attrs
    if attrs_init == attrs:
        return True
    return False


def test_op_info_kernel_name(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    op_info1.kernel_name = "kernel_name"
    kernel_name = op_info1.kernel_name
    if kernel_name == "kernel_name":
        return True
    return False


def test_op_info_extra_params(_):
    op_info1 = op_info.OpInfo("op_name", "op_type")
    extra_params_init = {"param1", "param2"}
    op_info1.extra_params = extra_params_init
    extra_params = op_info1.extra_params
    if extra_params == extra_params_init:
        return True
    return False


test_func_list = [
    test_op_info_name,
    test_op_info_type,
    test_op_info_pattern,
    test_op_info_inputs,
    test_op_info_outputs,
    test_op_info_extra_params,
    test_op_info_attrs,
    test_op_info_kernel_name
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
