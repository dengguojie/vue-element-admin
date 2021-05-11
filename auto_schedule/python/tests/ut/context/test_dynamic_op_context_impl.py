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

ut_case = OpUT("context", "context.test_dynamic_op_context_impl", "dsl_vsub")


def test_op_context_op_info(_):
    opInfo1 = op_info.OpInfo("opname1", "opType1")
    opInfo2 = op_info.OpInfo("opname2", "opType2")
    opContext = op_context.OpContext()
    opContext.add_op_info(opInfo1)
    opContext.add_op_info(opInfo2)
    op_info_get = opContext.get_op_info("opname1")
    if op_info_get == opInfo1:
        return True
    return False


def test_op_context_op_mode(_):
    opContext = op_context.OpContext()
    opContext.set_op_mode("opMode")
    opMode_get = opContext.get_op_mode()
    if opMode_get == "opMode":
        return True
    return False


def test_op_context_build_res_none(_):
    opContext = op_context.OpContext()
    build_res_none = opContext.get_build_res()
    if 0 == len(build_res_none):
        return True
    return False


def test_op_context_build_res(_):
    opContext = op_context.OpContext()
    opContext.add_build_res("build_res_k", "build_res_v")
    build_res_v = opContext.get_build_res("build_res_k")
    if build_res_v == "build_res_v":
        return True
    return False


def test_op_context_buffer_manager(_):
    opContext = op_context.OpContext()
    opContext.set_buffer_manager("buffer_manager")
    buffer_manager = opContext.get_buffer_manager()
    if buffer_manager == "buffer_manager":
        return True
    return False


def test_op_context_build_type(_):
    opContext = op_context.OpContext()
    opContext.set_build_type("build_type")
    build_type = opContext.get_build_type()
    if build_type == "build_type":
        return True
    return False


def test_op_context_missing_support_info(_):
    opContext = op_context.OpContext()
    opContext.set_missing_support_info("missing_support_info")
    missing_support_info = opContext.get_missing_support_info()
    if missing_support_info == "missing_support_info":
        return True
    return False


def test_op_context_build_json_result(_):
    opContext = op_context.OpContext()
    opContext.add_build_json_result("build_json_result_k", "build_json_result_v")
    build_json_result_v = opContext.get_build_json_result("build_json_result_k")
    if build_json_result_v == "build_json_result_v":
        return True
    return False


def test_op_context_build_json_result(_):
    opContext = op_context.OpContext()
    opContext.add_addition("addition_k", "addition_v")
    addition_v = opContext.get_addition("addition_k")
    if addition_v == "addition_v":
        return True
    return False


test_func_list = [
    test_op_context_op_info,
    test_op_context_op_mode,
    test_op_context_build_res_none,
    test_op_context_build_res,
    test_op_context_buffer_manager,
    test_op_context_build_type,
    test_op_context_missing_support_info,
    test_op_context_build_json_result

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
