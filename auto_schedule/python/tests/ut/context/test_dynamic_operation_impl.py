# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe import tvm
from tbe.dsl.base import var
from tbe.dsl.base import context
from tbe.dsl.base import operation

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_operation_impl", "dsl_operation")


def test_operation_var_exception(_):
    try:
        operation.var("_varName", None, "int32", None)
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_operation_var(_):
    tvm_var = operation.var("varName", None, "int32", None)
    if tvm_var.name == "varName":
        return True
    return False


def test_operation_var_attr_simple_dtype(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        tvm_var = operation.var_attr("varName", None, "int32", None)
        if tvm_var.name == "varName" \
                and operation.get_context().get_attr_vars_desc()[-1].name == "varName":
            return True
    return False


def test_operation_var_attr_dtype_exception(_):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            operation.var_attr("varName", None, "*int*32", None)
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_operation_var_attr_dtype_list(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        vars = operation.var_attr("varName", None, "int32[2]", None)
        if vars[0].name == "varName_0" and vars[1].name == "varName_1":
            return True
    return False


def test_attr_var_index(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        var_0 = operation.var_attr("var_0", None, "int32", addition={"index": 0})
        index = operation.get_context().get_attr_vars_desc()[-1].index

    return index == 0


def test_attr_var_index_default(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        var_0 = operation.var_attr("var_0", None, "int32")
        index = operation.get_context().get_attr_vars_desc()[-1].index

    return index == -1


def test_attr_var_index_non_int(_):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            var_0 = operation.var_attr("var_0", None, "int32", addition={"index": "a"})
            index = operation.get_context().get_attr_vars_desc()[-1].index
    except RuntimeError as e:
        error_msg = e.args[0].get("detailed_cause")
        return "Index in attr var must greater than or equal to 0" in error_msg

    return False


def test_attr_var_index_lt_0(_):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            var_0 = operation.var_attr("var_0", None, "int32", addition={"index": -1})
            index = operation.get_context().get_attr_vars_desc()[-1].index
    except RuntimeError as e:
        error_msg = e.args[0].get("detailed_cause")
        return "Index in attr var must greater than or equal to 0" in error_msg

    return False


def test_operation_var_inner_exception(_):
    try:
        operation.var_inner("varName", None, "int32", None)
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_operation_add_compile_info_with_exception(_):
    try:
        operation.add_compile_info("_varName", "int32")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_operation_add_exclude_bound_var(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        var1 = var.Var("var1", 1, "float16")
        operation.add_exclude_bound_var(var1)

        if var1 in operation.get_context().get_exclude_bound_vars():
            return True
    return False


def test_operation_operator_static(_):
    operator = operation.static()
    if operator.get_op_mode() == "static":
        return True
    return False


def test_operation_operator_dynamic(_):
    operator = operation.dynamic()
    if operator.get_op_mode() == "dynamic":
        return True
    return False


def test_operation_add_compile_info_inner_with_exception(_):
    try:
        operation.add_compile_info_inner("varName", "int32")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


test_func_list = [
    test_operation_var_exception,
    test_operation_var,
    test_operation_var_attr_simple_dtype,
    test_operation_var_attr_dtype_exception,
    test_operation_var_attr_dtype_list,
    test_attr_var_index,
    test_attr_var_index_default,
    test_attr_var_index_non_int,
    test_attr_var_index_lt_0,
    test_operation_var_inner_exception,
    test_operation_add_compile_info_with_exception,
    test_operation_add_compile_info_inner_with_exception,
    test_operation_add_exclude_bound_var,
    test_operation_operator_static,
    test_operation_operator_dynamic

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
