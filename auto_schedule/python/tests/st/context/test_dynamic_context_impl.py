# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe.common.context import op_info
from tbe.dsl.base import var
from tbe.dsl.base import context
from tbe.dsl.base import operation

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_context_impl", "dsl_context")


def test_operator_context_op_type1(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        opInfo1 = op_info.OpInfo("opname1", "opType1")
        operation.get_op_context().add_op_info(opInfo1)
        opType = context.OperatorContext.get_op_type()

        if opType == "opType1":
            return True
    return False


def test_operator_context_op_type2(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        opInfo1 = op_info.OpInfo("opname1", "opType1")
        opInfo2 = op_info.OpInfo("opname1", "opType2")
        operation.get_op_context().add_op_info(opInfo1)
        operation.get_op_context().add_op_info(opInfo2)
        opType = context.OperatorContext.get_op_type()

        if opType == "fusion_opType1_opType2":
            return True
    return False


def test_operator_context_op_type_none(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        opType = context.OperatorContext.get_op_type()
        if opType is None:
            return True
    return False


def test_operator_context_var(_):
    operatorContext = context.OperatorContext()
    var1 = var.Var("var1", 1, "float16")
    operatorContext.add_var(var1)
    var2 = var.Var("var2", 1, "float16")
    operatorContext.add_var(var2)
    var2_get = operatorContext.get_var("var2")
    if var2_get == var2:
        return True
    return False


def test_operator_context_var_none(_):
    operatorContext = context.OperatorContext()
    var2_get = operatorContext.get_var("var2")
    if var2_get is None:
        return True
    return False


def test_operator_context_var_with_compute(_):
    computeContext = context.ComputeContext()
    operatorContext = context.OperatorContext()
    operatorContext.begin_compute(computeContext)
    var1 = var.Var("var1", 1, "float16")
    operatorContext.add_var(var1)
    var2 = var.Var("var2", 1, "float16")
    operatorContext.add_var(var2)
    var2_get = operatorContext.get_var("var2")

    operatorContext.end_compute(computeContext)
    if var2_get == var2:
        return True
    return False


def test_operator_context_var_list(_):
    operatorContext = context.OperatorContext()
    var1 = var.Var("var1", 1, "float16")
    operatorContext.add_var(var1)
    var2 = var.Var("var2", 1, "float16")
    operatorContext.add_var(var2)
    varList = operatorContext.get_vars()
    if var1 in varList and var2 in varList:
        return True
    return False


def test_operator_context_var_desc(_):
    operatorContext = context.OperatorContext()
    varDesc1 = var.AttrVarDesc("varDesc1", "float16", "float16")
    operatorContext.add_attr_var_desc(varDesc1)
    varDesc2 = var.AttrVarDesc("varDesc2", "float16", "float16")
    operatorContext.add_attr_var_desc(varDesc2)
    varDesc2_get = operatorContext.get_attr_vars_desc()
    if varDesc1 in varDesc2_get and varDesc2 in varDesc2_get:
        return True
    return False


def test_operator_context_var_desc_with_compute(_):
    computeContext = context.ComputeContext()
    operatorContext = context.OperatorContext()
    operatorContext.begin_compute(computeContext)
    varDesc1 = var.AttrVarDesc("varDesc1", "float16", "float16")
    operatorContext.add_attr_var_desc(varDesc1)
    varDesc2 = var.AttrVarDesc("varDesc2", "float16", "float16")
    operatorContext.add_attr_var_desc(varDesc2)
    varDesc2_get = operatorContext.get_current_compute().get_attr_vars_desc()
    operatorContext.end_compute(computeContext)
    if varDesc1 in varDesc2_get and varDesc2 in varDesc2_get:
        return True
    return False


def test_operator_context_pattern(_):
    operatorContext = context.OperatorContext()
    operatorContext.set_pattern("pattern")
    pattern = operatorContext.get_pattern()
    if pattern == "pattern":
        return True
    return False


def test_operator_context_addition(_):
    operatorContext = context.OperatorContext()
    operatorContext.add("additionKey", "additionValue")
    additionValue = operatorContext.get("additionKey")
    if additionValue == "additionValue":
        return True
    return False


def test_operator_context_exclude_bound_var(_):
    operatorContext = context.OperatorContext()
    var4 = var.Var("var4", 1, "float16")
    operatorContext.add_exclude_bound_var(var4)
    exclude_bound_var4_get = operatorContext.get_exclude_bound_vars()
    if var4 in exclude_bound_var4_get:
        return True
    return False


def test_operator_context_exclude_bound_var_with_compute(_):
    computeContext = context.ComputeContext()
    operatorContext = context.OperatorContext()
    operatorContext.begin_compute(computeContext)
    var4 = var.Var("var4", 1, "float16")
    operatorContext.add_exclude_bound_var(var4)
    exclude_bound_var = operatorContext.get_current_compute().get_exclude_bound_vars()
    operatorContext.end_compute(computeContext)
    if var4 in exclude_bound_var:
        return True
    return False


def test_operator_context_begin_compute(_):
    operatorContext = context.OperatorContext()
    operatorContext.begin_compute("compute1")
    try:
        operatorContext.begin_compute("compute2")
        return False
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_operator_context_begin_compute_none(_):
    operatorContext = context.OperatorContext()
    if len(operatorContext.get_computes()) == 0:
        return True
    return False


def test_operator_context_end_compute(_):
    operatorContext = context.OperatorContext()
    operatorContext.begin_compute("compute1")
    try:
        operatorContext.end_compute("compute2")
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_operator_context_set_default(_):
    operatorContext = context.OperatorContext()
    """
        set_default
        """
    default_value = operatorContext.set_default("default_key", "default_value")
    if default_value == "default_value":
        return True
    return False


def compute_default_stub():
    return "12345"


def test_operator_context_compute_default(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        operatorContext = context.OperatorContext()
        default_value = operatorContext.compute_default("default_key2", compute_default_stub)
        if "12345" == default_value:
            return True
        return False


def test_compute_context_enter_exit(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        with context.ComputeContext():
            computeContext = context.ComputeContext()
            current_compute = computeContext.get_operator_context().get_current_compute()
            if current_compute is not None:
                return True
    return False


def test_compute_context_begin_schedule(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        computeContext.begin_schedule(context.ScheduleContext())

        try:
            computeContext.begin_schedule(context.ScheduleContext())
        except RuntimeError as e:
            errorCode = e.args[0].get("errCode")
            if errorCode == "E90001":
                return True
    return False


def test_compute_context_end_schedule_exception(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        computeContext.begin_schedule(context.ScheduleContext())

        try:
            computeContext.end_schedule(context.ScheduleContext())
        except RuntimeError as e:
            errorCode = e.args[0].get("errCode")
            if errorCode == "E90001":
                return True
    return False


def test_compute_context_end_schedule(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        scheduleContext = context.ScheduleContext()
        computeContext.begin_schedule(scheduleContext)
        computeContext.end_schedule(scheduleContext)
        if computeContext.get_current_schedule() is None:
            return True
    return False


def test_compute_context_get_schedules(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        if len(computeContext.get_schedules()) == 0:
            return True
    return False


def test_compute_context_get_operator_context(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()

        if computeContext.get_operator_context is not None:
            return True
    return False


def test_compute_context_pattern(_):
    computeContext = context.ComputeContext()
    computeContext.set_pattern("pattern")
    pattern = computeContext.get_pattern()
    if pattern == "pattern":
        return True
    return False


def test_compute_context_get_var_none(_):
    computeContext = context.ComputeContext()
    varNone = computeContext.get_var("var1")
    if varNone is None:
        return True
    return False


def test_compute_context_get_var(_):
    computeContext = context.ComputeContext()
    var1 = var.Var("var1", 1, "float16")
    computeContext.add_var(var1)
    var2 = var.Var("var2", 1, "float16")
    computeContext.add_var(var2)
    var2_get = computeContext.get_var("var2")
    if var2_get == var2:
        return True
    return False


def test_compute_context_get_var_with_schedule(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        scheduleContext = context.ScheduleContext()
        computeContext.begin_schedule(scheduleContext)
        var1 = var.Var("var1", 1, "float16")
        computeContext.add_var(var1)
        vars_get = computeContext.get_var("var1")
        computeContext.end_schedule(scheduleContext)

        if vars_get is not None:
            return True
    return False


def test_compute_context_get_vars(_):
    computeContext = context.ComputeContext()
    var1 = var.Var("var1", 1, "float16")
    computeContext.add_var(var1)
    var2 = var.Var("var2", 1, "float16")
    computeContext.add_var(var2)
    var2_get = computeContext.get_vars()

    if var2 in var2_get:
        return True
    return False


def test_compute_attr_var_desc(_):
    computeContext = context.ComputeContext()
    attrVarDesc = var.AttrVarDesc("attr", "float16", "float16")
    computeContext.add_attr_var_desc(attrVarDesc)
    get_attr_vars_desc = computeContext.get_attr_vars_desc()
    if attrVarDesc in get_attr_vars_desc:
        return True
    return False


def test_compute_attr_var_desc_with_schedule(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        scheduleContext = context.ScheduleContext()

        computeContext.begin_schedule(scheduleContext)
        attrVarDesc = var.AttrVarDesc("var3", "float16", "float16")
        computeContext.add_attr_var_desc(attrVarDesc)
        exclude_bound_attrVarDesc_get = computeContext.get_current_schedule().get_attr_vars_desc()
        computeContext.end_schedule(scheduleContext)

        if attrVarDesc in exclude_bound_attrVarDesc_get:
            return True
    return False


def test_compute_add_exclude_bound_var(_):
    computeContext = context.ComputeContext()
    var3 = var.Var("var3", 1, "float16")
    computeContext.add_exclude_bound_var(var3)
    exclude_bound_var3_get = computeContext.get_exclude_bound_vars()
    if exclude_bound_var3_get[0] == var3:
        return True
    return False


def test_compute_add_exclude_bound_var_with_schedule(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        scheduleContext = context.ScheduleContext()
        computeContext.begin_schedule(scheduleContext)

        var3 = var.Var("var3", 1, "float16")
        computeContext.add_exclude_bound_var(var3)
        exclude_bound_var3_get = computeContext.get_current_schedule().get_exclude_bound_vars()
        computeContext.end_schedule(scheduleContext)
        if var3 in exclude_bound_var3_get:
            return True
    return False


def test_compute_context_addition(_):
    computeContext = context.ComputeContext()
    computeContext.add("additionKey", "additionValue")
    additionValue = computeContext.get("additionKey")

    if additionValue == "additionValue":
        return True
    return False


def test_compute_context_set_default(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        default_value = computeContext.set_default("default_key", "default_value")
        if default_value == "default_value":
            return True
    return False


def test_compute_context_compute_default(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        computeContext = context.ComputeContext()
        default_value = computeContext.compute_default("default_key2", compute_default_stub)
        if "12345" == default_value:
            return True
    return False


def test_schedule_context_compute_context(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        scheduleContext = context.ScheduleContext()
        compute_context = scheduleContext.get_compute_context()
        if compute_context is not None:
            return False
    return True


def test_schedule_context_attr_var_desc(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        scheduleContext = context.ScheduleContext()
        attrVarDesc = var.AttrVarDesc("attr", "float16", "float16")
        scheduleContext.add_attr_var_desc(attrVarDesc)
        add_attr_var_desc = scheduleContext.get_attr_vars_desc()
        if attrVarDesc in add_attr_var_desc:
            return True
    return False


def test_schedule_context_set_default(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        scheduleContext = context.ScheduleContext()
        default_value = scheduleContext.set_default("default_key", "default_value")
        if default_value == "default_value":
            return True
    return False


def test_schedule_context_exclude_bound_var(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        scheduleContext = context.ScheduleContext()
        var4 = var.Var("var2", 1, "float16")
        scheduleContext.add_exclude_bound_var(var4)
        exclude_bound_var1_get = scheduleContext.get_exclude_bound_vars()
        if var4 in exclude_bound_var1_get:
            return True
    return False


def test_schedule_context_enter_exit(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        with context.ComputeContext():
            with context.ScheduleContext():
                scheduleContext = context.ScheduleContext()
                current_schedule = scheduleContext.get_compute_context().get_current_schedule()
                if current_schedule is not None:
                    return True
    return False


def test_schedule_context_get_vars(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        with context.ComputeContext():
            scheduleContext = context.ScheduleContext()
            var1 = var.Var("var1", 1, "float16")
            scheduleContext.add_var(var1)
            var2 = var.Var("var2", 1, "float16")
            scheduleContext.add_var(var2)
            var2_get = scheduleContext.get_vars()
            if var2 in var2_get:
                return True
    return False


def test_schedule_context_get_var_none(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        with context.ComputeContext():
            scheduleContext = context.ScheduleContext()
            var_get = scheduleContext.get_var("abc")
            if var_get is None:
                return True
    return False


def test_schedule_context_compute_default(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        scheduleContext = context.ScheduleContext()
        default_value = scheduleContext.compute_default("default_key2", compute_default_stub)
        if "12345" == default_value:
            return True
    return False


test_func_list = [
    test_operator_context_op_type1,
    test_operator_context_op_type2,
    test_operator_context_op_type_none,
    test_operator_context_var,
    test_operator_context_var_with_compute,
    test_operator_context_var_none,
    test_operator_context_var_list,
    test_operator_context_var_desc,
    test_operator_context_var_desc_with_compute,
    test_operator_context_pattern,
    test_operator_context_begin_compute,
    test_operator_context_end_compute,
    test_operator_context_begin_compute_none,
    test_operator_context_set_default,
    test_operator_context_compute_default,
    test_operator_context_exclude_bound_var,
    test_operator_context_exclude_bound_var_with_compute,
    test_compute_context_enter_exit,
    test_compute_context_begin_schedule,
    test_compute_context_end_schedule_exception,
    test_compute_context_end_schedule,
    test_compute_context_get_schedules,
    test_compute_context_get_var_none,
    test_compute_context_get_var,
    test_compute_context_get_vars,
    test_compute_context_get_var_with_schedule,
    test_compute_attr_var_desc,
    test_compute_context_get_operator_context,
    test_compute_context_pattern,
    test_compute_add_exclude_bound_var,
    test_compute_attr_var_desc_with_schedule,
    test_compute_context_addition,
    test_compute_context_compute_default,
    test_compute_context_set_default,
    test_compute_add_exclude_bound_var_with_schedule,
    test_schedule_context_compute_context,
    test_schedule_context_attr_var_desc,
    test_schedule_context_set_default,
    test_schedule_context_exclude_bound_var,
    test_schedule_context_enter_exit,
    test_schedule_context_get_vars,
    test_schedule_context_get_var_none,
    test_schedule_context_compute_default
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
