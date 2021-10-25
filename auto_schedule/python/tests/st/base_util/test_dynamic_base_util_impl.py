from tbe.tvm import expr
from sch_test_frame.ut import OpUT
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base import classify



ut_case = OpUT("check_util", "para_check.test_dynamic_base_util_impl")

def test_expr_equal_condition_not_None(_):
    try:
        expr_equal(10, 100, "condition")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return False

def test_expr_equal_type(_):
    try:
        input1 = "input1"
        input2 = "input2"
        expr_equal(input1, input2)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return False


def test_classify_disable_optimization_in_elewise(_):
    try:
        classify([{"shape":[100,]},], "elewise", {"disable_optimization":True})
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return False

def test_classify_keepdims_none_in_reduce(_):
    try:
        classify([{"shape":[100,]},], "reduce", None)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return False


ut_case.add_cust_test_func(test_func=test_expr_equal_condition_not_None)
ut_case.add_cust_test_func(test_func=test_expr_equal_type)
ut_case.add_cust_test_func(test_func=test_classify_disable_optimization_in_elewise)
ut_case.add_cust_test_func(test_func=test_classify_keepdims_none_in_reduce)