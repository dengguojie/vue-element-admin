from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.context import op_context
from tbe.dsl.compute.reduce import _single_reduce_op
from tbe.dsl.compute.reduce import _tuple_reduce_op
from tbe.dsl.compute.reduce import tuple_sum

ut_case = OpUT("reduce_compute", "reduce_compute.test_dynamic_reduce_compute_impl")


def test_single_reduce_op_coverage_1(_):
    shape = (10,0,2)
    data = tvm.placeholder(shape, name="data", dtype="float16")
    axis = [1]
    try:
        with op_context.OpContext("dynamic"):
            _single_reduce_op(data, axis, "reduce_sum", False)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return True


def test_tuple_reduce_op_coverage(_):
    shape = (10,)
    data = tvm.placeholder(shape, name="data", dtype="float16")
    axis = None
    try:
        _tuple_reduce_op([data, data], axis, "tuple_reduce_sum", False)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return True


def test_auto_cast_of_tuple_reduce_coverage(_):
    shape = (10,20,30)
    data = tvm.placeholder(shape, name="data", dtype="float16")
    input_list = (data, data)
    axis = [1]
    keepdims = True
    try:
        tuple_sum(input_list, axis, keepdims)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E90001"
    return True


ut_case.add_cust_test_func(test_func=test_single_reduce_op_coverage_1)
ut_case.add_cust_test_func(test_func=test_tuple_reduce_op_coverage)
ut_case.add_cust_test_func(test_func=test_auto_cast_of_tuple_reduce_coverage)
