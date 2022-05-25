# # -*- coding:utf-8 -*-
import warnings

import tbe
import te.lang.cce
from tbe import tvm
from tbe.dsl.unify_schedule.vector.tuple_reduce.tuple_reduce_schedule_helper import Schedule
from tbe.dsl.unify_schedule.vector.tuple_reduce.tuple_reduce_schedule_helper import Compute

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("tuple_reduce_schedule_helper", "tuple_reduce.test_tuple_reduce_schedule_helper_impl")

# noinspection PyTypeChecker
def test_tuple_reduce_schedule_helper_poset(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 4)
    sum_1, sum_2 = tbe.dsl.tuple_sum([add_1, mul_1], (1,))

    sch = Schedule([sum_1, sum_2])
    partial_order_set_stage = sch.poset(mul_1)

    graph = Compute([sum_1, sum_2])
    partial_order_set_tensor = graph.poset(mul_1)
    return len(list(partial_order_set_stage)) == len(list(partial_order_set_tensor)) == 1

# noinspection PyTypeChecker
def test_tuple_reduce_schedule_helper_reduce_filter(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 4)
    sum_1, sum_2 = tbe.dsl.tuple_sum([add_1, mul_1], (1,))

    sch = Schedule([sum_1, sum_2])
    reduce_tensor = sch.reduce_tensors
    return len(list(reduce_tensor)) == 2

# noinspection PyTypeChecker
def test_tuple_reduce_schedule_helper_get_tensor(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 4)
    sum_1, sum_2 = tbe.dsl.tuple_sum([add_1, mul_1], (1,))

    sch = Schedule([sum_1, sum_2])
    stage = sch[mul_1]
    tensor = sch.get_tensor(stage)
    return tensor == mul_1

# noinspection PyTypeChecker
def test_tuple_reduce_schedule_helper_postorder(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 4)
    sum_1, sum_2 = tbe.dsl.tuple_sum([add_1, mul_1], (1,))

    sch = Schedule([sum_1, sum_2])
    postorder_list = sch.postorder
    return True


test_funcs = [
    test_tuple_reduce_schedule_helper_poset,
    test_tuple_reduce_schedule_helper_reduce_filter,
    test_tuple_reduce_schedule_helper_get_tensor,
    test_tuple_reduce_schedule_helper_postorder
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
