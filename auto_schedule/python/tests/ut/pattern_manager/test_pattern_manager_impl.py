# # -*- coding:utf-8 -*-
import warnings

import tbe
from tbe import tvm
from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.pattern_manager import parse

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("pattern_manager", "pattern_manager.test_pattern_manager_parser_impl")


def test_parse_when_broadcast_pattern(_):
    shape1 = (4, 1, 5, 16)
    shape2 = (4, 10, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape2, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    broadcast_1 = tbe.dsl.broadcast(add_1, (4, 10, 5, 16))
    mul_1 = tbe.dsl.vmul(ph_2, broadcast_1)

    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 2,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 5,
    }

    compute_type_tensor_map = {
        ComputeType.PLACEHOLDER: [ph_1, ph_2],
        ComputeType.ELEWISE: [add_1, mul_1],
        ComputeType.BROADCAST: [broadcast_1],
        ComputeType.ANY: [ph_1, ph_2, add_1, mul_1, broadcast_1],
    }

    pattern = parse([mul_1], compute_type_size_map, compute_type_tensor_map)
    return pattern == Pattern.BROADCAST


def test_parse_when_no_exited_pattern(_):
    shape1 = (4, 2, 16)
    shape2 = (4, 6, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape2, dtype=fp16, name="ph_1")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 2)
    concat_1 = tbe.dsl.concat([add_1, mul_1], 1)
    transpose_1 = tbe.dsl.transpose(concat_1, [0, 2, 1])
    reduce_1 = tbe.dsl.reduce_max(transpose_1, [1], keepdims=True)
    broadcast_1 = tbe.dsl.broadcast(reduce_1, transpose_1.shape)

    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 2,
        ComputeType.CONCAT: 1,
        ComputeType.TRANSPOSE: 1,
        ComputeType.REDUCE: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 8,
    }

    compute_type_tensor_map = {
        ComputeType.PLACEHOLDER: [ph_1, ph_2],
        ComputeType.ELEWISE: [add_1, mul_1],
        ComputeType.CONCAT: [concat_1],
        ComputeType.TRANSPOSE: [transpose_1],
        ComputeType.REDUCE: [reduce_1],
        ComputeType.BROADCAST: [broadcast_1],
        ComputeType.ANY: [ph_1, ph_2, add_1, mul_1, concat_1, transpose_1, reduce_1, broadcast_1],
    }

    pattern = parse([mul_1], compute_type_size_map, compute_type_tensor_map)
    return pattern is None


test_funcs = [
    test_parse_when_broadcast_pattern,
    test_parse_when_no_exited_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
