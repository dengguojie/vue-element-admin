# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import do_op_tiling
from tbe.dsl import classify


def test_invalid_do_op_tiling(kernel_name):
    do_op_tiling("INVALID", "{\"INVALID\": \"INVALID\"", [], [])


ut_case = OpUT("do_op_tiling", "do_op_tiling.test_do_op_tiling_impl", "test_invalid_do_op_tiling")


case1 = {
    "params": [],
    "case_name":
        "test_invalid_do_op_tiling",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

ut_case.add_case(["all"], case1)
