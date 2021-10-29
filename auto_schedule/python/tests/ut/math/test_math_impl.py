# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe import tvm
from tbe.dsl.base import var
from tbe.dsl.compute import math

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_op_context_impl", "dsl_vsub")


def test_auto_cast_of_elewise_first_input_not_tensor_exception(_):
    @math._auto_cast_of_elewise
    def test_vadd(args1, args2):
        return None

    lhs_shape = (16, 64)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs_shape = (16, 64)
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    try:
        test_vadd(lhs, rhs)
    except RuntimeError as _:
        return False
    return True


test_func_list = [
    test_auto_cast_of_elewise_first_input_not_tensor_exception,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
