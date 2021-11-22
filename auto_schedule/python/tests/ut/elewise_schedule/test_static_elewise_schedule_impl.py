# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from tbe.dsl.static_schedule.elewise_schedule import *

warnings.filterwarnings("ignore")


ut_case = OpUT("elewise_schedule", "cce_schedule.test_static_elewise_schedule_impl")


def test_cceop_init_scope(_):
    try:
        CceOp("gm")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_check_is_elewise_single_and_broadcast(_):
    try:
        op = CceOp(".ub")
        op._res_tensor = tvm.placeholder((1,), name="input1", dtype="float16")
        op._origin_tensor = [tvm.placeholder((3, 4), name="input1", dtype="float16")]
        op._check_is_elewise_single_and_broadcast()
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensorize_for_op(_):
    try:
        op = CceOp(".ub")
        op.tensorize_for_op({"op": "11_22", "cache_buffer": 111, "tensorize_shape": 11, "tensorize_axis":1})
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_cceop_init_scope,
    test_check_is_elewise_single_and_broadcast,
    test_tensorize_for_op,
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
