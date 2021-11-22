# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from tbe.dsl.static_schedule.cce_schedule import *

warnings.filterwarnings("ignore")


ut_case = OpUT("cce_schedule", "cce_schedule.test_static_cce_schedule_impl")


def test_schedule_cce_len_out_large_than_one(_):
    try:
        input1 = tvm.placeholder((1,), name="input1", dtype="float16")
        input2 = tvm.placeholder((1,), name="input1", dtype="float16")
        input3 = tvm.placeholder((1,), name="input1", dtype="float16")
        outs = [input1, input2, input3]
        schedule_cce(outs, None)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_cce_build_code_tensor_list_is_non(_):
    try:
        config_map = {"name": 'test_cce_build_code_tensor_list_is_non'}
        cce_build_code(None, config_map)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_schedule_dispatch(_):
    try:
        sh_dispatch = ScheduleDispatch()
        sh_dispatch.handle_case(None)
    except NameError as e:
        print(e)
    return True


test_func_list = [
    test_schedule_cce_len_out_large_than_one,
    test_cce_build_code_tensor_list_is_non,
    test_schedule_dispatch,
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
