# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from tbe.dsl.static_schedule.vector_schedule import *

warnings.filterwarnings("ignore")


ut_case = OpUT("vector_schedule", "cce_schedule.test_static_vector_schedule_impl")


def test_get_l1fuison_flag(_):
    try:
        vector = VectorSchedule()
        vector._fusion_params = {"l1_fusion_type": 1}
        vector._get_l1fuison_flag()
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True



test_func_list = [
    test_get_l1fuison_flag,
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
