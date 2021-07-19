# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")



ut_case = OpUT("ReduceMeanD", "impl.reduce_mean_d", "reduce_mean_d")

case1 = {
    "params": [{"shape": (1, 3, 250, 250, 16), "dtype": "float16", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (1, 40, 250, 250)},
               {"shape": (1, 3, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_format": "ND","ori_shape": (1, 40, 1, 1)},
               [2, 3],
               True
               ],
    "case_name": "test_reduce_mean_1",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1
]
for item in compile_case_list:
    ut_case.add_case(case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
