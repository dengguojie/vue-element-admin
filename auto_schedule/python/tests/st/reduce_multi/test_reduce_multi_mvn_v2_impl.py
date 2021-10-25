# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from mvn_v2 import mvn_v2

warnings.filterwarnings("ignore")


def dsl_reduce_multi_mvn_v2(input_x, output_y, eps, axis, kernel_name="test_dsl_reduce_multi_mvn_v2"):
    mvn_v2(input_x, output_y, eps, axis, kernel_name)

ut_case = OpUT("reduce_multi", "reduce_multi.test_reduce_multi_mvn_v2_impl", "dsl_reduce_multi_mvn_v2")


case1 = {"params": [{"shape": (7, 5, 42, 14), "dtype": "float32", "format": "NCHW", "ori_shape": (7, 5, 42, 14),"ori_format": "NCHW"},
                    {"shape": (7, 5, 42, 14), "dtype": "float32", "format": "NCHW", "ori_shape": (7, 5, 42, 14),"ori_format": "NCHW"},
                    9.0,
                    [3, 2, 1, 0]
                    ],
         "case_name": "test_dsl_reduce_multi_mvn_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
         }
case2 = {"params": [{"shape": (5, 31, 27, 26), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 31, 27, 26),"ori_format": "NCHW"},
                    {"shape": (5, 31, 27, 26), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 31, 27, 26),"ori_format": "NCHW"},
                    9.0,
                    [0, 2, 1]
                    ],
         "case_name": "test_dsl_reduce_multi_mvn_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
         }
case3 = {"params": [{"shape": (64, 61, 92, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (64, 61, 92, 1),"ori_format": "NCHW"},
                    {"shape": (64, 61, 92, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (64, 61, 92, 1),"ori_format": "NCHW"},
                    9.0,
                    [0]
                    ],
         "case_name": "test_dsl_reduce_multi_mvn_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
        }

compile_case_list = [
    case1, case2, case3
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
