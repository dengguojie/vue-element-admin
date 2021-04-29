# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from softmax_v2 import softmax_v2

warnings.filterwarnings("ignore")


def dsl_reduce_multi_sv2(input_x, output_y, axis=-1, kernel_name="test_dsl_reduce_multi_sv2"):
    softmax_v2(input_x, output_y, axis, kernel_name)

ut_case = OpUT("reduce_multi", "reduce_multi.test_reduce_multi_sv2_impl", "dsl_reduce_multi_sv2")


case1 = {"params": [{"shape": (24,16,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,512,512),"ori_format": "NCHW"},
                    {"shape": (24,16,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,512,512),"ori_format": "NCHW"},
                    [-1]
                    ],
         "case_name": "test_dsl_reduce_multi_sv2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
         }
case2 = {"params": [{"shape": (1,11,11,11,17,17), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,11,187,187),"ori_format": "NCHW"},
                    {"shape": (1,11,11,11,17,17), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,11,187,187),"ori_format": "NCHW"},
                    [-1]
                    ],
         "case_name": "test_dsl_reduce_multi_sv2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
         }
case3 = {"params": [{"shape": (1,16,1,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,16,16,16),"ori_format": "NCHW"},
        {"shape": (1,16,1,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,16,16,16),"ori_format": "NCHW"},
        [-1]
        ],
"case_name": "test_dsl_reduce_multi_sv2_3",
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
