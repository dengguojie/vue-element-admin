# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vabs(x, _, kernel_name='dsl_vabs'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.vabs(data1)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vabs", "vabs.test_vabs_impl", "dsl_vabs")

case1 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_vabs_1",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND"}
                    ],
         "case_name": "test_vabs_2",
         "expect": "success",
         "support_expect": True
         }

compile_case = [
    case1,
    case2
]
for item in compile_case:
    ut_case.add_case(case=item)


def calc_expect_func(x, _):
    x_value = x.get("value")
    output = np.abs(x_value)
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vabs_precision_faster_rcnn_resnet_v1_50",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vabs_precision_opticalflow",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vabs_precision_yolo_person_detect",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
