# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vexp(x, _, kernel_name='dsl_vexp'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.vexp(data1)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vexp", "vexp.test_vexp_impl", "dsl_vexp")

case1 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_vexp_1",
         "expect": "success",
         "support_expect": True
         }

ut_case.add_case(case=case1)


def calc_expect_func(x, _):
    x_value = x.get("value")
    output = np.exp(x_value)
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 1, 3, 2), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 1, 3, 2), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vexp_precision_01",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 8, 8, 3, 2), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 8, 8, 3, 2), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vexp_precision_yolov3_coco",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (8, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (8, ), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vexp_precision_Cascade_RCNN_inference",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4, 4), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4, 4), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vexp_precision_fcos_res50",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 2), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, 2), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vexp_precision_RefineDet",
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
