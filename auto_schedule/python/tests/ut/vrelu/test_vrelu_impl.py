# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vrelu(x, _,  kernel_name='dsl_vrelu'):
    input1_shape = x.get("shape")
    input1_dtype = x.get("dtype")
    data1 = tvm.placeholder(input1_shape, name='data1', dtype=input1_dtype)
    res = tbe.vrelu(data1)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vrelu", "vrelu.test_vrelu_impl", "dsl_vrelu")


def test_shape_value_less_than_zero(_):
    try:
        input1 = tvm.placeholder((-1, ), name="input1", dtype="float16")
        tbe.vrelu(input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_shape_value_less_than_zero, None],
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (1, 13, 13, 1000), "dtype": "float16", "format": "ND"},
                    {"shape": (1, 13, 13, 1000), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_vrelu_squeeze1_1_frozen",
         "expect": "success",
         "support_expect": True
         }

case1 = {"params": [{"shape": (128, 50), "dtype": "float16", "format": "ND"},
                    {"shape": (128, 50), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_vrelu_widedeep_tf_all",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case1, None]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _):
    x_value = x.get("value")
    output = np.maximum(x_value, 0)
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (2, 64, 64, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (2, 64, 64, 128), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vrelu_precision_2d_ncp_segmentation",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 10, 64), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 10, 64), "dtype": "float16", "param_type": "output"}
                   ],
        "case_name": "test_vrelu_precision_cae_bt_al00",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (16, 16, 16, 16, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (16, 16, 16, 16, 16), "dtype": "float16", "param_type": "output"}
                   ],
        "case_name": "test_vrelu_precision_3D_airway_segmentation",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (2, 512), "dtype": "float16", "param_type": "input"},
                   {"shape": (2, 512), "dtype": "float16", "param_type": "output"}
                   ],
        "case_name": "test_vrelu_precision_deepFm_frozen_model",
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
