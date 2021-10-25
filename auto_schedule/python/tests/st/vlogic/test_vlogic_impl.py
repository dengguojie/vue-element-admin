# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from te.utils import shape_util

warnings.filterwarnings("ignore")


def dsl_vlogic(x, _, operation, kernel_name='dsl_vlogic'):
    input1_shape = x[0].get("shape")
    input1_dtype = x[0].get("dtype")
    data1 = tvm.placeholder(input1_shape, name='data1', dtype=input1_dtype)

    if len(x) == 2:
        input2_shape = x[1].get("shape")
        input2_dtype = x[1].get("dtype")
        input1_shape, input2_shape, shape_max = shape_util.broadcast_shapes(input1_shape,input2_shape,
                                                                            param_name_input1="x",
                                                                            param_name_input2="y")
        data2 = tvm.placeholder(input2_shape, name='data2', dtype=input2_dtype)


    if len(x) == 1:
        res = tbe.vlogic(data1, operation=operation)   
        tensor_list = [data1, res]
    else:
        res = tbe.vlogic(data1, data2, operation)
        tensor_list = [data1, data2, res]

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)

ut_case = OpUT("vlogic", "vlogic.test_vlogic_impl", "dsl_vlogic")

case1 = {"params": [[{"shape": (4, 1, 8, 6), "dtype": "bool", "format": "ND"},
                    {"shape": (4, 1, 8, 6), "dtype": "bool", "format": "ND"}],
                    {"shape": (4, 1, 8, 6), "dtype": "bool", "format": "ND"},
                    "logic_and"
                    ],
         "case_name": "test_vlogic_and_1",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [[{"shape": (2167, 8, 6), "dtype": "bool", "format": "ND"},
                    {"shape": (2167, 8, 6), "dtype": "bool", "format": "ND"}],
                    {"shape": (2167, 8, 6), "dtype": "bool", "format": "ND"},
                    "logic_or"
                    ],
         "case_name": "test_vlogic_or_2",
         "expect": "success",
         "support_expect": True
         }
    
case3 = {"params": [[{"shape": (200, 8, 6), "dtype": "bool", "format": "ND"}],
                    {"shape": (200, 8, 6), "dtype": "bool", "format": "ND"},
                    "logic_not"
                    ],
         "case_name": "test_vlogic_not_3",
         "expect": "success",
         "support_expect": True
         }

case4 = {"params": [[{"shape": (2, 8, 6), "dtype": "bool", "format": "ND"},
                    {"shape": (2, 8, 6), "dtype": "bool", "format": "ND"}],
                    {"shape": (2, 8, 6), "dtype": "bool", "format": "ND"},
                    "logic_all"
                    ],
         "case_name": "test_vlogic_other_4",
         "expect": "RuntimeError",
         "support_expect": True
         }
    

compile_case = {
    "1": [case1, "Ascend310"],
    "2": [case2, "Ascend910A"],
    "3": [case3, "all"]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _, operation):
    x1_value = x[0].get("value")
    if len(x) == 2:
        x2_value = x[1].get("value")

    if operation == "logic_and":
        output = np.logical_and(x1_value, x2_value)
    elif operation == "logic_or":
        output = np.logical_and(x1_value, x2_value)
    else:
        output = np.logical_not(x1_value)
    return output


ut_case.add_precision_case(
    "Ascend310", {
        "params": [[{"shape": (1, 4, 4), "dtype": "bool", "param_type": "input"},
                   {"shape": (1, 4, 4), "dtype": "bool", "param_type": "input"}],
                   {"shape": (1, 4, 4), "dtype": "bool", "param_type": "output"},
                   "logic_and"
                   ],
        "case_name": "test_vdiv_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
    })


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
