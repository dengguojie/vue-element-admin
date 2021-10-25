# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vlog(x, _, priority, kernel_name='dsl_vlog'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.vlog(data1, priority_flag=priority)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vlog", "vlog.test_vlog_impl", "dsl_vlog")


def test_shape_value_less_than_zero(_):
    try:
        input1 = tvm.placeholder((-1,), name="input1", dtype="float16")
        tbe.vlog(input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_shape_value_less_than_zero, None]
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (1024, ), "dtype": "float16", "format": "ND"},
                    {"shape": (1024, ), "dtype": "float16", "format": "ND"},
                    1
                    ],
         "case_name": "test_vlog_by_taylor",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    0
                    ],
         "case_name": "test_vlog_1",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, "Ascend310"],
    "2": [case2, "Ascend910A"]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _, __):
    x_value = x.get("value")
    output = np.log(x_value)
    return output


ut_case.add_precision_case(
    "Ascend310", {
        "params": [{"shape": (1, 4, 1), "dtype": "float32", "param_type": "input", "value_range": [0.5, 1]},
                   {"shape": (1, 4, 1), "dtype": "float32", "param_type": "output"},
                   1
                   ],
        "case_name": "test_vlog_precision_transformer_run_graph",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (8, 1, 2), "dtype": "float32", "param_type": "input", "value_range": [0.5, 1]},
                   {"shape": (8, 1, 2), "dtype": "float32", "param_type": "output"},
                   0
                   ],
        "case_name": "test_vlog_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)
    })


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
