# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vlrelu(x, _, factor, is_tvm_const, kernel_name='dsl_vlrelu'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    if is_tvm_const:
        data2 = tvm.const(factor, dtype=input_dtype)
        res = tbe.vlrelu(data1, data2)
    else:
        res = tbe.vlrelu(data1, factor)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vlrelu", "vlrelu.test_vlrelu_impl", "dsl_vlrelu")


def test_alpha_is_const_dtype_not_same(_):
    try:
        input1 = tvm.placeholder((-1,), name="input1", dtype="float16")
        factor = tvm.const(1, dtype="float32")
        tbe.vlrelu(input1, factor)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_alpha_is_tvm_var(_):
    try:
        input1 = tvm.placeholder((-1,), name="input1", dtype="float16")
        factor = tvm.var(name="alpha", dtype="float16")
        tbe.vlrelu(input1, factor)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_alpha_is_const_dtype_not_same, None],
    "2": [test_alpha_is_tvm_var, None]
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (16, 256, 26, 26), "dtype": "float16", "format": "ND"},
                    {"shape": (16, 256, 26, 26), "dtype": "float16", "format": "ND"},
                    1.0,
                    False
                    ],
         "case_name": "test_vlrelu_frozen_darknet_yolov3_model",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    1,
                    True
                    ],
         "case_name": "test_vlrelu_1",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case2, None]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _, factor, __):
    x_value = x.get("value")
    output = np.where(x_value > 0, x_value, factor * x_value)
    return output


ut_case.add_precision_case(
    "Ascend310", {
        "params": [{"shape": (1, 8, 8, 512), "dtype": "int32", "param_type": "input"},
                   {"shape": (1, 8, 8, 512), "dtype": "int32", "param_type": "output"},
                   1,
                   False
                   ],
        "case_name": "test_vlrelu_precision_yolov3_coco_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend310", {
        "params": [{"shape": (1, 8, 8, 512), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 8, 8, 512), "dtype": "float16", "param_type": "output"},
                   0,
                   False
                   ],
        "case_name": "test_vlrelu_precision_yolov3_coco_2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend310", {
        "params": [{"shape": (1, 8, 8, 512), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 8, 8, 512), "dtype": "float16", "param_type": "output"},
                   1,
                   False
                   ],
        "case_name": "test_vlrelu_precision_yolov3_coco_3",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend310", {
        "params": [{"shape": (1, 8, 8, 512), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 8, 8, 512), "dtype": "float16", "param_type": "output"},
                   2,
                   False
                   ],
        "case_name": "test_vlrelu_precision_yolov3_coco_4",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (2, 15, 15, 256), "dtype": "float16", "param_type": "input"},
                   {"shape": (2, 15, 15, 256), "dtype": "float16", "param_type": "output"},
                   1,
                   False
                   ],
        "case_name": "test_vlrelu_precision_yolov3_coco_5",
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
