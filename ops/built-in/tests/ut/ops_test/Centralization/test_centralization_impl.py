# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os
ut_case = OpUT("Centralization", None, None)

def calc_expect_func_fz(x, y, dimension):
    x_shape = x.get("shape")
    x_value = x.get("value")
    output_data=np.mean(x_value,axis=0,keepdims=True)
    output_data=np.mean(output_data,axis=3,keepdims=True)
    result = x_value - output_data

    return (result,)


case_1 = {"params": [
    {"shape": (7*7*16,1,16,16), "dtype": "float32", "ori_shape": (16,7,7,256), "ori_format": "NHWC", "format": "FRACTAL_Z", "param_type": "input"},
    {"shape": (7*7*16,1,16,16), "dtype": "float32", "ori_shape": (16,7,7,256), "ori_format": "NHWC", "format": "NHWC", "param_type": "output"},
    [1, 2, 3],
    ],
    "calc_expect_func": calc_expect_func_fz,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    }

ut_case.add_precision_case("Ascend910A", case_1)


def calc_expect_func_nz(x, y, dimension):
    x_shape = x.get("shape")
    x_value = x.get("value")
    output_data=np.mean(x_value,axis=1,keepdims=True)
    output_data=np.mean(output_data,axis=2,keepdims=True)
    result = x_value - output_data

    return (result,)

case_2 = {"params": [
    {"shape": (63, 2048//16, 16, 16), "dtype": "float32", "ori_shape": (2048, 1000), "ori_format": "NHWC", "format": "FRACTAL_NZ", "param_type": "input"},
    {"shape": (63, 2048//16, 16, 16), "dtype": "float32", "ori_shape": (2048, 1000), "ori_format": "NHWC", "format": "FRACTAL_NZ", "param_type": "output"},
    [0],
    ],
    "calc_expect_func": calc_expect_func_nz,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    }

ut_case.add_precision_case("Ascend910A", case_2)


def calc_expect_func(x, y, dimension):
    x_shape = x.get("shape")
    x_value = x.get("value")
    output_data=np.mean(x_value,axis=dimension[0],keepdims=True)
    for axis in dimension[1:]:
        output_data=np.mean(output_data,axis=axis,keepdims=True)
    result = x_value - output_data

    return (result,)

case_3 = {"params": [
    {"shape": (16, 16), "dtype": "float32", "ori_shape": (16, 16), "ori_format": "NHWC", "format": "NHWC", "param_type": "input"},
    {"shape": (16, 16), "dtype": "float32", "ori_shape": (16, 16), "ori_format": "NHWC", "format": "NHWC", "param_type": "output"},
    [0],
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    }
ut_case.add_precision_case("Ascend910A", case_3)


case_4 = {"params": [
    {"shape": (7*7*16,1,16,16), "dtype": "float16", "ori_shape": (16,7,7,256), "ori_format": "NHWC", "format": "FRACTAL_Z", "param_type": "input"},
    {"shape": (7*7*16,1,16,16), "dtype": "float16", "ori_shape": (16,7,7,256), "ori_format": "NHWC", "format": "NHWC", "param_type": "output"},
    [1, 2, 3],
    ],
    "case_name": "FRACTAL_Z_fp16",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case_4)

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, "Ascend/ascend-toolkit/20.1.rc1/x86_64-linux/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

