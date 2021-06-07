# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_reduce_5hdc_sum(x, _, axis, keep_dim, kernel_name='dsl_reduce_5hdc_sum'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_format = x.get("format")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.sum(data1, axis, keep_dim)

    if input_format == "NC1HWC0":
        res.ori_shape = x["ori_shape"]
        res.ori_format = x["ori_format"]

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("reduce_sum", "reduce_sum.test_reduce_5hdc_sum_impl", "dsl_reduce_5hdc_sum")


case1 = {
    "params": [{"shape": (1, 1, 6, 7, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 6, 7, 5), "ori_format": "NHWC"},
               {"shape": (1, 1, 6, 7, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 6, 7, 1), "ori_format": "NHWC"},
               [1, 4],
               True
               ],
    "case_name": "test_reduce_5hdc_sum_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [{"shape": (16, 444, 128, 11, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 128, 11, 7103), "ori_format": "NHWC"},
               {"shape": (16, 444, 128, 11, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 128, 11, 7103), "ori_format": "NHWC"},
               [1, 4],
               True
               ],
    "case_name": "test_reduce_5hdc_sum_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [{"shape": (64, 16, 64, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 1024, 256), "ori_format": "ND"},
               {"shape": (16, 1024, 1), "dtype": "float16", "format": "ND", "ori_shape": (16, 1024, 1), "ori_format": "ND"},
               [1, 4],
               True
               ],
    "case_name": "test_reduce_5hdc_sum_3",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,case2,case3
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
