# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from te.utils import shape_util


warnings.filterwarnings("ignore")


def dsl_multi_out(x1, x2, x3, out1, out2, kernel_name='dsl_multi_out'):
    input_shape1 = x1.get("shape")
    input_dtype1 = x1.get("dtype")
    input_shape2 = x2.get("shape")
    input_dtype2 = x2.get("dtype")
    input_shape3 = x3.get("shape")
    input_dtype3 = x3.get("dtype")

    data1 = tvm.placeholder(input_shape1, name='data1', dtype=input_dtype1)
    data2 = tvm.placeholder(input_shape2, name='data2', dtype=input_dtype2)
    data3 = tvm.placeholder(input_shape3, name='data3', dtype=input_dtype3)

    _, _, shape_max = shape_util.broadcast_shapes(input_shape1,input_shape2,
                                                                param_name_input1="x",
                                                                param_name_input2="y")
    _, _, shape_max2 = shape_util.broadcast_shapes(input_shape3,shape_max,
                                                                param_name_input1="x",
                                                                param_name_input2="y")                                                            

    data1_br = tbe.broadcast(data1, shape_max)
    data2_br = tbe.broadcast(data2, shape_max)

    res_add = tbe.vadd(data1_br, data2_br)
    res_sqrt = tbe.vsqrt(res_add)

    res_sqrt2 = tbe.broadcast(res_sqrt, shape_max2)
    data3_br = tbe.broadcast(data3, shape_max2)

    res = tbe.vmul(res_sqrt2, data3_br)

    tensor_list = [data1, data2, data3, res_sqrt, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule([res_sqrt, res])
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("multi_out", "multi_out.test_multi_out_impl", "dsl_multi_out")

case1 = {
    "params": [{"shape": (1, 1, 1, 1), "dtype": "float16", "format": "ND"},
               {"shape": (32, 149, 1, 1), "dtype": "float16", "format": "ND"},
               {"shape": (32, 149, 16, 32), "dtype": "float16", "format": "ND"},
               {"shape": (32, 149, 16, 32), "dtype": "float16", "format": "ND"},
               {"shape": (32, 149, 16, 32), "dtype": "float16", "format": "ND"}
               ],
    "case_name": "test_multi_out_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (2, 1, 1, 1), "dtype": "float32", "format": "ND"},
               {"shape": (2, 149, 16, 16), "dtype": "float32", "format": "ND"},
               {"shape": (2, 149, 16, 16), "dtype": "float32", "format": "ND"},
               {"shape": (2, 149, 16, 16), "dtype": "float32", "format": "ND"},
               {"shape": (2, 149, 16, 16), "dtype": "float32", "format": "ND"}
               ],
    "case_name": "test_multi_out_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{"shape": (5, 5, 1), "dtype": "float32", "format": "ND"},
               {"shape": (5, 5, 1), "dtype": "float32", "format": "ND"},
               {"shape": (5, 5, 512), "dtype": "float32", "format": "ND"},
               {"shape": (5, 5, 1), "dtype": "float32", "format": "ND"},
               {"shape": (5, 5, 512), "dtype": "float32", "format": "ND"}
               ],
    "case_name": "test_multi_out_3",
    "expect": "success",
    "support_expect": True
}

compile_case = [
    case1,
    case2,
    case3
]
for item in compile_case:
    ut_case.add_case(case=item)


def calc_expect_func(x, y, z, out, out1):
    x_value = x.get("value")
    y_value = y.get("value")
    z_value = z.get("value")
    res_add = np.add(x_value, y_value)
    res = np.sqrt(res_add)
    res2 = np.multiply(res, z_value)

    return (res, res2)


ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1, 1, 1, 1), "dtype": "float16", "value_range": [0,2],"param_type": "input"},
                   {"shape": (32, 149, 1, 1), "dtype": "float16", "value_range": [0,2],"param_type": "input"},
                   {"shape": (32, 149, 16, 32), "dtype": "float16", "value_range": [0,2],"param_type": "input"},
                   {"shape": (32, 149, 1, 1), "dtype": "float16", "param_type": "output"},
                   {"shape": (32, 149, 16, 32), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_multi_out_precision_1",
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
