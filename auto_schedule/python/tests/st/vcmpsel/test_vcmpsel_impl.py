# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vcmpsel(x1, x2, y1, y2, _, operation, kernel_name='dsl_vcmpsel'):
    input_dtype = x1.get("dtype")
    input_shape = x1.get("shape")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    tensor_list = [data1]
    # x1 must be tensor, x2, y1, y2 can be tensor or scalar or None
    if not isinstance(x2, dict):
        data2 = x2
    else:
        input2_dtype = x2.get("dtype")
        input2_shape = x2.get("shape")
        data2 = tvm.placeholder(input2_shape, name='data2', dtype=input2_dtype)
        tensor_list.append(data2)

    if not isinstance(y1, dict):
        data3 = y1
    else:
        input3_dtype = y1.get("dtype")
        input3_shape = y1.get("shape")
        data3 = tvm.placeholder(input3_shape, name='data3', dtype=input3_dtype)
        tensor_list.append(data3)

    if not isinstance(y2, dict):
        data4 = y2
    else:
        input4_dtype = y2.get("dtype")
        input4_shape = y2.get("shape")
        data4 = tvm.placeholder(input4_shape, name='data4', dtype=input4_dtype)
        tensor_list.append(data4)

    res = tbe.vcmpsel(data1, data2, operation, data3, data4)
    tensor_list.append(res)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "save_temp_cce_file": True
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vcmpsel", "vcmpsel.test_vcmpsel_impl", "dsl_vcmpsel")


def test_lhs_is_not_tensor(_):
    try:
        input1 = tvm.const(2, dtype="bool")
        tbe.vcmpsel(input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_operation_is_not_valid(_):
    try:
        input1 = tvm.placeholder((128, ), name='data4', dtype="float16")
        tbe.vcmpsel(input1, operation="ee")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_lhs_is_not_tensor, None],
    "2": [test_operation_is_not_valid, None],
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])


case1 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_tensor_tensor_tensor",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    None,
                    None,
                    None,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_scalar_tensor_scalar",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    None,
                    1.0,
                    None,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_scalar_scalar_scalar",
         "expect": "success",
         "support_expect": True
         }

case4 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    1.0,
                    2.0,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_tensor_scalar_scalar",
         "expect": "success",
         "support_expect": True
         }

case5 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    None,
                    1.0,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_scalar_scalar_tensor",
         "expect": "success",
         "support_expect": True
         }

case6 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    2.0,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_tensor_tensor_scalar",
         "expect": "success",
         "support_expect": True
         }

case7 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    2.0,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_tensor_scalar_tensor",
         "expect": "success",
         "support_expect": True
         }

case8 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    None,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    "gt"
                    ],
         "case_name": "test_vcmpsel_scalar_tensor_tensor",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case2, ["Ascend910A", "Ascend710", "Ascend310"]],
    "3": [case3, ["Ascend910A", "Ascend710", "Ascend310"]],
    "4": [case4, ["Ascend910A", "Ascend710", "Ascend310"]],
    "5": [case5, None],
    "6": [case6, ["Ascend910A", "Ascend710", "Ascend310"]],
    "7": [case7, None],
    "8": [case8, None],
}
for _, item in compile_case.items():
    support_soc = ["Ascend910A", "Ascend310"],
    ut_case.add_case(case=item[0], support_soc=support_soc)


def calc_expect_func(x1, x2, y1, y2, z, operation):
    x1_value = x1.get("value")
    output_shape = z.get("shape")
    output_dtype = z.get("dtype")
    # x1 must be tensor, x2, y1, y2 can be tensor or scalar or None
    if x2 is None:
        x2_value = np.full(output_shape, 2.0)
    elif isinstance(x2, dict):
        x2_value = x2.get("value")
    else:
        x2_value = np.full(output_shape, x2)

    if y1 is None:
        y1_value = x1_value
    elif isinstance(y1, dict):
        y1_value = y1.get("value")
    else:
        y1_value = np.full(output_shape, y1)

    if y2 is None:
        if isinstance(x2, dict):
            y2_value = x2_value
        else:
            y2_value = np.full(output_shape, 0.0)
    elif isinstance(y2, dict):
        y2_value = y2.get("value")
    else:
        y2_value = np.full(output_shape, y2)

    benchmark_data = np.zeros(output_shape, dtype=output_dtype)
    import operator
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}

    for i in range(0, output_shape[0]):
        for j in range(0, output_shape[1]):
            if op_dict.get(operation)(x1_value[i][j], x2_value[i][j]):
                benchmark_data[i][j] = y1_value[i][j]
            else:
                benchmark_data[i][j] = y2_value[i][j]

    res = np.array(benchmark_data)
    return res

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
