# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_reduce_sum(x, _, axis, keep_dim, kernel_name='dsl_reduce_sum'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.sum(data1, axis, keep_dim)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("reduce_sum", "reduce_sum.test_reduce_sum_impl", "dsl_reduce_sum")


def test_axis_in_none(_):
    try:
        input1 = tvm.placeholder((128, 128), name="input1", dtype="float16")
        tbe.sum(input1, [None])
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_axis_in_none
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (32, 4, 128, 64, 16), "dtype": "float16", "format": "ND"},
               {"shape": (32, 4, 128, 64, 16), "dtype": "float16", "format": "ND"},
               [0, 2, 3],
               True
               ],
    "case_name": "test_reduce_sum_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (32, 8, 256, 64, 16), "dtype": "float32", "format": "ND"},
               {"shape": (8, 16), "dtype": "float32", "format": "ND"},
               [0, 2, 3],
               False
               ],
    "case_name": "test_reduce_sum_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{"shape": (300, 8, 16), "dtype": "float16", "format": "ND"},
               {"shape": (8, 16), "dtype": "float16", "format": "ND"},
               [0,],
               False
               ],
    "case_name": "test_reduce_sum_3",
    "expect": "success",
    "support_expect": True
}
case4 = {
    "params": [{"shape": (64, 8, 8, 8, 16), "dtype": "float32", "format": "ND"},
               {"shape": (64,8, 16), "dtype": "float32", "format": "ND"},
               [2,3],
               False
               ],
    "case_name": "test_reduce_sum_4",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,
    case2,
    case3,
    case4
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, _, axis, keep_dims):
    x_value = x.get("value")
    res = np.sum(x_value, axis=axis, keepdims=keep_dims)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "float16", "param_type": "output"},
                   (2, ),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_att_seq2seq_small",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 256), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "float16", "param_type": "output"},
                   (1, ),
                   True
                   ],
        "case_name": "test_reduce_sum_precision_bert_vector",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 39, 1), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "float16", "param_type": "output"},
                   (1, ),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_DeepFM_frozen_model",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 4, 128), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, ), "dtype": "float16", "param_type": "output"},
                   (1, 2, 3),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_DorefaNet_directSession",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 8631), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "float16", "param_type": "output"},
                   (1, ),
                   True
                   ],
        "case_name": "test_reduce_sum_precision_Facenet",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4), "dtype": "float16", "param_type": "input"},
                   {"shape": (1,), "dtype": "float16", "param_type": "output"},
                   (1, ),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_frozen_graph",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 23), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, 4, 1), "dtype": "float16", "param_type": "output"},
                   (2, ),
                   True
                   ],
        "case_name": "test_reduce_sum_precision_Transformer_directSession",
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
