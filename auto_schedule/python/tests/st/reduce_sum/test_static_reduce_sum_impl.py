# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import tbe
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common.context import op_info

warnings.filterwarnings("ignore")


def dsl_reduce_sum(x, _, axis, keepdims, kernel_name='dsl_reduce_sum'):
    with tbe.common.context.op_context.OpContext("static") as f:
        opInfo = op_info.OpInfo("reduce_prod1", "reduce_sum")
        f.add_op_info(opInfo)

        input_dtype = x.get("dtype")
        x["rel_pos_to_reduce"] = 'before'
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
        ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})
        schedules, tensors = [], []

        for (x, axis) in ins:
            print("x axis", x , axis)
            with tbe.dsl.compute():
                shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
                data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                res = tbe.dsl.reduce_sum(data1, axis.get("value"), keepdims)
                tensors.append([data1, res])

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)


ut_case = OpUT("reduce_sum", "reduce_sum.test_static_reduce_sum_impl", "dsl_reduce_sum")


def test_axis_in_none(_):
    try:
        input1 = tvm.placeholder((128, 128), name="input1", dtype="float16")
        tbe.dsl.reduce_sum(input1, [None])
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_axis_in_none
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


case1 = {
    "params": [{"shape": (1, 112640000), "dtype": "float16", "format": "ND"},
               {"shape": (5, ), "dtype": "float16", "format": "ND"},
               [-1],
               True
               ],
    "case_name": "test_reduce_sum_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (32, 8, 256, 64, 16), "dtype": "int32", "format": "ND"},
               {"shape": (8, 16), "dtype": "int32", "format": "ND"},
               [0, 2, 3],
               False
               ],
    "case_name": "test_reduce_sum_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{"shape": (300, 8, 16), "dtype": "int32", "format": "ND"},
               {"shape": (8, 16), "dtype": "int32", "format": "ND"},
               [0,],
               False
               ],
    "case_name": "test_reduce_sum_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [{"shape": (64, 8, 8, 8, 16), "dtype": "int32", "format": "ND"},
               {"shape": (64, 8, 16), "dtype": "int32", "format": "ND"},
               [2,3],
               False
               ],
    "case_name": "test_reduce_sum_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [{"shape": (16, 32, 512, 512), "dtype": "int32", "format": "ND"},
               {"shape": (32, ), "dtype": "int32", "format": "ND"},
               [0,2,3],
               False
               ],
    "case_name": "test_reduce_sum_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [{"shape": (16, 31, 511, 511), "dtype": "int32", "format": "ND"},
               {"shape": (511, ), "dtype": "int32", "format": "ND"},
               [0,1,2],
               False
               ],
    "case_name": "test_reduce_sum_6",
    "expect": "success",
    "support_expect": True
}
case7 = {
    "params": [{"shape": (16, 32, 512, 512), "dtype": "int32", "format": "ND"},
               {"shape": (512, ), "dtype": "int32", "format": "ND"},
               [0,1,2,3],
               False
               ],
    "case_name": "test_reduce_sum_7",
    "expect": "success",
    "support_expect": True
}
case8 = {
    "params": [{"shape": (3200, 32, 20, 16), "dtype": "int32", "format": "ND"},
               {"shape": (32, ), "dtype": "int32", "format": "ND"},
               [0,2,3],
               False
               ],
    "case_name": "test_reduce_sum_8",
    "expect": "success",
    "support_expect": True
}
case9 = {
    "params": [{"shape": (32, 32, 1600), "dtype": "int32", "format": "ND"},
               {"shape": (32, ), "dtype": "int32", "format": "ND"},
               [0,2],
               False
               ],
    "case_name": "test_reduce_sum_9",
    "expect": "success",
    "support_expect": True
}
case10 = {
    "params": [{"shape": (32, 32, 1600), "dtype": "int32", "format": "ND"},
               {"shape": (1, ), "dtype": "int32", "format": "ND"},
               [0,1,2],
               False
               ],
    "case_name": "test_reduce_sum_10",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8,
    case9,
    case10,
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, _, axis, keep_dims):
    x_value = x.get("value")
    res = np.sum(x_value, axis=axis, keepdims=keep_dims)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 128), "dtype": "int32", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "int32", "param_type": "output"},
                   (2, ),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_att_seq2seq_small_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 256), "dtype": "int32", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "int32", "param_type": "output"},
                   (1, ),
                   True
                   ],
        "case_name": "test_reduce_sum_precision_bert_vector_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 39, 1), "dtype": "int32", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "int32", "param_type": "output"},
                   (1, ),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_DeepFM_frozen_model_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 4, 128), "dtype": "int32", "param_type": "input"},
                   {"shape": (4, ), "dtype": "int32", "param_type": "output"},
                   (1, 2, 3),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_DorefaNet_directSession_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 8631), "dtype": "int32", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "int32", "param_type": "output"},
                   (1, ),
                   True
                   ],
        "case_name": "test_reduce_sum_precision_Facenet_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4), "dtype": "int32", "param_type": "input"},
                   {"shape": (1,), "dtype": "int32", "param_type": "output"},
                   (1, ),
                   False
                   ],
        "case_name": "test_reduce_sum_precision_frozen_graph_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 23), "dtype": "int32", "param_type": "input"},
                   {"shape": (4, 4, 1), "dtype": "int32", "param_type": "output"},
                   (2, ),
                   True
                   ],
        "case_name": "test_reduce_sum_precision_Transformer_directSession_int32",
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
