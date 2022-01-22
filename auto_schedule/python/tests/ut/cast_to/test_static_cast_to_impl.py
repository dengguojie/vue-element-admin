# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import warnings

from te import tvm
import tbe

warnings.filterwarnings("ignore")


def dsl_cast_to(x, _, dest_type, f16_8_flag, kernel_name='dsl_cast_to'):
    with tbe.common.context.op_context.OpContext("static"):
        input_shape = x.get("shape")
        input_dtype = x.get("dtype")

        data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
        with tbe.dsl.compute():
            res = tbe.dsl.cast_to(data1, dest_type, f16_8_flag)
            tensor_list = [data1, res]
            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            config = {
                "print_ir": False,
                "name": kernel_name,
                "tensor_list": tensor_list,
                "save_temp_cce_file": True
            }
            tbe.dsl.build(sch, config)


ut_case = OpUT("cast_to", "cast_to.test_static_cast_to_impl", "dsl_cast_to")


def test_input_not_tensor(_):
    try:
        input1 = tvm.const(1, dtype="float16")
        tbe.dsl.cast_to(input1, "float32")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_input_not_tensor,
]
for item in test_func_list:
    ut_case.add_cust_test_func(support_soc=["Ascend910A"], test_func=item)

case1 = {"params": [{"shape": (1, 227, 227, 3), "dtype": "int64", "format": "ND"},
                    {"shape": (1, 227, 227, 3), "dtype": "int32", "format": "ND"},
                    "int32",
                    False
                    ],
         "case_name": "test_cast_to_s64_s32",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (1, 227, 227, 3), "dtype": "int32", "format": "ND"},
                    {"shape": (1, 227, 227, 3), "dtype": "int64", "format": "ND"},
                    "int64",
                    False
                    ],
         "case_name": "test_cast_to_s32_s64",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [{"shape": (1, 227, 227, 3), "dtype": "bfloat16", "format": "ND"},
                    {"shape": (1, 227, 227, 3), "dtype": "float32", "format": "ND"},
                    "float32",
                    False
                    ],
         "case_name": "test_cast_to_bf16_f32",
         "expect": "success",
         "support_expect": True
         }

compile_case_list = {
    # "1": [case1, "Ascend920A"],
    # "2": [case2, "Ascend920A"],
    # "3": [case3, "Ascend920A"],
}
for _, item in compile_case_list.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _, dest_type, __):
    x_value = x.get("value")
    output = x_value.astype(dest_type)
    return output


ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1, 1), "dtype": "float32", "param_type": "input"},
                   {"shape": (1, 1), "dtype": "int32", "param_type": "output"},
                   "int32",
                   False
                   ],
        "case_name": "test_cast_to_precision_fp322s32_faster_rcnn_resnet",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1024, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (1024, ), "dtype": "int32", "param_type": "output"},
                   "int32",
                   False
                   ],
        "case_name": "test_cast_to_precision_fp162s32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1024, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (1024, ), "dtype": "int8", "param_type": "output"},
                   "int8",
                   False
                   ],
        "case_name": "test_cast_to_precision_fp162int8",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1024, ), "dtype": "int32", "param_type": "input"},
                   {"shape": (1024, ), "dtype": "float16", "param_type": "output"},
                   "float16",
                   False
                   ],
        "case_name": "test_cast_to_precision_int322fp16",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (17810, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (17810, ), "dtype": "int32", "param_type": "output"},
                   "int32",
                   False
                   ],
        "case_name": "test_cast_to_precision_fp162int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (602112,), "dtype": "uint8", "param_type": "input"},
                   {"shape": (602112,), "dtype": "float32", "param_type": "output"},
                   "float32",
                   False
                   ],
        "case_name": "test_cast_to_precision_u82fp32_frozen_inference_graph2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (4, 15, 15, 3, 2), "dtype": "int32", "param_type": "input"},
                   {"shape": (4, 15, 15, 3, 2), "dtype": "float32", "param_type": "output"},
                   "float32",
                   False
                   ],
        "case_name": "test_cast_to_precision_s322fp32_yolov3_coco",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (16, ), "dtype": "uint8", "param_type": "input"},
                   {"shape": (16, ), "dtype": "int32", "param_type": "output"},
                   "int32",
                   False
                   ],
        "case_name": "test_cast_to_precision_u82s32_train_FreeSpace",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
