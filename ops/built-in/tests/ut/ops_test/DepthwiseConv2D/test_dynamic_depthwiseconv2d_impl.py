#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2D", "impl.dynamic.depthwise_conv2d", "depthwise_conv2d")

depthwise_conv2d_dynamic_ut_testcase = [
    # ============ success =====================
    ["all", {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [1, 32, 1, 1], "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (32, 32), (1, 1), (1, 1)]}, None, None, {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, "success"],
    ["all", {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [1, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 5), (32, 32), (16, 16), (16, 16)]}, {"ori_shape": [1, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (16, 16), (16, 16)]}, {"ori_shape": [1, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, 16, 16), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, "success"],
    ["all", {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (10, 25), (10, 25)]}, {"ori_shape": [1, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (32, 32), (3, 3), (3, 3)]}, None, None, {'ori_shape': (-1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, "success"],
    # ============ test_conv2d_invalid_stride_shape ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_dilation_range ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1), "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_format ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format": "NCHW11", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, RuntimeError],
    # ============ test_conv2d_invalid_weight_format_shape ===============
    ["all", {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format": "NCHW", "range": [(1, 1), (16, 16), (3, 40), (3, 40)]}, {"ori_shape": (32, 16, 3, 3), "dtype": "float16", "ori_format": "NCHW1111", "range": [(32, 32), (16, 16), (3, 3), (3, 3)]}, None, None, {'ori_shape': (1, 16, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'}, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), "NCHW", 0, RuntimeError],
]

# def gen_trans_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, data_format, offset_x, expect):
#     return {"params": [inputs, weights, bias, offset_w, outputs, strides, dilations, pads, data_format, offset_x],
#             "case_name": "dynamic_depthwise_conv2d_case",
#             "expect": expect
#             }

# print("adding DepthwiseConv2d dyanmic op testcases")
# for test_case  in depthwise_conv2d_dynamic_ut_testcase:
#     ut_case.add_case(test_case[0], gen_trans_data_case(*test_case[1:]))

# fuzz build ut case
def test_depthwise_fuzz_build_case_01(test_arg):
    from impl.dynamic.depthwise_conv2d import depthwise_conv2d_generalization
    input_list = [
        {"ori_shape": (2, 32, 56, 56),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        {"ori_shape": (2, 2, 32, 1),
         "ori_format": "HWCN",
         "dtype": "float16"
        },
        None,
        None,
        {"ori_shape": (2, 32, 28, 28),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", 0, "depthwise_fuzz_generalization"
    ]
    depthwise_conv2d_generalization(*input_list)
print("add depthwise fuzzbuild ut case test_depthwise_fuzz_build_case_01")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_case_01)

def test_depthwise_fuzz_build_case_02(test_arg):
    from impl.dynamic.depthwise_conv2d import depthwise_conv2d_generalization
    input_list = [
        {"ori_shape": (2, 32, -1, 56),
         "ori_format": "NCHW",
         "dtype": "float16",
         "ori_range": [[1,-1], [32, 32], [30, 78], [1, -1]]
        },
        {"ori_shape": (2, 2, 32, 1),
         "ori_format": "HWCN",
         "dtype": "float16"
        },
        {"ori_shape": (32,),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        None,
        {"ori_shape": (2, 32, -1, 28),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", 0, "depthwise_fuzz_generalization"
    ]
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        pass
print("add depthwise fuzzbuild ut case test_depthwise_fuzz_build_case_02")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_case_02)

def test_depthwise_fuzz_build_case_03(test_arg):
    from impl.dynamic.depthwise_conv2d import depthwise_conv2d_generalization
    input_list = [
        {"ori_shape": (2, 32, -1, -1),
         "ori_format": "NCHW",
         "dtype": "float16",
         "ori_range": [[1,-1], [32, 32], [30, 78], [1, -1]]
        },
        {"ori_shape": (2, 2, 32, 1),
         "ori_format": "HWCN",
         "dtype": "float16"
        },
        {"ori_shape": (32,),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        None,
        {"ori_shape": (2, 32, -1, 28),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), "NCHW", 0, "depthwise_fuzz_generalization"
    ]
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        pass
print("add depthwise fuzzbuild ut case test_depthwise_fuzz_build_case_03")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_case_03)


# def test_depthwise_fuzz_build_case_correct_range(test_arg):
#     from impl.dynamic.depthwise_conv2d import depthwise_conv2d
#     from tbe.common.context import op_context
#     with op_context.OpContext("dynamic"):
#         input_list = [
#             {"ori_shape": (2, 32, 1080, -1),
#             "shape": (2, 2, 1080, -1, 16),
#             "ori_format": "NCHW",
#             "format": "NC1HW0",
#             "dtype": "float16",
#             "ori_range": [[1,-1], [32, 32], [30, 78], [1, None]],
#             "range": [[1,-1], [32, 32], [30, 78], [1, None]]
#             },
#             {"ori_shape": (28, 14, 32, 1),
#             "ori_format": "HWCN",
#             "dtype": "float16"
#             },
#             {"ori_shape": (32,),
#             "ori_format": "NCHW",
#             "dtype": "float16"
#             },
#             None,
#             {"ori_shape": (2, 32, 1080, 2056),
#             "ori_format": "NCHW",
#             "dtype": "float16"
#             },
#             (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", 0, "depthwise_fuzz_generalization"
#         ]
#         depthwise_conv2d(*input_list)
# print("add depthwise fuzzbuild ut case test_depthwise_fuzz_build_case_correct_range")
# ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_case_correct_range)


if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310"])
