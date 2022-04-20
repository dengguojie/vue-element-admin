#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.compress_mat_mul import get_op_support_info
from op_test_frame.ut import OpUT

ut_case = OpUT("CompressMatMul", None, None)

# reference to lenet
case1 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# maybe "expect fail" trans_a=true
case2 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    True, False],
         "case_name": "MatMulV2Compress_2",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}

# maybe "expect fail" trans_b=true
case3 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, True],
         "case_name": "MatMulV2Compress_3",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}

# shape_bias[0]%16 != 0
case4 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1023, ), "dtype": "int32", "format": "ND", "ori_shape": (1023, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# input_x1[0]%16 == 0
case5 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (16,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# input_x1[1]%32 != 0
case6 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3137),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3137,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# input_x2[1]%16 != 0
case7 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1023),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# input_x1.get("ori_shape").length < 2    input_x2.get("ori_shape").length < 2
case8 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (3136,),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    None,
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# offset_w is not None
case9 = {"params": [{"shape": (98,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,3136),"ori_format": "ND"},
                    {"shape": (98,64,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (3136,1024),"ori_format": "HWCN"},
                    {"shape": (1568, ), "dtype": "int8", "format": "ND", "ori_shape": (1568, ), "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "int32", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    {"shape": (64,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1024),"ori_format": "NHWC"},
                    False, False],
         "case_name": "MatMulV2Compress_9",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], case4)
ut_case.add_case(["Ascend310"], case5)
ut_case.add_case(["Ascend310"], case6)
ut_case.add_case(["Ascend310"], case7)
ut_case.add_case(["Ascend310"], case8)
ut_case.add_case(["Ascend310"], case9)

# case from vgg16
case10 = {"params": [{"shape": (784,19,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (304,25088),"ori_format": "ND"},
                    {"shape": (784,256,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (25088,4096),"ori_format": "HWCN"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                    {"shape": (4096, ), "dtype": "int32", "format": "ND", "ori_shape": (4096, ),"ori_format": "ND"},
                    None,
                    {"shape": (256,19,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (304,4096),"ori_format": "ND"},
                    False, False],
         "case_name": "MatMulV2Compress_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend710", "SD3403", "Ascend910A"], case10)

# case from vgg16
case11 = {"params": [{"shape": (128,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (1,4096),"ori_format": "ND"},
                    {"shape": (128,63,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (4096,1008),"ori_format": "HWCN"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                    None,
                    None,
                    {"shape": (63,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1,1008),"ori_format": "ND"},
                    False, False],
         "case_name": "MatMulV2Compress_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend710", "SD3403", "Ascend310"], case11)

# batch_matmul
case12 = {"params": [{"shape": (2,784,19,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,304,25088),"ori_format": "ND"},
                    {"shape": (2,784,256,16,32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (2,25088,4096),"ori_format": "HWCN"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                    {"shape": (4096, ), "dtype": "int32", "format": "ND", "ori_shape": (4096, ),"ori_format": "ND"},
                    None,
                    {"shape": (2,256,19,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,304,4096),"ori_format": "ND"},
                    False, False],
         "case_name": "MatMulV2Compress_12",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend910A"], case12)

def test_split_matmul(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "int8", "shape": (1, 1, 16, 32), "ori_shape": (16, 32)}
    x2 = {"format": "FRACTAL_Z","ori_format": "ND", "dtype": "int8", "shape": (1, 1, 16, 32), "ori_shape": (32, 16)}
    compress_index = {"format": "ND","ori_format": "ND", "dtype": "int8", "shape": (1,), "ori_shape": (1, )}
    bias = {"shape": (16, ), "dtype": "int32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"}
    y = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "int32", "shape": (1, 1, 16, 16), "ori_shape": (16, 16)}
    get_op_support_info(x1, x2, compress_index, bias, None, output_y=y)
ut_case.add_cust_test_func(test_func=test_split_matmul)

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
