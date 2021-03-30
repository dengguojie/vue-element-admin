#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)