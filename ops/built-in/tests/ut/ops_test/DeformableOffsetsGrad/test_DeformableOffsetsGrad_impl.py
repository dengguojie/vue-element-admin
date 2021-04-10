#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
ut_case = OpUT("DeformableOffsetsGrad", "impl.deformable_offsets_grad", "deformable_offsets_grad")

case1 = {"params": [{"shape": (8,48,48,64), "dtype": "float32", "format": "NHWC", "ori_shape": (8,48,48,64),"ori_format": "NHWC"},
                    {"shape": (8,16,16,64), "dtype": "float32", "format": "NHWC", "ori_shape": (8,16,16,64),"ori_format": "NHWC"},
                    {"shape": (8,16,16,27), "dtype": "float32", "format": "NHWC", "ori_shape": (8,16,16,27),"ori_format": "NHWC"},
                    {"shape": (1,16,16,27), "dtype": "float32", "format": "NHWC", "ori_shape": (1,16,16,27),"ori_format": "NHWC"},
                    {"shape": (8,16,16,64), "dtype": "float32", "format": "NHWC", "ori_shape": (8,16,16,64),"ori_format": "NHWC"},
                    {"shape": (8,16,16,27), "dtype": "float32", "format": "NHWC", "ori_shape": (8,16,16,27),"ori_format": "NHWC"},
                    [1,1,1,1],[1,1,1,1],[3,3],[1,1,1,1],"NHWC",1,True],
         "case_name": "DeformableOffsetsGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (32,48,48,64), "dtype": "float32", "format": "NHWC", "ori_shape": (32,48,48,64),"ori_format": "NHWC"},
                    {"shape": (32,16,16,64), "dtype": "float32", "format": "NHWC", "ori_shape": (32,16,16,64),"ori_format": "NHWC"},
                    {"shape": (32,16,16,27), "dtype": "float32", "format": "NHWC", "ori_shape": (32,16,16,27),"ori_format": "NHWC"},
                    {"shape": (1,16,16,27), "dtype": "float32", "format": "NHWC", "ori_shape": (1,16,16,27),"ori_format": "NHWC"},
                    {"shape": (32,16,16,64), "dtype": "float32", "format": "NHWC", "ori_shape": (32,16,16,64),"ori_format": "NHWC"},
                    {"shape": (32,16,16,27), "dtype": "float32", "format": "NHWC", "ori_shape": (32,16,16,27),"ori_format": "NHWC"},
                    [1,1,1,1],[1,1,1,1],[3,3],[1,1,1,1],"NHWC",1,True],
         "case_name": "DeformableOffsetsGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (8,192,192,64), "dtype": "float32", "format": "NHWC", "ori_shape": (8,192,192,64),"ori_format": "NHWC"},
                    {"shape": (8,64,64,64), "dtype": "float32", "format": "NHWC", "ori_shape": (8,64,64,64),"ori_format": "NHWC"},
                    {"shape": (8,64,64,27), "dtype": "float32", "format": "NHWC", "ori_shape": (8,64,64,27),"ori_format": "NHWC"},
                    {"shape": (1,64,64,27), "dtype": "float32", "format": "NHWC", "ori_shape": (1,64,64,27),"ori_format": "NHWC"},
                    {"shape": (8,64,64,64), "dtype": "float32", "format": "NHWC", "ori_shape": (8,64,64,64),"ori_format": "NHWC"},
                    {"shape": (8,64,64,27), "dtype": "float32", "format": "NHWC", "ori_shape": (8,64,64,27),"ori_format": "NHWC"},
                    [1,1,1,1],[1,1,1,1],[3,3],[1,1,1,1],"NHWC",1,True],
         "case_name": "DeformableOffsetsGrad_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)