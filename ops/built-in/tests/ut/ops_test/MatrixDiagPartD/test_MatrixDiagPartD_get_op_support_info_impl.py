#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("MatrixDiagPartD", "impl.matrix_diag_part_d", "get_op_support_info")

case1 = {"params": [{"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_part_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
