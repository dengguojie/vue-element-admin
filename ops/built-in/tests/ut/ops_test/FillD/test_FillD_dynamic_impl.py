#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("FillD", "impl.dynamic.fill_d", "fill_d")

case1 = {"params": [{"shape": (-1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32", "range":[[1, 10]]},
                    {"shape": (2, 2), "ori_shape": (2, 2), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32", "range": [[2, 2], [2, 2]]},
                    (2, 2)],
         "case_name": "fill_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
