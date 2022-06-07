#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("BroadcastToD", "impl.dynamic.broadcast_to_d", "broadcast_to_d")

case1 = {"params": [{"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (-1, ),"ori_format": "ND",
                     "range": [[1, 10]]},
                    {"shape": (3, 3), "dtype": "float32", "format": "ND", "ori_shape": (3, 3),"ori_format": "ND"},
                    (3, 3)],
         "case_name": "broadcast_to_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
