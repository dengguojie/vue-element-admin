#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("L2Loss", "impl.l2_loss", "op_select_format")

case1 = {"params": [{"shape": (1024, 1024, 256, 256), "dtype": "float", "format": "NCHW", "ori_shape": (1024, 1024, 256, 256),"ori_format": "NCHW"},
                    {"shape": (1024, 1024, 256, 256), "dtype": "float", "format": "NCHW", "ori_shape": (1024, 1024, 256, 256),"ori_format": "NCHW"}],
         "case_name":"l2loss_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)


