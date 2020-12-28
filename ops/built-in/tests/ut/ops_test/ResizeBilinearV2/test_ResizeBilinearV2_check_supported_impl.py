
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("ResizeBilinearV2D", "impl.resize_bilinear_v2_d", "check_supported")

case1 = {"params": [{"shape": (2,260,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2,260,1,16), "dtype": "float32", "format": "NHC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,2, 2,16), "dtype": "float32", "format": "NHC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (-2,), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

