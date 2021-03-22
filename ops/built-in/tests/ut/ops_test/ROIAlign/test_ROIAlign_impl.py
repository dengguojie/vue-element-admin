# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("ROIAlign", "impl.roi_align", "roi_align")

case1 = {"params": [{"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                     "ori_format": "NHWC"},
                    {"shape": (256, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),
                     "ori_format": "NHWC"},
                    None,
                    {"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (2, 260, 1, 1, 16), "ori_format": "NHWC"},
                    0.25,
                    7,
                    7,
                    2,
                    1
                    ],
         "case_name": "roi_align_01",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (1, 16, 380, 6400, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 256, 380, 6400),
     "ori_format": "NHWC"},
    {"shape": (56, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5), "ori_format": "NHWC"},
    None,
    {"shape": (56, 16, 7, 7, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (56, 256, 7, 7),
     "ori_format": "NHWC"},
    0.25,
    7,
    7,
    2,
    1
    ],
         "case_name": "roi_align_02",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [
    {"shape": (1, 16, 380, 6400, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 256, 380, 6400),
     "ori_format": "NHWC"},
    {"shape": (56, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5), "ori_format": "NHWC"},
    None,
    {"shape": (56, 16, 7, 7, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (56, 256, 7, 7),
     "ori_format": "NHWC"},
    0.25,
    7,
    7,
    2,
    0
    ],
         "case_name": "roi_align_03",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1, 16, 38, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                     "ori_format": "NHWC"},
                    {"shape": (256, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (256, 5),
                     "ori_format": "NHWC"},
                    None,
                    {"shape": (1, 16, 38, 64, 16), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (2, 260, 1, 1, 16), "ori_format": "NHWC"},
                    0.25,
                    7,
                    7,
                    2,
                    1
                    ],
         "case_name": "roi_align_04",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [
    {"shape": (1, 16, 380, 6400, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 256, 380, 6400),
     "ori_format": "NHWC"},
    {"shape": (56, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (256, 5), "ori_format": "NHWC"},
    None,
    {"shape": (56, 16, 7, 7, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (56, 256, 7, 7),
     "ori_format": "NHWC"},
    0.25,
    7,
    7,
    2,
    1
    ],
         "case_name": "roi_align_05",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [
    {"shape": (1, 16, 380, 6400, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 256, 380, 6400),
     "ori_format": "NHWC"},
    {"shape": (56, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (256, 5), "ori_format": "NHWC"},
    None,
    {"shape": (56, 16, 7, 7, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (56, 256, 7, 7),
     "ori_format": "NHWC"},
    0.25,
    7,
    7,
    2,
    0
    ],
         "case_name": "roi_align_06",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (3, 1, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (3, 1, 5, 5),
                     "ori_format": "NHWC"},
                    {"shape": (3, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 5), "ori_format": "NHWC"},
                    {"shape": (3,), "dtype": "float16", "format": "NHWC", "ori_shape": (3,), "ori_format": "NHWC"},
                    {"shape": (1, 1, 10, 10, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 10, 10),
                     "ori_format": "NHWC"},
                    0.25,
                    5,
                    5,
                    2,
                    0
                    ],
         "case_name": "roi_align_07",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
