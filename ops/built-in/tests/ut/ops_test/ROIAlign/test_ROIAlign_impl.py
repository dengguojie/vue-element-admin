# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from te import platform as cce_conf
from impl.roi_align import roi_align
from tbe.common.platform.platform_info import set_current_compile_soc_info

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

case8 = {"params": [{"shape": (3, 1, 5, 5, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (3, 1, 5, 5),
                     "ori_format": "NHWC"},
                    {"shape": (3, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 5), "ori_format": "NHWC"},
                    {"shape": (3,), "dtype": "float32", "format": "NHWC", "ori_shape": (3,), "ori_format": "NHWC"},
                    {"shape": (1, 1, 10, 10, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 10, 10),
                     "ori_format": "NHWC"},
                    0.25,
                    5,
                    5,
                    2,
                    2
                    ],
         "case_name": "roi_align_08",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (1, 16, 380, 6400, 16), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (2, 256, 380, 6400), "ori_format": "NHWC"},
                    {"shape": (56, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),
                     "ori_format": "NHWC"},
                    None,
                    {"shape": (56, 16, 7, 7, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (56, 256, 7, 7),
                     "ori_format": "NHWC"},
                    0.25,
                    7,
                    7,
                    2,
                    2
                    ],
         "case_name": "roi_align_09",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
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
                     2
                     ],
          "case_name": "roi_align_10",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}


case11 = {"params": [{"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (256, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),
                      "ori_format": "NHWC"},
                     None,
                     {"shape": (256, 16, 16, 16, 16), "dtype": "float32", "format": "NHWC",
                      "ori_shape": (256, 16, 16, 256), "ori_format": "NHWC"},
                     0.25,
                     16,
                     16,
                     2,
                     2
                     ],
          "case_name": "roi_align_10",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

case12 = {"params": [{"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (256, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),
                      "ori_format": "NHWC"},
                     None,
                     {"shape": (256, 16, 16, 1, 16), "dtype": "float32", "format": "NHWC",
                      "ori_shape": (256, 16, 1, 256), "ori_format": "NHWC"},
                     0.25,
                     16,
                     1,
                     2,
                     2
                     ],
          "case_name": "roi_align_10",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

case13 = {"params": [{"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (256, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),
                      "ori_format": "NHWC"},
                     None,
                     {"shape": (256, 16, 16, 1, 16), "dtype": "float32", "format": "NHWC",
                      "ori_shape": (256, 16, 1, 256), "ori_format": "NHWC"},
                     0.25,
                     16,
                     1,
                     2,
                     3
                     ],
          "case_name": "roi_align_13",
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
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)


def roi_align_v200_001(test_arg):
    set_current_compile_soc_info("Ascend710")
    roi_align({"shape": (3, 8, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (3, 128, 5, 5), "ori_format": "NHWC"},
              {"shape": (3, 5), "dtype": "float16", "format": "NHWC",
               "ori_shape": (3, 5), "ori_format": "NHWC"},
              {"shape": (3,), "dtype": "float16", "format": "NHWC",
               "ori_shape": (3,), "ori_format": "NHWC"},
              {"shape": (1, 8, 10, 10, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (1, 128, 10, 10), "ori_format": "NHWC"},
              0.25, 5, 5, 2, 0)

    set_current_compile_soc_info(test_arg)


def roi_align_v200_002(test_arg):
    set_current_compile_soc_info("Ascend610")
    roi_align({"shape": (1, 1, 58, 90, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (1, 16, 58, 90), "ori_format": "NHWC"},
              {"shape": (96, 4), "dtype": "float16", "format": "NHWC",
               "ori_shape": (96, 4), "ori_format": "NHWC"},
              None,
              {"shape": (96, 1, 7, 7, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (96, 16, 7, 7), "ori_format": "NHWC"},
              0.25, 7, 7, 2, 1)

    set_current_compile_soc_info(test_arg)


def roi_align_v200_003(test_arg):
    cce_conf.cce_conf.te_set_version("Ascend710")
    roi_align({"shape": (3, 16, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (3, 256, 5, 5), "ori_format": "NHWC"},
              {"shape": (3, 5), "dtype": "float16", "format": "NHWC",
               "ori_shape": (3, 5), "ori_format": "NHWC"},
              {"shape": (3,), "dtype": "float16", "format": "NHWC",
               "ori_shape": (3,), "ori_format": "NHWC"},
              {"shape": (1, 16, 10, 10, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (1, 256, 10, 10), "ori_format": "NHWC"},
              0.25, 5, 5, 2, 0)
    cce_conf.cce_conf.te_set_version(test_arg)


def roi_align_v200_004(test_arg):
    set_current_compile_soc_info("Hi3796CV300CS")
    roi_align({"shape": (3, 32, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (3, 512, 5, 5), "ori_format": "NHWC"},
              {"shape": (3, 5), "dtype": "float16", "format": "NHWC",
               "ori_shape": (3, 5), "ori_format": "NHWC"},
              {"shape": (3,), "dtype": "float16", "format": "NHWC",
               "ori_shape": (3,), "ori_format": "NHWC"},
              {"shape": (1, 32, 10, 10, 16), "dtype": "float16", "format": "NC1HWC0",
               "ori_shape": (1, 512, 10, 10), "ori_format": "NHWC"},
              0.25, 5, 5, 2, 0)

    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=roi_align_v200_001)
ut_case.add_cust_test_func(test_func=roi_align_v200_002)
ut_case.add_cust_test_func(test_func=roi_align_v200_003)
ut_case.add_cust_test_func(test_func=roi_align_v200_004)
