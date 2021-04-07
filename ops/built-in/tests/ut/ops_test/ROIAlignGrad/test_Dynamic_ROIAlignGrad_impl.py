# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import te

ut_case = OpUT("ROIAlignGrad", "impl.dynamic.roi_align_grad", "roi_align_grad")

case1 = {"params": [{"shape": (1280, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (1280, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (1280, 5), "dtype": "float32", "format": "ND", "ori_shape": (1280, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (1280,), "dtype": "int32", "format": "ND", "ori_shape": (1280,),
                     "ori_format": "ND", "range":((1280,1280),)},
                    {"shape": (2, 16, 48, 80, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 48, 80),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    [2, 256, 48, 80], 7, 7, 0.0625, 2
                    ],
         "case_name": "RoiAlignGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1280, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (1280, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (1280, 5), "dtype": "float32", "format": "ND", "ori_shape": (1280, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (1280,), "dtype": "int32", "format": "ND", "ori_shape": (1280,),
                     "ori_format": "ND", "range":((1280,1280),)},
                    {"shape": (2, 16, 336, 336, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 336, 336),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    [2, 256, 336, 336], 7, 7, 2.0, 2
                    ],
         "case_name": "RoiAlignGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1, 1, 36, 36, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 36, 36, 16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (200, 5), "dtype": "float16", "format": "ND", "ori_shape": (200, 5),"ori_format": "ND"},
                    {"shape": (7, 5), "dtype": "int32", "format": "ND", "ori_shape": (7, 5),"ori_format": "ND"},
                    {"shape": (1, 1, 36, 36, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 36, 36, 16),"ori_format": "NC1HWC0"},
                    [2,], 7, 7, 2.0, 0
                    ],
         "case_name": "RoiAlignGrad_3",
         "expect": RuntimeError,
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
