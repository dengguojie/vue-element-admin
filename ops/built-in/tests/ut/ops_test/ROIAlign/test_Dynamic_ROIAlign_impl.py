# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import te

ut_case = OpUT("ROIAlign", "impl.dynamic.roi_align", "roi_align")

case1 = {"params": [{"shape": (2, 16, 24, 40, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 24, 40),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (32, 5), "dtype": "float32", "format": "ND", "ori_shape": (32, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    None,
                    {"shape": (32, 16, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (32, 256, 14, 14),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    0.03125, 14, 14, 2
                    ],
         "case_name": "RoiAlign_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2, 16, 336, 336, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 336, 336),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (1280, 5), "dtype": "float32", "format": "ND", "ori_shape": (1280, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (1280,), "dtype": "int32", "format": "ND", "ori_shape": (1280,),
                     "ori_format": "ND", "range":((1280,1280),)},
                    {"shape": (1280, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (1280, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    0.25, 7, 7, 2, 1
                    ],
         "case_name": "dynamic_RoiAlign_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2, 16, 336, 336, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 336, 336),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (256, 5), "dtype": "float32", "format": "ND", "ori_shape": (1280, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (256,), "dtype": "int32", "format": "ND", "ori_shape": (1280,),
                     "ori_format": "ND", "range":((256,256),)},
                    {"shape": (256, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (1280, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    0.25, 7, 7, 2, 1
                    ],
         "case_name": "dynamic_RoiAlign_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2, 16, 5, 5, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 5, 5),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (32, 5), "dtype": "float32", "format": "ND", "ori_shape": (1280, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (32,), "dtype": "int32", "format": "ND", "ori_shape": (1280,),
                     "ori_format": "ND", "range":((256,256),)},
                    {"shape": (32, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (1280, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    0.25, 7, 7, 2, 1
                    ],
         "case_name": "dynamic_RoiAlign_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (2, 16, 2, 336, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 2, 336),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))}, #x
                    {"shape": (32, 5), "dtype": "float32", "format": "ND", "ori_shape": (32, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (32,), "dtype": "int32", "format": "ND", "ori_shape": (32,),
                     "ori_format": "ND", "range":((256,256),)},
                    {"shape": (32, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (32, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    0.25, 7, 7, 2, 1
                    ],
         "case_name": "dynamic_RoiAlign_5",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)

if __name__ == "__main__":
    ut_case.run(["Ascend910A"])