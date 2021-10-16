#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("LayerNorm", "impl.dynamic.layer_norm", "layer_norm")

case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    2, 2],
         # "case_name": "layer_norm_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    0, 2],
         # "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 304, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 304, 512), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (512,), "dtype": "float16", "format": "NCHW", "ori_shape": (512,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (512,), "dtype": "float16", "format": "NCHW", "ori_shape": (512,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (32, 304, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 304, 512),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (32, 304, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 304, 1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (32, 304, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 304, 1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    0, 2],
         # "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (-1, -1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (-1, -1), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None), (1, None))},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1, -1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (-1, 1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None), (1, None))},
                    {"shape": (-1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None))},
                    {"shape": (-1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, 1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None))},
                    1, 1],
         # "case_name": "layer_norm_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (32, 304, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 304, 512), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (304, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (304, 512),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (304, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (304, 512),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (32, 304, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 304, 512),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (32, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 1, 1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (32, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 1, 1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    1, 1],
         # "case_name": "layer_norm_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)

if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(Ascend910A)
