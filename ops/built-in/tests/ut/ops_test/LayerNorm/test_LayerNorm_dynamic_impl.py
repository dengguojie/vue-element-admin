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
         #  "case_name": "layer_norm_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# ut_case.add_case(["Ascend910A"], case1)

if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(Ascend910A)
