#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatV2D", "impl.dynamic.concat_v2_d", "get_op_support_info")

case1 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                     {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                    1],
         "case_name": "dynamic_concat_v2_d_op_support_info_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910"], case1)
