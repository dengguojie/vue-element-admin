#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("GatherV2D", "impl.gather_v2_d", "op_select_format")

case1 = {"params": [{"shape": (256, 256, 256, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 256, 256, 256),"ori_format": "NHWC"},
                    {"shape": (128,), "dtype": "int32", "format": "NHWC", "ori_shape": (128,),"ori_format": "NHWC"},
                    {"shape": (256, 256, 256, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 256, 256, 256),"ori_format": "NHWC"},
                    1],
         "case_name": "gather_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (256, 256, 256, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 256, 256, 256),"ori_format": "NHWC"},
                    {"shape": (128, 16), "dtype": "int32", "format": "NHWC", "ori_shape": (128, 16),"ori_format": "NHWC"},
                    {"shape": (256, 256, 256, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 256, 256, 256),"ori_format": "NHWC"},
                    1],
         "case_name": "gather_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910", "Ascend310"], case1)
ut_case.add_case(["Ascend910", "Ascend310"], case2)
