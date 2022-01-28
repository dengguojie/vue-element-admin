#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatD", "impl.dynamic.concat_d", "op_select_format")

case1 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                     {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                    1],
         "case_name": "dyanmic_concat_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910"], case1)
