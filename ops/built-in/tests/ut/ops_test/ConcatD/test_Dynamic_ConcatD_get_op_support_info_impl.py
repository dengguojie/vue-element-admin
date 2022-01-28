#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatD", "impl.dynamic.concat_d", "get_op_support_info")

case1 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                     {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                    1],
         "case_name": "dynamic_concat_d_op_support_info_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [[{"shape": (128, 8, 128, 128, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                     {"shape": (128, 8, 128, 128, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"}],
                    {"shape": (128, 8, 128, 128, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                    -1],
         "case_name": "dynamic_concat_d_op_op_support_info_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
    exit(0)
