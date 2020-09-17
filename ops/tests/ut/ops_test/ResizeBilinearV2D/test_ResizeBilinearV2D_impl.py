
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeBilinearV2D", None, None)

case1 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (2, 2), False, False],
         "case_name": "resize_bilinear_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 2), False, True],
         "case_name": "resize_bilinear_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (25, 2), False, False],
         "case_name": "resize_bilinear_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    (2, 2), False, False],
         "case_name": "resize_bilinear_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (5,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (5,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (5,3,10,10,16), "dtype": "float32", "format": "NHWC", "ori_shape": (5,3,10,10,16),"ori_format": "NHWC"},
                    (10, 10), False, True],
         "case_name": "resize_bilinear_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,260,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)


if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)








