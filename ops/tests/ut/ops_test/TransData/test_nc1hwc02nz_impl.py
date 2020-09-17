#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_Z"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NC1HWC0", "format": "NC1HWC0"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       "NC1HWC0", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


# c1*h*w == 1
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10000,1,1,1,16), (3, 2, 16,16),
                                     "float16", "nc1hwc0_1", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1000,1,1,1,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_2", "success"))

# n < 3040
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000,3,100,1,16), (3, 2, 16,16),
                                     "float16", "nc1hwc0_3", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1000,3,100,1,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_4", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((100,3,100,1,16), (3, 2, 16,16),
#                                      "float16", "nc1hwc0_5", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((10,6,100,1,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_6", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((10,10,32,1,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_7", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((10,1,32,1,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_8", "success"))

# n >= 3040
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((5001,3,20,1,16), (3, 2, 16,16),
                                     "float16", "nc1hwc0_9", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((5001,3,20,1,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_10", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((3041,3,20,80,32), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_11", "success"))

# exception
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000,1,1,1,16), (3, 2, 16,16),
                                     "float32", "nc1hwc0_12", RuntimeError))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1000,1,1,1,15), (3, 2, 16,16),
#                                      "float16", "nc1hwc0_13", RuntimeError))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1000,1,1,1,16), (3, 2, 16,16),
#                                      "int8", "nc1hwc0_14", RuntimeError))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1000,1,1,16), (3, 2, 16,16),
#                                      "float16", "nc1hwc0_15", RuntimeError))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
