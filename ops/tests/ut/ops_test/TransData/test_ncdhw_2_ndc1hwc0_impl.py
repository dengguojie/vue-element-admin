#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data",
               "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="NDC1HWC0"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": "NCHW", "format": "NCHW"},
                       "NCDHW", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


# network shape
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,32,21,504,504), (3, 2, 16,16),
                                     "float16", "ncdhw_2_ndc1hwc0_001", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((2,64,11,252,252), (3, 2, 16,16),
#                                      "float16", "ncdhw_2_ndc1hwc0_002", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((35,35,5,3,5), (3, 2, 16,16),
#                                      "float16", "ncdhw_2_ndc1hwc0_003", "success"))
# TODO fix me run failed
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((35,10,5,252,252), (3, 2, 16,16),
#                                      "float32", "ncdhw_2_ndc1hwc0_004", "success"))

# exception
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((128,64,3,3,3), (3, 2, 16,16),
#                                      "float16", "ncdhw_2_ndc1hwc0_005", RuntimeError, "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128,64,3,3,3), (3, 2, 16,16),
                                     "int8", "ncdhw_2_ndc1hwc0_006", RuntimeError))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
