#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, dst_dtype, case_name_val, expect,
                        dst_format="NDC1HWC0", src_format="NDHWC"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NDHWC", "format": "NDHWC"},
                       {"shape": dst, "dtype": dst_dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


# normal
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1,1,1,1,16), (1, 1, 1, 1, 1, 16),
                                     "float16", "float16", "ndhwc_1",
                                     "success"))

# exception
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1,1,1,1,16), (1, 1, 1, 1, 1, 16),
#                                      "float16", "float32","ndhwc_2",
#                                      RuntimeError))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1,1,1,1,16), (1, 1, 1, 1, 1, 15),
#                                      "float16", "float16","ndhwc_3",
#                                      RuntimeError))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1,1,1,1,16), (1, 1, 2, 1, 1, 16),
#                                      "float16", "float16", "ndhwc_4",
#                                      RuntimeError))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1,1,1,1,16), (1, 1, 1, 1, 1, 16),
#                                      "float16", "float16", "ndhwc_5",
#                                      RuntimeError,
#                                      src_format="NHWC"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((1,1,1,1,16), (1, 1, 1, 1, 1, 16),
#                                      "float16", "float16", "ndhwc_6",
#                                      RuntimeError,
#                                      dst_format="NC1HWC0"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1,1,1,1,16), (1, 1, 1, 2, 1, 16),
                                     "float16", "float16", "ndhwc_7",
                                     RuntimeError))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
