#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format, src_format="NC1HWC0"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": src_format, "format": src_format},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
                                     "float16", "nchw_1", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
                                     "int8", "nchw_2", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
                                     "float32", "nchw_3", "success", "NCHW"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
#                                      "uint8", "nchw_4", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((16, 20, 13, 7, 16), (16, 311, 13, 7),
                                     "float32", "nchw_5", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 16, 48, 72, 16), (2, 256, 48, 72),
                                     "float32", "nchw_6", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 2, 41, 101, 16), (2, 31, 41, 101),
                                     "float32", "nchw_7", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 2, 9, 11, 16), (2, 29, 9, 11),
                                     "float16", "nchw_8", "success", "NCHW"))
ut_case.add_case(["Ascend910", "Ascend310"],
                 gen_trans_data_case((2, 4, 65, 65, 16), (2, 58, 65, 65),
                                     "float16", "nchw_9", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3200, 25, 1, 304, 16), (3200, 400, 1, 304),
                                     "float16", "nchw_10", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3200, 25, 1, 304, 16), (3200, 400, 1, 304),
                                     "bfloat16", "nchw_10", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3, 2, 4, 5, 16), (3, 4, 5, 19),
                                     "float16", "nhwc_1", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3, 3968, 4, 5, 16), (3, 4, 5, 63488),
                                     "float32", "nhwc_2", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3, 16, 31, 5001, 16), (3, 31, 5001, 16*16),
                                     "float32", "nhwc_2", "success", "NHWC"))
ut_case.add_case(["Ascend310"],
                 gen_trans_data_case((10000, 1, 127, 127, 16), (10000, 127, 127, 1),
                                     "float32", "nhwc_127_127", "success", "NHWC"))
#invalid
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,3,4,5,16), (2,30,4,5),
                                     "float32", "err_1", RuntimeError, "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,3,4,5,16), (2,3,5,48),
                                     "float32", "err_2", RuntimeError, "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,3,4,5,16), (2,48,3,5),
                                     "float32", "err_3", RuntimeError, "NCHW"))


#five 2 four int8
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1,1,1,16,16), (1, 1, 1,16),
                                     "int8", "int8_1", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,2,16,16), (3, 2, 16,16),
                                     "int8", "int8_2", "success", "NHWC"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3,1,2,16,32), (3, 2, 2, 16),
                                     "int8", "int8_3", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3,3,2001,16,32), (3, 95, 2001, 16),
                                     "int8", "int8_4", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((31,3,201,16,32), (31, 92, 201, 16),
                                     "int8", "int8_5", "success", "NCHW"))

#five 2 four float 
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((2560, 32, 4, 26, 16), (2560, 512, 4, 26),
                                     "float16", "float16_1", "success", "NCHW"))
if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
    exit(0)
