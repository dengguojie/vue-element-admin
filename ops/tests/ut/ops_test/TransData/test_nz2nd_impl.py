#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="ND", src_format="FRACTAL_NZ"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": src_format, "format": src_format},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("Ascend910", gen_trans_data_case((1, 64, 1, 16, 16), (1, 16, 1024), "float32", "nz2nd_1", "success"))
ut_case.add_case("Ascend910", gen_trans_data_case((25, 32, 1, 16, 16), (25, 16, 512), "float16", "nz2nd_2", "success"))
ut_case.add_case("Ascend910", gen_trans_data_case((1, 32, 1, 16, 16), (1, 16, 512), "int32", "nz2nd_3", "success"))

# ut_case.add_case("Ascend910", gen_trans_data_case((1, 64, 1, 16, 16), (1, 1024, 1, 16), "float32", "nz2nchw_1", "success", "NCHW"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
