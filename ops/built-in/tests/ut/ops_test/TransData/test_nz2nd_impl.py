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

def gen_trans_data_case_1(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_NZ", src_format="ND"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": src_format, "format": src_format},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def test_nd_2_nz_1(test_args):
    from impl.trans_data import trans_data
    from te import platform as cce_conf
    cce_conf.cce_conf.te_set_version("Ascend710", core_type="VectorCore")
    trans_data({"shape": (1, 16, 512), "dtype": "int32", "format": "ND", "ori_shape": (1, 16, 512),"ori_format": "ND"},
               {"shape": (1, 32, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (1, 32, 1, 16, 16),"ori_format": "FRACTAL_NZ"},
               "ND", "FRACTAL_NZ")
    cce_conf.cce_conf.te_set_version(test_args)

def test_nd_2_nz_2(test_args):
    from impl.trans_data import trans_data
    from te import platform as cce_conf
    cce_conf.cce_conf.te_set_version("Hi3796CV300CS")
    trans_data({"shape": (8, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (4, 128),"ori_format": "ND"},
               {"shape": (4, 128), "dtype": "int32", "format": "ND", "ori_shape": (4, 128),"ori_format": "ND"},
               "FRACTAL_NZ", "ND")
    cce_conf.cce_conf.te_set_version(test_args)

#ut_case.add_cust_test_func(test_func=test_nd_2_nz_1)
#ut_case.add_cust_test_func(test_func=test_nd_2_nz_2)
#ut_case.add_case("Ascend310", gen_trans_data_case((1, 64, 1, 16, 16), (1, 16, 1024), "float32", "nz2nd_1", "success"))
#ut_case.add_case("Ascend310", gen_trans_data_case((25, 32, 1, 16, 16), (25, 16, 512), "float16", "nz2nd_2", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((32, 32, 16, 16), (512, 512), "float16", "nz2nd_4", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((32, 64, 16, 16), (1024, 512), "float16", "nz2nd_5", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((32, 768, 16, 16), (12288, 512), "float16", "nz2nd_6", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((12, 4, 64, 64, 16, 16), (12, 4, 1024, 1024), "float16", "nz2nd_7", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((48, 8, 64, 16, 16), (48, 1024, 128), "float16", "nz2nd_8", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((1024, 8, 3, 16, 16), (1024, 48, 127), "float16", "nz2nd_9", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((64, 768, 16, 16), (12288, 1024), "float16", "nz2nd_10", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((415, 768, 16, 16), (12288, 6632), "float16", "nz2nd_11", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((1024, 32, 1, 16, 16), (1024, 12, 512), "float16", "nz2nd_12", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((48, 64, 64, 16, 16), (48, 1024, 1024), "float16", "nz2nd_13", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((48, 64, 8, 16, 16), (48, 128, 1024), "float16", "nz2nd_14", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((64, 32, 16, 16), (512, 1024), "float16", "nz2nd_15", "success"))
#
#ut_case.add_case("Ascend910A", gen_trans_data_case((32, 32, 16, 16), (512, 512), "float32", "nz2nd_16", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((32, 64, 16, 16), (1024, 512), "float32", "nz2nd_17", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((32, 768, 16, 16), (12288, 512), "float32", "nz2nd_18", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((12, 4, 64, 64, 16, 16), (12, 4, 1024, 1024), "float32", "nz2nd_19", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((48, 8, 64, 16, 16), (48, 1024, 128), "float32", "nz2nd_20", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((1024, 8, 3, 16, 16), (1024, 48, 127), "float32", "nz2nd_21", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((64, 768, 16, 16), (12288, 1024), "float32", "nz2nd_22", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((415, 768, 16, 16), (12288, 6632), "float32", "nz2nd_23", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((1024, 32, 1, 16, 16), (1024, 12, 512), "float32", "nz2nd_24", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((48, 64, 64, 16, 16), (48, 1024, 1024), "float32", "nz2nd_25", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((48, 64, 8, 16, 16), (48, 128, 1024), "float32", "nz2nd_26", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((64, 32, 16, 16), (512, 1024), "float32", "nz2nd_27", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case((16000, 1, 1, 16, 16), (16000, 1, 10), "float16", "nz2nd_28", "success"))
#ut_case.add_case("Ascend910A", gen_trans_data_case_1((76800, 54), (4, 4800, 16, 16), "float16", "nz2nd_29", "success"))
## ut_case.add_case("Ascend910", gen_trans_data_case((1, 64, 1, 16, 16), (1, 1024, 1, 16), "float32", "nz2nchw_1", "success", "NCHW"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    ut_case.run("Hi3796CV300CS")
    exit(0)
