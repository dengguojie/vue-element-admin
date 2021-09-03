#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dtype, case_name_val, expect,
                        src_format, dst_format="NC1HWC0", dst=(8,1,1,1,16), groups=1):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": src_format, "format": src_format},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format, groups],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

#NHWC
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,3,10,32),"float16", "nhwc_1", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,33,21,27),"float16", "nhwc_2", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,33,21,7),"float16", "nhwc_3", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,3,10,32),"float32", "nhwc_4", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,33,21,27),"float32", "nhwc_5", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,300,751,3),"float32", "nhwc_6", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,30,751,2001),"float16", "nhwc_7", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1,3,1,100000),"float32", "nhwc_8", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1,3,1,63488),"float32", "nhwc_8", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,32,15,3968),"float32", "nhwc_8", "success", "NHWC"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((8,640,640,3),"float16", "nhwc_9", "success", "NHWC"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((8,640,640,3),"int8", "nhwc_10", "success", "NHWC", "NC1HWC0",(8, 1, 1, 1, 32)))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((8,2,3,1),"int8", "nhwc_11", "success", "NHWC", "NC1HWC0",(8, 1, 1, 1, 32)))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((1,1,1,10001),"int8", "nhwc_12", "success", "NHWC", "NC1HWC0",(8, 1, 1, 1, 32)))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((10001,1,1,11),"int8", "nhwc_13", "success", "NHWC", "NC1HWC0",(8, 1, 1, 1, 32)))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((10001,1,1,11),"bfloat16", "nhwc_14", "success", "NHWC", "NC1HWC0",(8, 1, 1, 1, 16)))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((2,1,1,11),"float16", "nhwc_15", "success", "HWCN", "FRACTAL_Z",(1, 1, 16, 16), 11))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((2,1,16,16),"float16", "nhwc_16", "success", "FRACTAL_Z", "HWCN",(2, 1, 1, 11), 11))

# NCHW
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,35,3,10),"float16", "nchw_1", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,27,33,21),"float16", "nchw_2", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,35,3,10),"float32", "nchw_3", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,27,33,21),"float32", "nchw_4", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,27,330,2001),"float32", "nchw_5", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10,27,330,2001),"float16", "nchw_6", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((10,27,330,2001),"int8", "nchw_7", "success", "NCHW", "NC1HWC0", (10, 1, 330, 2001, 32)))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((16,240,7,7),"float16", "nchw_8", "success", "NCHW", "NC1HWC0", (16, 15, 7, 7, 16)))

# error
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,3,10,32),"int32", "err_1", RuntimeError, "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,3,10,32),"float16", "err_2", RuntimeError, "NC1HWC0"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,3,10),"float16", "err_3", RuntimeError, "NHWC"))


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
