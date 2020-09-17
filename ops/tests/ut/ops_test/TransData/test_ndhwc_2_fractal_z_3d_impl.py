#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_Z_3D"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": "NCHW", "format": "NCHW"},
                       "NDHWC", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


# network shape
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((16,3,3,3,5), (3, 2, 16,16),
                                     "float16", "ndhwc_2_fractal_z_3d_001", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((16,3,3,3,5), (3, 2, 16,16),
                                     "float32", "ndhwc_2_fractal_z_3d_002", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((128,3,3,3,64), (3, 2, 16,16),
#                                      "float16", "ndhwc_2_fractal_z_3d_003", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((128,3,3,3,64), (3, 2, 16,16),
#                                      "float32", "ndhwc_2_fractal_z_3d_004", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((128,3,3,3,2), (3, 2, 16,16),
#                                      "float16", "ndhwc_2_fractal_z_3d_005", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((116,3,3,1000,61), (3, 2, 16,16),
#                                      "float16", "ndhwc_2_fractal_z_3d_006", "success"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((116,3,3,1000,61), (3, 2, 16,16),
#                                      "float32", "ndhwc_2_fractal_z_3d_007", "success"))

# exception
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((128,64,3,3,3), (3, 2, 16,16),
#                                      "float16", "ndhwc_2_fractal_z_3d_008", RuntimeError, "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128,64,3,3,3), (3, 2, 16,16),
                                     "int8", "ndhwc_2_fractal_z_3d_009", RuntimeError))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
