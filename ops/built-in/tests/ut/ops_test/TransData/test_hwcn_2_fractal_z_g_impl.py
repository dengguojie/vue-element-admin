#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
import math
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, groups, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "HWCN", "format": "HWCN"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "HWCN", "format": "HWCN"},
                       "HWCN",
                       "FRACTAL_Z",
                       groups],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def lcm(m,n):
    return (m*n) // math.gcd(m,n)

def ceil(m,n):
    return (m+n-1)//n

def calc_expect_func(in_tensor, out_tensor, src_format, dst_format, groups):
    shape_in = in_tensor.get("shape")
    in_dtype=in_tensor.get("dtype").lower()
    v=1
    for i in shape_in:
        v *= i
    filter_hwcn = np.arange(1, v+1, 1, dtype="int").reshape(shape_in)
    CUBE_K = 16
    CUBE_N = 16
    kh, kw, Cin, Cout = shape_in
    Cin_ori = Cin
    Cout_ori = Cout // groups
    A = lcm(Cin_ori, CUBE_K) // Cin_ori
    B = lcm(Cout_ori, CUBE_N) // Cout_ori
    C = lcm(A, B)
    E = min(C, groups)
    Cin_opt = ceil(E * Cin_ori, CUBE_K) * CUBE_K
    Cout_opt = ceil(E * Cout_ori, CUBE_N) * CUBE_N
    G = ceil(groups, E)

    print("Cin_ori=", Cin_ori, "Cout_ori=", Cout_ori)
    print("A=", A, "B=", B, "C=", C, "E=", E)
    print("Cin_opt", Cin_opt)
    print("Cout_opt", Cout_opt)
    print("G:", G)

    filter_shape_gc1hwnc0 = (G, Cin_opt// CUBE_K, kh, kw, Cout_opt, CUBE_K)
    filter_gc1hwnc0 =  np.zeros(filter_shape_gc1hwnc0, in_dtype)
    for g in range(groups):
        for ci in range(Cin_ori):
            for co in range(Cout_ori):
                e = g % E
                dst_ci = e * Cin_ori + ci
                dst_co = e * Cout_ori + co
                src_co = g * Cout_ori + co
                filter_gc1hwnc0[g//E, dst_ci//CUBE_K, :, :, dst_co, dst_ci % CUBE_K] = filter_hwcn[:, :, ci, src_co]
    dst_4d = filter_gc1hwnc0.reshape(G * (Cin_opt // CUBE_K) * kh * kw, Cout_opt // CUBE_N, CUBE_N, CUBE_K)
    return dst_4d


def gen_trans_data_precision_case(src, dst, dtype, groups, case_name_val, expect):
    v = 1
    for i in src:
        v *= i
    input_value = np.arange(1,v+1, dtype="int16")
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "HWCN", "format": "HWCN", "param_type": "input", "value": input_value},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "HWCN", "format": "HWCN", "param_type": "output"},
                       "HWCN",
                       "FRACTAL_ZN",
                       groups],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.00, 0.00)}


#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((6, 4, 3, 3), (9, 1, 16, 16),
#                                                         "int16", 2, "hwcn_2_fractal_z_g_precision_001",
#                                                         "success"))
#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((9, 4, 3, 3), (9, 1, 16, 16),
#                                                         "int16", 3, "hwcn_2_fractal_z_g_precision_002",
#                                                         "success"))
#
#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((16, 3, 3, 3), (9, 1, 16, 16),
#                                                         "int16", 4, "hwcn_2_fractal_z_g_precision_003",
#                                                         "success"))
#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((16, 5, 3, 3), (18, 1, 16, 16),
#                                                         "int16", 4, "hwcn_2_fractal_z_g_precision_004",
#                                                         "success"))
#
#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((32, 5, 3, 3), (18, 2, 16, 16),
#                                                         "int16", 4, "hwcn_2_fractal_z_g_precision_005",
#                                                         "success"))
#
#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((32, 5, 3, 3), (27, 2, 16, 16),
#                                                         "int16", 8, "hwcn_2_fractal_z_g_precision_006",
#                                                         "success"))

#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((8, 32, 3, 3), (36, 1, 16, 16),
#                                                         "int16", 2, "hwcn_2_fractal_z_g_precision_007",
#                                                         "success"))

#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((320, 3, 2, 2), (24, 10, 16, 16),
#                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_008",
#                                                         "success"))
#
#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((320, 4, 2, 2), (32, 5, 16, 16),
#                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_009",
#                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 2, 128), (36, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_101",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 4, 128), (72, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_102",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 8, 128), (144, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_103",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 8, 256), (144, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_104",
                                                         "success"))
ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 16, 256), (288, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_105",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 16, 512), (288, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_106",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 32, 512), (576, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_107",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 32, 1024), (576, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_108",
                                                         "success"))

ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 64, 1024), (1152, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_109",
                                                         "success"))




if __name__ == '__main__':
    simulator_lib_path ="/home/shenmin/Ascend/toolkit/tools/simulator"
    #ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
