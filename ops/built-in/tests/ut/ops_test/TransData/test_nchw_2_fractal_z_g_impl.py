#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
import math
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, groups, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "NCHW", "format": "NCHW"},
                       "NCHW",
                       "FRACTAL_Z_G",
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
    filter_nchw = np.arange(1, v+1, 1, dtype="int").reshape(shape_in)
    #print(filter_nchw)
    CUBE_K = 16
    CUBE_N = 16
    Cout, Cin, kh, kw = shape_in
    Cin_ori = Cin
    Cout_ori = Cout // groups
    print("Cin_ori=", Cin_ori, "Cout_ori=", Cout_ori)
    A = lcm(Cin_ori, CUBE_K) // Cin_ori
    B = lcm(Cout_ori, CUBE_N) // Cout_ori
    C = lcm(A, B)
    E = min(C, groups)
    Cin_opt = ceil(E * Cin_ori, CUBE_K) * CUBE_K
    Cout_opt = ceil(E * Cout_ori, CUBE_N) * CUBE_N
    G = ceil(groups, E)

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
                filter_gc1hwnc0[g//E, dst_ci//CUBE_K, :, :, dst_co, dst_ci % CUBE_K] = filter_nchw[src_co, ci, :, :]
    dst_4d = filter_gc1hwnc0.reshape(G * (Cin_opt // CUBE_K) * kh * kw, Cout_opt // CUBE_N, CUBE_N, CUBE_K)
    return dst_4d



def gen_trans_data_precision_case(src, dst, dtype, groups, case_name_val, expect):
    v = 1
    for i in src:
        v *= i
    input_value = np.arange(1,v+1, dtype="int16")
    #print(input_value)
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "NCHW", "format": "NCHW", "param_type": "input", "value": input_value},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "NCHW", "format": "NCHW", "param_type": "output"},
                       "NCHW",
                       "FRACTAL_ZN",
                       groups],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.00, 0.00)}


ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((6, 4, 3, 3), (9, 1, 16, 16),
                                                         "int16", 2, "nchw_2_fractal_z_g_precision_001",
                                                         "success"))
ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((9, 4, 3, 3), (9, 1, 16, 16),
                                                         "int16", 3, "nchw_2_fractal_z_g_precision_002",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((16, 3, 3, 3), (9, 1, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_003",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((16, 5, 3, 3), (18, 1, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_004",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((32, 5, 3, 3), (18, 2, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_005",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((32, 5, 3, 3), (27, 2, 16, 16),
                                                         "int16", 8, "nchw_2_fractal_z_g_precision_006",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((8, 32, 3, 3), (36, 1, 16, 16),
                                                         "int16", 2, "nchw_2_fractal_z_g_precision_007",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((320, 3, 2, 2), (24, 10, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_008",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((320, 4, 2, 2), (32, 5, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_009",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((640, 3, 2, 2), (48, 10, 16, 16),
                                                         "int16", 64, "nchw_2_fractal_z_g_precision_010",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((80, 6, 5, 5), (50, 5, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_011",
                                                         "success"))
ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((800, 6, 5, 5), (50, 50, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_012",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((800, 7, 2, 2), (84, 20, 16, 16),
                                                         "int16", 40, "nchw_2_fractal_z_g_precision_013",
                                                         "success"))
ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((80, 15, 3, 3), (36, 5, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_014",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((80, 70, 2, 2), (72, 5, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_015",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((160, 70, 2, 2), (72, 10, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_015",
                                                         "success"))


ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((128, 2, 3, 3), (36, 2, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_101",
                                                         "success"))

# ut_case.add_precision_case(["Ascend910A", "Ascend310"],
#                            gen_trans_data_precision_case((128, 4, 3, 3), (72, 1, 16, 16),
#                                                          "int16", 32, "nchw_2_fractal_z_g_precision_net_102",
#                                                          "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((128, 8, 3, 3), (144, 1, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_103",
                                                         "success"))

# ut_case.add_precision_case(["Ascend910A", "Ascend310"],
#                            gen_trans_data_precision_case((256, 8, 3, 3), (144, 1, 16, 16),
#                                                          "int16", 32, "nchw_2_fractal_z_g_precision_net_104",
#                                                          "success"))
ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((256, 16, 3, 3), (288, 1, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_105",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((512, 32, 3, 3), (576, 1, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_106",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((1024, 32, 3, 3), (576, 2, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_107",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((1024, 64, 3, 3), (1152, 2, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_108",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((512, 16, 3, 3), (288, 1, 16, 16),
                                                         "int16", 32, "nchw_2_fractal_z_g_precision_net_109",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((448, 56, 3, 1), (42, 14, 16, 16),
                                                         "int16", 4, "nchw_2_fractal_z_g_precision_net_110",
                                                         "success"))




if __name__ == '__main__':
    simulator_lib_path ="/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
