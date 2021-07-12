#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
import math
import time
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

    print("calc_expect_func Cin_ori=", Cin_ori, "Cout_ori=", Cout_ori)
    print("calc_expect_func A=", A, "B=", B, "C=", C, "E=", E)
    print("calc_expect_func Cin_opt", Cin_opt)
    print("calc_expect_func Cout_opt", Cout_opt)
    print("calc_expect_func G:", G)

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
    print(filter_gc1hwnc0.shape)
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


ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((2, 2, 3, 320), (24, 10, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_001",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 37, 256), (666, 8, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_002",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 2, 128), (36, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_101",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 4, 128), (72, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_102",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 8, 128), (144, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_103",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 8, 256), (144, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_104",
                                                         "success"))
ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 16, 256), (288, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_105",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 16, 512), (288, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_106",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 32, 512), (576, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_107",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 32, 1024), (576, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_108",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 64, 1024), (1152, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_109",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 8, 112), (63, 1, 16, 16),
                                                         "int16", 14, "hwcn_2_fractal_z_g_precision_net_201",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((2, 2, 8, 256), (64, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_202",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 8, 256), (144, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_203",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((2, 2, 16, 512), (128, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_204",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 16, 512), (288, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_205",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((2, 2, 32, 1024), (256, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_206",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 32, 1024), (576, 2, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_207",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 1, 32), (18, 1, 16, 16),
                                                         "int16", 32, "hwcn_2_fractal_z_g_precision_net_301",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 1, 64), (36, 1, 16, 16),
                                                         "int16", 64, "hwcn_2_fractal_z_g_precision_net_302",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 1, 128), (72, 1, 16, 16),
                                                         "int16", 128, "hwcn_2_fractal_z_g_precision_net_303",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 1, 256), (144, 1, 16, 16),
                                                         "int16", 256, "hwcn_2_fractal_z_g_precision_net_304",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 1, 512), (288, 1, 16, 16),
                                                         "int16", 512, "hwcn_2_fractal_z_g_precision_net_305",
                                                         "success"))

ut_case.add_precision_case(["Ascend910A", "Ascend310", "Ascend310"],
                           gen_trans_data_precision_case((3, 3, 1, 1024), (576, 1, 16, 16),
                                                         "int16", 1024, "hwcn_2_fractal_z_g_precision_net_306",
                                                         "success"))





if __name__ == '__main__':
    simulator_lib_path ="/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
