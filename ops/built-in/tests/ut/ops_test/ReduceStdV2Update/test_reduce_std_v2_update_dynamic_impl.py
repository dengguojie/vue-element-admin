# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceStdV2Update", "impl.dynamic.reduce_std_v2_update", "reduce_std_v2_update")

def gen_case(dtype_val, inp_shape_val, inp_ori_shape_val, inp_range_val,outp_shape_val, outp_ori_shape_val, outp_range_val,
             dim_val, if_std_val, unbiased_val, keepdim_val):
    return({
        "params": [{"dtype": dtype_val,
                    "format": "ND", "ori_format": "ND",
                    "shape": inp_shape_val, "ori_shape": inp_ori_shape_val,
                    "range": inp_range_val},
                   {"dtype": dtype_val,
                    "format": "ND", "ori_format": "ND",
                    "shape": inp_shape_val, "ori_shape": inp_ori_shape_val,
                    "range": inp_range_val},
                   {"dtype": dtype_val,
                    "format": "ND", "ori_format": "ND",
                    "shape": outp_shape_val, "ori_shape": outp_ori_shape_val,
                   "range": outp_range_val},
                   dim_val, if_std_val, unbiased_val, keepdim_val],
        "case_name": dtype_val
    })


ut_case.add_case("all", gen_case("float16", (-1, -1), (3, 4), [(3, 3), (4, 4)],
                                 (-1, -1), (3, 1), [(3, 3), (4, 4)], [1, ], True, True, True))

ut_case.add_case("all", gen_case("float32", (-1, -1, -1), (3, 4, 5), [(3, 3), (4, 4), (5, 5)],
                                 (-1, -1), (3, 4), [(3, 3), (4, 4)], [2, ], False, False, False))

ut_case.add_case("all", gen_case("float16", (-1, -1, -1), (3, 4, 5), [(3, 3), (4, 4), (5, 5)],
                                 (-1, -1), (3, 5), [(3, 3), (5, 5)], [1, ], False, True, False))

ut_case.add_case("all", gen_case("float32", (-1, -1, -1), (3, 4, 5), [(3, 3), (4, 4), (5, 5)],
                                 (-1, -1, -1), (3, 1, 5), [(3, 3), (1, 1), (5, 5)], [1, ], True, True, True))

if __name__ == "__main__":
    ut_case.run("Ascend310")
    #ut_case.run("Ascend910A")
