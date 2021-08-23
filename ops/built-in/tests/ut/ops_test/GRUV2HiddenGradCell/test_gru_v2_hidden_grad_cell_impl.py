# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import numpy as np
import os
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from gru_numpy_tik import gruv2_hidden_grad_data


ut_case = OpUT("gru_v2_hidden_grad_cell", "impl.gru_v2_hidden_grad_cell", "gru_v2_hidden_grad_cell")


def calc_expect_func(dh_pre_t, h, dy, dh, update, reset, new, hidden_new, dh_prev, dgate_h, dnt_x, t_state=0):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" * 3)
    print("dh_prev", dh_prev["valuey"].shape, dh_prev["shape"])
    print("dgate_h", dgate_h["valuey"].shape, dgate_h["shape"])
    print("dnt_x", dnt_x["valuey"].shape, dnt_x["shape"])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" * 3)
    return dh_prev["valuey"], dgate_h["valuey"], dnt_x["valuey"]


def gen_gru_v2_hidden_grad_case(t, n, output_size, dtype="float32"):
    case_name = "case_%s_%s_%s" % (t, n, output_size)
    return {"params": [
        {"shape": (output_size * 3, output_size, 16, 16), "dtype": "float16"},
        {"shape": (output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, 3 * output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype}],
        "case_name": case_name
    }


def gen_gru_v2_hidden_grad_precision_case(shape_val, dtype, t_state=0):
    # shape_val= (5,32,64,64)
    t = shape_val[0]
    batch = shape_val[1] // 16
    input_dim = shape_val[2] // 16
    output_dim = shape_val[3] // 16

    gru_dict = gruv2_hidden_grad_data(t, batch * 16, input_dim * 16, output_dim * 16, dtype, (0, 1), t_state,
                                      gate_order="zrh", kenel_name="gru_grad")

    #weight_hidden = {"shape": (output_dim * 3, output_dim, 16, 16), "dtype": "float16", "param_type": "input",
    #                 "value": gru_dict["weight_hidden"]}
    #init_h = {"shape": (output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input",
    #          "value": gru_dict["init_h"]}
    # h = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["h"]}
    if t_state == t - 1:
        h = {"shape": (output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["init_h"]}
    else:
        h = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["h"]}
    dy = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["dy"]}
    dh = {"shape": (output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["dh"]}
    update = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input",
              "value": gru_dict["update"]}
    reset = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input",
             "value": gru_dict["reset"]}
    new = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["new"]}
    hidden_new = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input",
                  "value": gru_dict["hidden_new"]}
    dh_pre_t = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "input", "value": gru_dict["dh_pre_t"]}

    # output
    dh_prev = {"shape": (output_dim, batch, 16, 16), "dtype": dtype, "param_type": "output",
               "valuey": gru_dict["dh_prev"]}
    dgate_h = {"shape": (t, 3 * output_dim, batch, 16, 16), "dtype": dtype, "param_type": "output",
               "valuey": gru_dict["dgate_h"]}
    dnt_x = {"shape": (t, output_dim, batch, 16, 16), "dtype": dtype, "param_type": "output",
             "valuey": gru_dict["dnt_x"]}
    return {
        "params": [dh_pre_t, h, dy, dh, update, reset, new, hidden_new, dh_prev, dgate_h, dnt_x, t_state],
        "case_name": "gru_v2_hidden_grad_cell",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.005, 0.001)
    }


#ut_case.add_precision_case(['Ascend910'], gen_gru_v2_hidden_grad_precision_case((5, 32, 64, 128), "float32"))
ut_case.add_precision_case(['Ascend910A'], gen_gru_v2_hidden_grad_precision_case((1, 32, 64, 128), "float32"))
ut_case.add_case(['Ascend910'], gen_gru_v2_hidden_grad_case(5, 1, 4))

'''
ut_case.add_case(['Ascend910'], gen_gru_v2_hidden_grad_case(5, 8, 8))
ut_case.add_case(['Ascend910'], gen_gru_v2_hidden_grad_case(1, 8, 8))
ut_case.add_case(['Ascend910'], gen_gru_v2_hidden_grad_case(5, 32, 4))
ut_case.add_case(['Ascend910'], gen_gru_v2_hidden_grad_case(5, 36, 10))
'''


if __name__ == '__main__':
    ut_case.run('Ascend910A')
    exit(0)
