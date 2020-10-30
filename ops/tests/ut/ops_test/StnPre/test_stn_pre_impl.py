# # -*- coding:utf-8 -*-
import sys
import numpy as np
import math
from op_test_frame.ut import OpUT

ut_case = OpUT("stn_pre")

# [TODO] coding expect function here
g_default_theta = [-0.129, 0.626, 0.344, 0.157]
default_theta = [-0.129, 0.626, 0.344, 0.157]
input_theta = [-0.5318887233734131, -0.4005831480026245]


def calc_expect_func(theta, w_index, h_index, pos_coef, pos_offset, size,
                     default_theta, use_default_theta=[False, False], align_corners=False):
    n = theta.get('shape')[0]
    c1 = w_index.get('shape')[1]
    input_h = size[0]
    input_w = size[1]
    output_h = size[2]
    output_w = size[3]
    res = _gen_theta_offset(n, c1, input_h, input_w, output_h, output_w, default_theta, use_default_theta, theta, False)
    return [res[0], res[1], ]


def _gen_theta_offset(n, c1, h, w, output_h, output_w, default_theta, use_default_theta, input_theta, jump_to_read):
    """
    origin_theta [0.5, 0.5, 0.5, 0.5, -0.4, -0.4]
    output1: [theta, ...] len is 4hw
    output2: [offset, ...] len is 4hw
    [[p // output_w * 1.0 / out_h * 2 - 1], [p % output_w * 1.0 / output_w * 2 - 1], [1]]
    """
    # calc x y
    thetas = []
    for b in range(n):
        default_theta_index = 0
        input_theta_index = 0
        b_thetas = []
        for i in range(6):
            if use_default_theta[i]:
                b_thetas.append(default_theta[default_theta_index])
                default_theta_index += 1
            else:
                b_thetas.append(input_theta[b][input_theta_index])
                input_theta_index += 1
        b_thetas = [b_thetas[:3], b_thetas[3:]]
        thetas.append(np.array(b_thetas))
    print(thetas)
    res_theta = []
    res_offset = []
    res_xx = []
    res_yy = []
    print_count = True
    for batch in range(n):
        # theta = np.array([[0.5, 0.0, -0.5], [0.0, 0.5, -0.5]])
        theta = thetas[batch]
        if jump_to_read:
            for p in range(output_h * output_w):
                pos = np.array([[p // output_w * 1.0 / output_h * 2 - 1], [p % output_w * 1.0 / output_w * 2 - 1], [1]])
                origin_pos = np.dot(theta, pos)
                xx = (origin_pos[0][0] + 1) / 2 * h
                yy = (origin_pos[1][0] + 1) / 2 * w
                res_xx.append(xx)
                res_yy.append(yy)
                tmp_theta = []
                tmp_offset = []
                if print_count:
                    print('o_h', xx, 'o_w', yy, 'floor_h', int(xx), 'ceil_h', math.ceil(yy), 'floor_w', int(yy),
                          'ceil_w', math.ceil(yy))
                    print_count = False
                aa = math.floor(xx)
                bb = math.floor(yy)
                if aa < 0 or aa >= h or bb < 0 or bb >= w:
                    tmp_theta.append(0)
                    tmp_offset.append(0)
                else:
                    tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                    tmp_offset.append(aa * w * 16 + bb * 16)
                aa = math.floor(xx)
                bb = math.ceil(yy)
                if aa < 0 or aa >= h or bb < 0 or bb >= w:
                    tmp_theta.append(0)
                    tmp_offset.append(0)
                else:
                    tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                    tmp_offset.append(aa * w * 16 + bb * 16)
                aa = math.ceil(xx)
                bb = math.floor(yy)
                if aa < 0 or aa >= h or bb < 0 or bb >= w:
                    tmp_theta.append(0)
                    tmp_offset.append(0)
                else:
                    tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                    tmp_offset.append(aa * w * 16 + bb * 16)
                aa = math.ceil(xx)
                bb = math.ceil(yy)
                if aa < 0 or aa >= h or bb < 0 or bb >= w:
                    tmp_theta.append(0)
                    tmp_offset.append(0)
                else:
                    tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                    tmp_offset.append(aa * w * 16 + bb * 16)

                if math.floor(xx) == math.ceil(xx):
                    tmp_theta[2] = 0
                    tmp_theta[3] = 0
                if math.floor(yy) == math.ceil(yy):
                    tmp_theta[1] = 0
                    tmp_theta[3] = 0
                res_theta.extend(tmp_theta)
                res_offset.extend(tmp_offset)
        else:
            for c1_count in range(c1):
                for p in range(output_h * output_w):
                    pos = np.array(
                        [[p // output_w * 1.0 / output_h * 2 - 1], [p % output_w * 1.0 / output_w * 2 - 1], [1]])
                    origin_pos = np.dot(theta, pos)
                    xx = (origin_pos[0][0] + 1) / 2 * h
                    yy = (origin_pos[1][0] + 1) / 2 * w
                    res_xx.append(xx)
                    res_yy.append(yy)
                    tmp_theta = []
                    tmp_offset = []
                    if print_count:
                        print('o_h', xx, 'o_w', yy, 'floor_h', int(xx), 'ceil_h', math.ceil(yy), 'floor_w', int(yy),
                              'ceil_w', math.ceil(yy))
                        print_count = False
                    aa = math.floor(xx)
                    bb = math.floor(yy)
                    if aa < 0 or aa >= h or bb < 0 or bb >= w:
                        tmp_theta.append(0)
                        tmp_offset.append(0)
                    else:
                        tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                        tmp_offset.append(aa * w * 16 + bb * 16)
                    aa = math.floor(xx)
                    bb = math.ceil(yy)
                    if aa < 0 or aa >= h or bb < 0 or bb >= w:
                        tmp_theta.append(0)
                        tmp_offset.append(0)
                    else:
                        tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                        tmp_offset.append(aa * w * 16 + bb * 16)
                    aa = math.ceil(xx)
                    bb = math.floor(yy)
                    if aa < 0 or aa >= h or bb < 0 or bb >= w:
                        tmp_theta.append(0)
                        tmp_offset.append(0)
                    else:
                        tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                        tmp_offset.append(aa * w * 16 + bb * 16)
                    aa = math.ceil(xx)
                    bb = math.ceil(yy)
                    if aa < 0 or aa >= h or bb < 0 or bb >= w:
                        tmp_theta.append(0)
                        tmp_offset.append(0)
                    else:
                        tmp_theta.append((1 - abs(xx - aa)) * (1 - abs(yy - bb)))
                        tmp_offset.append(aa * w * 16 + bb * 16)

                    if math.floor(xx) == math.ceil(xx):
                        tmp_theta[2] = 0
                        tmp_theta[3] = 0
                    if math.floor(yy) == math.ceil(yy):
                        tmp_theta[1] = 0
                        tmp_theta[3] = 0
                    res_theta.extend(tmp_theta)
                    res_offset.extend(tmp_offset)
    if jump_to_read:
        res_theta.extend([0] * (n * (c1 - 1) * output_h * output_w * 4))
        res_offset.extend([0] * (n * (c1 - 1) * output_h * output_w * 4))
    return [res_theta, res_offset]


# [TODO] coding cases here
output_w = 5
output_h = 5
w_index = np.array([(i % output_w) * 1.0 / output_w * 2 - 1 for i in range(output_h * output_w)])
h_index = np.array([(i // output_w) * 1.0 / output_h * 2 - 1 for i in range(output_h * output_w)])

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2,), "shape": (1, 2,),
                "param_type": "input", "value": np.array(input_theta)},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,), "shape": (5, 5,),
                "param_type": "input", "value": w_index},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,),
                "shape": (5, 5,), "param_type": "input", "value": h_index},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 5 * 5 * 4),
                "shape": (1, 1, 5 * 5 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 5 * 5 * 4),
                "shape": (1, 1, 5 * 5 * 4), "param_type": "output"},
               [5, 5, 5, 5],
               [-0.129, 0.626, 0.344, 0.157],
               [True, True, True, True, False, False],
               False
               ],
    "expect": True,
    "case_name": "stn_pre2",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2,), "shape": (3, 2,),
                "param_type": "input", "value": np.array(input_theta)},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,), "shape": (5, 5,),
                "param_type": "input", "value": w_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,),
                "shape": (5, 5,), "param_type": "input", "value": h_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               [5, 5, 5, 5],
               [-0.129, 0.626, 0.344, 0.157],
               [True, True, True, True, False, False],
               False
               ],
    "expect": True,
    "case_name": "stn_pre3",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1,), "shape": (3, 1,),
                "param_type": "input", },
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,), "shape": (5, 5,),
                "param_type": "input", "value": w_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,),
                "shape": (5, 5,), "param_type": "input", "value": h_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               [5, 5, 5, 5],
               [-0.129, 0.626, 0.344, 0.157, 0.232323],
               [True, True, True, True, True, False],
               False
               ],
    "expect": True,
    "case_name": "stn_pre4",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5,), "shape": (3, 5,),
                "param_type": "input", },
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,), "shape": (5, 5,),
                "param_type": "input", "value": w_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,),
                "shape": (5, 5,), "param_type": "input", "value": h_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               [5, 5, 5, 5],
               [-0.129, 0.626, 0.344, 0.157, 0.232323],
               [False, False, False, False, False, True],
               False
               ],
    "expect": True,
    "case_name": "stn_pre6",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [None,
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,), "shape": (5, 5,),
                "param_type": "input", "value": w_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (5, 5,),
                "shape": (5, 5,), "param_type": "input", "value": h_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 5000, 5 * 5 * 4),
                "shape": (3, 5000, 5 * 5 * 4), "param_type": "output"},
               [5, 5, 5, 5],
               [-0.129, 0.626, 0.344, 0.157, 0.232323, 0.12344],
               [True, True, True, True, True, True],
               False
               ],
    "expect": True,
    "case_name": "stn_pre5",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [None,
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2,), "shape": (2, 2,),
                "param_type": "input", "value": w_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2,),
                "shape": (2, 2,), "param_type": "input", "value": h_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2 * 2 * 4),
                "shape": (1, 1, 2 * 2 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2 * 2 * 4),
                "shape": (1, 1, 2 * 2 * 4), "param_type": "output"},
               [2, 2, 2, 2],
               [-0.129, 0.626, 0.344, 0.157, 0.232323, 0.12344],
               [True, True, True, True, True, True],
               False
               ],
    "expect": True,
    "case_name": "stn_pre7",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1,), "shape": (1, 1,),
                "param_type": "input", },
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2,), "shape": (2, 2,),
                "param_type": "input", "value": w_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2,),
                "shape": (2, 2,), "param_type": "input", "value": h_index},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 500000, 2 * 2 * 4),
                "shape": (1, 500000, 2 * 2 * 4), "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 500000, 2 * 2 * 4),
                "shape": (1, 500000, 2 * 2 * 4), "param_type": "output"},
               [2, 2, 2, 2],
               [-0.129, 0.626, 0.344, 0.157, 0.232323, 0.12344],
               [True, True, True, True, True, False],
               False
               ],
    "expect": True,
    "case_name": "stn_pre8",
    "format_expect": [],
})

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Hi3796CV300CS"])
    exit(0)
