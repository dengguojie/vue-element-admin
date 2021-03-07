# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT
import torch
import numpy as np


shape_grads = [1, 2, 3, 4]
original_size = [1, 2, 3, 4]
align_corners = False
scales = (0, 0)
x = torch.ones(original_size)
self = torch.tensor(x, requires_grad=True)
res = torch._C._nn.upsample_bicubic2d(self, [shape_grads[2], shape_grads[3]],
                                      align_corners, scales[0], scales[1])

shape_grads_lin = [1, 1, 2]
original_size_lin = [1, 1, 1]
x_lin = torch.ones(original_size_lin, requires_grad=True)
res_lin = torch._C._nn.upsample_linear1d(x_lin, shape_grads_lin[2], align_corners=True)
res_lin = res_lin.unsqueeze(2)


ut_case = OpUT("ResizeGradD")


# pylint: disable=unused-argument,invalid-name,consider-using-enumerate
def calc_expect_func(input_x, output_z, ori_shape, scale, align, coor, cub, exc, extra, mode, near_mode):
    if mode == "cubic":
        global res
        grads = torch.ones_like(res)
        res.backward(grads, retain_graph=True)
        global self
        res_back = self.grad
        return res_back.numpy()
    elif mode == "linear":
        global res_lin
        grads = torch.ones_like(res_lin).float()
        res_lin.backward(grads)
        global x_lin
        res_value = x_lin.grad
        return res_value.numpy()
    else:
        raise RuntimeError("Upsample Not supported.")

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": res.size(), "shape": res.size(),
                "param_type": "input", "value": res.detach().numpy()},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": res.size(), "shape": res.size(),
                "param_type": "output"}, [1, 2, 3, 4], [0], [0.0, 0.0],
               "half_pixel", -0.75, 0, 0.0, "cubic", "round_prefer_floor"],
    "expect": SystemExit,
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": res_lin.size(), "shape": res_lin.size(),
                "param_type": "input", "value": res_lin.detach().numpy()},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": [1, 1, 1], "shape": [1, 1, 1],
                "param_type": "output"}, [1, 1, 1], [0], [2.0],
               "half_pixel", -0.75, 0, 0.0, "linear", "round_prefer_floor"],
    "expect": SystemExit,
    "calc_expect_func": calc_expect_func
})


