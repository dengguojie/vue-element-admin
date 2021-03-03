# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
cdist_grad
"""
import numpy as np
import torch
from op_test_frame.ut import OpUT
from op_test_frame.utils import op_param_util
from op_test_frame.common import op_status
from itertools import product

ut_case = OpUT("cdist_grad")


#pylint: disable=unused-argument
def torch_cdist_grad_except(input_grad, input_x1, input_x2, input_cdist, y, p):
    x1_value = torch.tensor(input_x1["value"], dtype=torch.float32,
                            requires_grad=True)
    x2_value = torch.tensor(input_x2["value"], dtype=torch.float32,
                            requires_grad=True)
    grad_value = torch.tensor(input_grad["value"], dtype=torch.float32)

    cdist_value = torch.tensor(input_cdist["value"], dtype=torch.float32)

    res = cdist_backward(x1_value, x2_value, p, grad_value, cdist_value)
    if input_x1["dtype"] == "flaot16":
        res = res.half()
    return [res.numpy(), ]


def cdist_backward(x1, x2, p, grad, cdist):
    diff = x1 - x2
    diff_abs = torch.abs(diff)
    nz_cdist = torch.where(cdist == 0, torch.ones_like(cdist), cdist)
    sign = torch.where(diff > 0, torch.ones_like(diff),
                       torch.full_like(diff, -1))
    sign = torch.where(diff == 0, torch.zeros_like(diff), sign)

    if p == 0.0:
        res = torch.zeros_like(diff)
    elif p == 1.0:
        res = grad * sign
    elif p < 2.0:
        res = sign * torch.pow(diff_abs, p - 1.0) * grad / torch.pow(nz_cdist,
                                                                     p - 1.0)
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
    elif p == 2.0:
        res = grad * diff / nz_cdist
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
    elif p == float("inf"):
        mask = torch.where(cdist - diff_abs > 0, torch.zeros_like(diff),
                           torch.ones_like(diff))
        res = grad * sign * mask
    else:
        res = diff * torch.pow(diff_abs, p - 2) * grad / torch.pow(nz_cdist,
                                                                   p - 1.0)
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
    res = torch.sum(res, -2)
    return res.detach()


def get_case(dtype, x1_shape, x2_shape, p=2.0,
             expect=op_status.SUCCESS, case_name=None):
    x1_shape_bc = list(x1_shape)
    x1_shape_bc.insert(-1, x2_shape[-2])
    x1_shape_bc = tuple(x1_shape_bc)
    x2_shape_bc = list(x2_shape)
    x2_shape_bc.insert(-2, x1_shape[-2])
    x2_shape_bc = tuple(x2_shape_bc)

    x1_info = (dtype, x1_shape_bc, "ND")
    x2_info = (dtype, x2_shape_bc, "ND")
    grad_info = (dtype, x1_shape_bc, "ND")

    grad_param = op_param_util.build_op_param(grad_info)
    x1_param = op_param_util.build_op_param(x1_info)
    x2_param = op_param_util.build_op_param(x2_info)
    cdist_param = op_param_util.build_op_param(grad_info)
    y_param = op_param_util.build_op_param(x1_info)

    param_list = [grad_param, x1_param, x2_param, cdist_param, y_param, p]
    if expect == op_status.SUCCESS:
        case = {"params": param_list, "case_name": case_name}
    else:
        case = {"params": param_list, "case_name": case_name, "expect": expect}
    return case


def pr_case(dtype, x1_shape, x2_shape, p=2.0,
            input_range=(0.1, 1.0), case_name=None):
    input_shape = list(x1_shape)
    input_shape.insert(-1, x2_shape[-2])
    input_shape = tuple(input_shape)
    input_info = (dtype, input_shape, "ND")
    output_info = (dtype, x1_shape, "ND")

    grad_param = op_param_util.build_op_param(input_info)
    x1_param = op_param_util.build_op_param(input_info)
    x2_param = op_param_util.build_op_param(input_info)
    cdist_param = op_param_util.build_op_param(input_info)
    y_param = op_param_util.build_op_param(output_info)

    np.random.seed(10086)
    x1_value = np.random.uniform(input_range[0], input_range[1],
                                 x1_shape).astype(dtype)
    x2_value = np.random.uniform(input_range[0], input_range[1],
                                 x2_shape).astype(dtype)
    cdist_output = torch.cdist(torch.from_numpy(x1_value).float(),
                               torch.from_numpy(x2_value).float(), p)
    cdist_value = cdist_output.numpy().astype(dtype)

    x1_value = np.expand_dims(x1_value, -2)
    x2_value = np.expand_dims(x2_value, -3)
    cdist_value = np.expand_dims(cdist_value, -1)

    x1_value = np.broadcast_to(x1_value, input_shape)
    x2_value = np.broadcast_to(x2_value, input_shape)
    cdist_value = np.broadcast_to(cdist_value, input_shape)
    grad_value = np.ones_like(cdist_value)

    for param, value in zip([grad_param, x1_param, x2_param, cdist_param],
                            [grad_value, x1_value, x2_value, cdist_value]):
        param["param_type"] = "input"
        param["value"] = value
    y_param["param_type"] = "output"

    param_list = [grad_param, x1_param, x2_param, cdist_param, y_param, p]
    case = {"params": param_list, "case_name": case_name,
            "calc_expect_func": torch_cdist_grad_except}
    return case


ut_case.add_case("all", case=get_case("float32", (4, 4, 4, 4),
                                      (4, 4, 4, 4), 2.0, expect=RuntimeError))
ut_case.add_case("all", case=get_case("float32", (4, 4, 5),
                                      (4, 4), 2.0, expect=RuntimeError))
ut_case.add_case("all", case=get_case("float32", (4, 4, 5),
                                      (4, 4, 6), 2.0, expect=RuntimeError))


ut_case.add_precision_case("Ascend910A", case=pr_case("float16",
                                                      [5, 8],
                                                      [4, 8],
                                                      0.0,
                                                      input_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                                      [5, 8],
                                                      [4, 8],
                                                      2.0,
                                                      input_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                               [2, 4, 20],
                                               [2, 5, 20],
                                               1.0,
                                               input_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                                      [2, 4, 20],
                                                      [2, 5, 20],
                                                      1.5,
                                                      input_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                                      [2, 4, 20],
                                                      [2, 5, 20],
                                                      2.0,
                                                      input_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                                      [2, 4, 20],
                                                      [2, 5, 20],
                                                      0.0,
                                                      input_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                                      [2, 4, 20],
                                                      [2, 5, 20],
                                                      3.0,
                                                      input_range=[0.01, 1]))

