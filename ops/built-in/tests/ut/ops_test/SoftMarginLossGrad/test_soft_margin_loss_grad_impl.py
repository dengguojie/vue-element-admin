# Copyright 2021 Huawei Technologies Co., Ltd
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
soft_margin_loss_grad
"""
import sys

from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("SoftMarginLossGrad")


#pylint: disable=unused-argument
def soft_margin_loss_grad_expect(input_predict, input_label, input_dout,
                                 output_gdient, reduction="mean"):
    input_predict = np.array(input_predict.get("value"))
    dtype = input_predict.dtype
    input_label = np.array(input_label.get("value"))
    input_dout = np.array(input_dout.get("value"))
    if dtype == 'float16':
        input_predict = input_predict.astype(np.float32)
        input_label = input_label.astype(np.float32)
        input_dout = input_dout.astype(np.float32)
    predict_neg = np.multiply(input_predict, -1)
    z = np.exp((predict_neg * input_label))

    num = 1
    if reduction == 'mean':
        for shape in input_predict.shape:
            num *= shape

    num = 1 / num
    input_label = np.multiply(input_label, -1)
    res1 = (np.multiply(input_label, num))
    res2 = np.divide(z, np.add(z, 1))
    res3 = np.multiply(res1, res2)
    res = np.multiply(res3, input_dout)

    if dtype == "float16":
        res = res.astype(np.float16)
    if dtype == "float32":
        res = res.astype(np.float32)
    return res


ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "output"}],
    "case_name": "test_is_soft_margin_loss_grad_case_1",
    "calc_expect_func": soft_margin_loss_grad_expect,
    "expect": "success"
})

# not support float32 under 310,we treansform float32 to float 16 and calculate and transform the result from float16
# to float32 so the precision is equal with float16 under 310 is 0.001
ut_case.add_precision_case("Ascend310", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 20, 30, 4, 5, 6),
                "shape": (10, 20, 30, 4, 5, 6),
                "param_type": "output"}],
    "case_name": "test_is_soft_margin_loss_grad_case_2",
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005),
    "calc_expect_func": soft_margin_loss_grad_expect,
    "expect": "success"
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "output"}],
    "case_name": "test_is_soft_margin_loss_grad_case_3",
    "calc_expect_func": soft_margin_loss_grad_expect,
    "expect": "success"
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "output"}],
    "case_name": "test_is_soft_margin_loss_grad_case_4",
    "calc_expect_func": soft_margin_loss_grad_expect,
    "expect": "success"
})

