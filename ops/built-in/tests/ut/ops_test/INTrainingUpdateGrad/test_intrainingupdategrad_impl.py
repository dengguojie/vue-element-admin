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
INTrainingUpdateV2 ut testcase
"""

# pylint: disable=too-many-arguments,unused-variable,invalid-name,missing-function-docstring,unused-argument,too-many-locals
import os
import numpy as np
from op_test_frame.ut import OpUT

ut_case = OpUT("INTrainingUpdateGrad", "impl.in_training_update_grad", "in_training_update_grad")


def gen_in_training_update_grad_case(shape_x, shape_var, data_format, dtype_x, dtype_var, kernel_name, expect):
    return {
        "params": [{
            "shape": shape_x,
            "ori_shape": shape_x,
            "dtype": dtype_x,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_x,
            "ori_shape": shape_x,
            "dtype": dtype_x,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_var,
            "ori_shape": shape_var,
            "dtype": dtype_var,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_var,
            "ori_shape": shape_var,
            "dtype": dtype_var,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_var,
            "ori_shape": shape_var,
            "dtype": dtype_var,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_var,
            "ori_shape": shape_var,
            "dtype": dtype_var,
            "format": data_format,
            "ori_format": data_format
        }],
        "case_name": kernel_name,
        "expect": expect
    }


ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_in_training_update_grad_case((2, 128, 2, 128, 128, 16), (2, 1, 2, 1, 1, 16), "NDC1HWC0", "float32",
                                                  "float32", "test_right_001", "success"))
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_in_training_update_grad_case((2, 128, 2, 128, 128, 16), (2, 1, 2, 1, 1, 16), "NDC1HWC0", "float16",
                                                  "float32", "test_right_002", "success"))

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, "/usr/local/Ascend/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
