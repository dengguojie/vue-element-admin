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
InstanceNorm ut testcase
"""

# pylint: disable=too-many-arguments,unused-variable,invalid-name,missing-function-docstring,unused-argument,too-many-function-args
import os
from op_test_frame.ut import OpUT

ut_case = OpUT("InstanceNorm", "impl.instance_norm", "instance_norm")


def gen_instance_norm_case(shape_x, shape_gamma, shape_mean, data_format, dtype, kernel_name, expect):
    return {
        "params": [{
            "shape": shape_x,
            "ori_shape": shape_x,
            "dtype": dtype,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_gamma,
            "dtype": dtype,
            "ori_shape": shape_gamma,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_gamma,
            "dtype": dtype,
            "ori_shape": shape_gamma,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_x,
            "ori_shape": shape_x,
            "dtype": dtype,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_mean,
            "ori_shape": shape_mean,
            "dtype": dtype,
            "format": data_format,
            "ori_format": data_format
        }, {
            "shape": shape_mean,
            "ori_shape": shape_mean,
            "dtype": dtype,
            "format": data_format,
            "ori_format": data_format
        }],
        "case_name": kernel_name,
        "expect": expect
    }


ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_instance_norm_case((2, 5, 8, 7, 16), (16,), (2, 1, 1, 1, 16), "NDHWC", "float32", "test_right_001",
                                        "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_instance_norm_case((2, 5, 8, 7, 16), (16,), (2, 1, 1, 1, 16), "NDHWC", "float16", "test_right_002",
                                        "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_instance_norm_case((2, 16, 5, 8, 7), (16,), (2, 16, 1, 1, 1), "NCDHW", "float32", "test_right_003",
                                        "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_instance_norm_case((2, 16, 5, 8, 7), (16,), (2, 16, 1, 1, 1), "NCDHW", "float16", "test_right_004",
                                        "success"))

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
