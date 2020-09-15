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
from op_test_frame.ut import BroadcastOpUT


def np_add(x, y, z, kernel_name="np_add"):
    _ = kernel_name
    z = x.get("value") + y.get("value")
    return z


ut_case = BroadcastOpUT("Add")

ut_case.add_broadcast_case_simple(["Ascend910", "Ascend310"], "float16", (16, 32), (16, 32))

ut_case.add_precision_case(case={
    "params": [{
        "shape": [32, 32],
        "ori_shape": [32, 32],
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": [32, 32],
        "ori_shape": [32, 32],
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": [32, 32],
        "ori_shape": [32, 32],
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }],
    "calc_expect_func": np_add
})


if __name__ == "__main__":
    ut_case.run("Ascend910", simulator_mode="pv", simulator_lib_path="")

