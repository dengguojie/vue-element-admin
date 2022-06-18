# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
ut for util_common.py
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("util_common", "impl.util.util_common")


# pylint: disable=unused-argument, import-outside-toplevel
def test_update_dtype_bool_to_int8(test_arg):
    """
    test for util_binary api
    """
    from impl.util.util_common import update_dtype_bool_to_int8
    input_dict = {"dtype": "bool"}
    input_int8_dict = update_dtype_bool_to_int8(input_dict)
    assert input_int8_dict.get("dtype") == "int8"

    input_dict = {"dtype": "float16"}
    input_int8_dict = update_dtype_bool_to_int8(input_dict)
    assert input_int8_dict.get("dtype") == "float16"

    input_dict = {"ori_dtype": "bool"}
    input_int8_dict = update_dtype_bool_to_int8(input_dict)
    assert input_int8_dict.get("dtype") is None


ut_case.add_cust_test_func("all", test_func=test_update_dtype_bool_to_int8)


if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
