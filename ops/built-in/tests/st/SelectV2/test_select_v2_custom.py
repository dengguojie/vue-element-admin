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
ut for util_binary.py
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("SelectV2", "impl.dynamic.select_v2", "select_v2")


def test_import_lib(test_arg):
    from impl.dynamic import binary_query_register


ut_case.add_cust_test_func(test_func=test_import_lib)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
