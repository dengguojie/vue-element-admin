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
test_l2_loss_for_custom
"""
from op_test_frame.ut import OpUT
from impl.dynamic.l2_loss import l2_loss


ut_case = OpUT("L2Loss", "impl.dynamic.l2_loss", "l2_loss")


def test_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.util.reduce_pattern_adapter"))

ut_case.add_cust_test_func(test_func=test_import_lib)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
