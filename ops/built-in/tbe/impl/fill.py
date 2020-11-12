# Copyright 2019 Huawei Technologies Co., Ltd
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


from te.utils import para_check


# pylint: disable=unused-argument
def check_supported(dims, value, y, kernel_name="fill"):
    """
    verify the types of cast supported by tbe
    """
    if -2 in dims["shape"] or -2 in y["shape"] or -1 in dims["shape"] or -2 in value["ori_shape"]:
        return False
    return True


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fill(dims, value, y, kernel_name="fill"):
    pass
