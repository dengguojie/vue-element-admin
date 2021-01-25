# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
op tiling interface
"""


def do_op_tiling(optype, compile_info, inputs, outputs, compile_info_hash=None, timer=None):
    """
    do op tilinng
    """
    from tbe.dsl.base.op_tiling import do_op_tiling as tbe_do_op_tiling
    return tbe_do_op_tiling(optype, compile_info, inputs, outputs, compile_info_hash, timer)


def decode(tiling_data, fmt):
    """decode tiling data"""
    from tbe.dsl.base.op_tiling import decode as tbe_decode
    return tbe_decode(tiling_data, fmt)
