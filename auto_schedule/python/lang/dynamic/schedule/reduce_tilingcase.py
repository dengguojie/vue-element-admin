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
reduce tiling case
"""

from te.lang.base import operation
from te.lang.base.operation import register_build_pointcut

from . import CompileInfo
from . import Pattern

from te.utils.error_manager.error_manager_util import get_error_message


@register_build_pointcut(pattern=Pattern.REDUCE)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    func(*args, **kwargs)
    _post_build()


def _post_build():
    is_const = operation.get_compile_info().get("reduce_shape_known")
    if is_const:
        tiling_keys = operation.get_context().get("_tiling_keys")
        built_jsons = operation.get_context().get("_built_info")
        block_dims, atomic_flags = {}, {}
        for tiling_key, item in zip(tiling_keys, built_jsons):
            tiling_key = str(tiling_key)
            block_dims[tiling_key] = item["blockDim"]
            atomic_flags[tiling_key] = any([x == 1 for x in item["parameters"]])

        operation.add_compile_info(CompileInfo.BLOCK_DIMS, block_dims)
        operation.add_compile_info(CompileInfo.ATOMIC_FLAGS, atomic_flags)


def _get_tiling_key(atomic, db, shape_type, block_split_axis,
                    ub_split_axis, shape, reduce_idx_list):
    """
    :param atomic: "True": atomic_reduce, "False": normal_reduce.
    :param db: int number in [0,1]. "0": enable db, "1": close db.
    :param shape_type: int number in [0,99]. Diff numbers represent diff types of
           shapes. Example: "0": normal shape, "1": const shape, "2": special shape.
    :param block_split_axis: int number in [0,7] that represent index of split axis
    :param ub_split_axis: int number in [0,7] that represent index of split axis.
    :param shape: shape before reduce
    :param reduce_idx_list:

    :return: key(int32)
    """

    def _check(idx, value):
        rule = [range(2), range(100), range(8), range(8),
                range(1000)]
        name = ["db", "shape_type", "block_split_axis", "ub_split_axis",
                "pattern"]
        if value not in rule[idx]:
            dict_args = dict()
            dict_args["errCode"] = "E90003"
            dict_args["detailed_cause"] = "%s should in %s, but is %d" % (
                                          name[idx], str(rule[idx]), value)
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def _get_pattern_key(shape, reduce_idx_list):
        pattern_key = 0
        length = len(shape)
        for i in range(length):
            if i in reduce_idx_list:
                pattern_key += 2 * 2 ** (length - i - 1)
            else:
                pattern_key += 2 ** (length - i - 1)

        return pattern_key

    pattern = _get_pattern_key(shape, reduce_idx_list)
    pos = (db, shape_type, block_split_axis, ub_split_axis,
           pattern)
    val = (10 ** 9, 10 ** 7, 10 ** 6, 10 ** 5, 10 ** 2)
    key = 0
    for item, value in enumerate(pos):
        _check(item, value)
        key += value * val[item]
    if not atomic:
        key *= -1
    return key
