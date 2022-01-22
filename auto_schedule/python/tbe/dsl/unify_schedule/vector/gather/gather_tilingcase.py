#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
gather tiling case
"""
from enum import Enum
from enum import auto
from functools import reduce
from typing import Any

from tbe.tvm.expr import Var
from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut

from ... import util
from ...computation import Computation
from ...constants import CompileInfo
from ...constants import GatherPattern
from ...constants import Pattern
from ...constants import DTYPE_BYTE_MAPPING

# gather mode
CONST = "const"
EMPTY = "empty"

DEFAULT = "default"

REMOVE_PAD_DTYPE = {
    "int8": 64, "uint8": 64, "int16": 160, "uint16": 160, "float16": 160, "int32": 168, "uint32": 168, "float": 168,
    "float32": 168, "int64": 168, "uint64": 168}


class GatherCompileInfo:
    """
    Built-in Compile info keys
    """
    CUSTOM_INFO = "_custom_info"
    GATHER_TYPE = "_gather_type"
    PARAMS_DTYPE_SIZE = "_params_dtype_size"
    INDICES_DTYPE_SIZE = "_indices_dtype_size"
    PARAMS_NUM = "_params_num"
    INDICES_NUM = "_indices_num"
    PARAMS_UB_NUM = "_params_ub_num"
    BATCH_DIMS = "_batch_dims"
    SPECIAL_PATTERN = "_special_pattern"
    CONST_AXIS = "_const_axis"
    TENSOR_SIZES = "_tensor_sizes"
    FAKE_SCHEDULE = "_fake_schedule"
    STATIC_SUCCESS = "_static_success"
    STATIC_CLOSE_PASS = "_static_close_pass"
    BASE_SCHEDULE_PATTERN = 0
    ZERO_SCHEDULE_PATTERN = 9


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    DYNAMIC = auto()
    STATIC = auto()
    CONST = auto()
    EMPTY = auto()
    ZEROS = auto()


class GatherComputation(Computation):
    """
    GatherComputation
    """
    def __init__(self, outs, option):
        self.outs = outs
        self.option = option
        indices_shape = operation.get_context().get_current_compute().get("_indices_shape")
        self.is_schedule_zero = 0 in indices_shape

    def do_tiling_case(self):
        is_static = operation.get_op_mode() == "static"
        gather_op_context = operation.get_context()
        batch_dims = gather_op_context.get("_batch_dims")
        axis = operation.get_context().get_current_compute().get("_axis")
        rank = operation.get_context().get_current_compute().get("_rank")
        indices_shape = operation.get_context().get_current_compute().get("_indices_shape")

        # zero shape
        if self.is_schedule_zero:
            tiling_key = 990000000 if (0, 0, 0) == tuple(indices_shape) else 990000001
            return [{
                "key": tiling_key,
                "block_tiling_axis": 0,
                "ub_tiling_axis": 0,
                "tiling_strategy": TilingStrategy.ZEROS}]

        # const/dynamic
        tiling_strategy = TilingStrategy.STATIC if is_static else TilingStrategy.DYNAMIC
        return self._calc_gather(batch_dims, axis, tiling_strategy, rank)

    def get_sub_pattern(self):  # type: () -> str
        if self.is_schedule_zero:
            return GatherPattern.ZERO_SCHEDULE
        return GatherPattern.NORMAL_SCHEDULE

    @staticmethod
    def gen_remov_pad_cases(dim_len, base_key, strategy, batch_dims, axis):
        remov_pad_cases = []
        for i in range(dim_len):
            for j in range(i, dim_len):
                tiling_key = base_key + i * dim_len + j
                # remove pad schedule
                # base_key + 7010
                remov_pad_cases.append({
                    "key": tiling_key + 7000,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": strategy,
                    "remove_pad": True,
                    "batch_dims": batch_dims,
                    "axis": axis,
                    "store_area": 0,
                    "is_params_align": False,
                    "is_need_align": True,
                })
        return remov_pad_cases

    @staticmethod
    def gen_store_ub_cases(dim_len, base_key, strategy, batch_dims, axis):
        store_ub_cases = []
        # scalar mode
        # pattern 6000
        for i in range(dim_len):
            for j in range(i, dim_len):
                tiling_key = base_key + i * dim_len + j
                store_ub_cases.append({
                    "key": tiling_key + 6000,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": strategy,
                    "scalar_mode": True,
                    "batch_dims": batch_dims,
                    "axis": axis,
                    "store_area": 1,
                    "is_params_align": False,
                    "is_need_align": False
                })

        for i in range(dim_len):
            for j in range(i, dim_len):
                tiling_key = base_key + i * dim_len + j
                # params store in ub and params no need align
                # pattern 1000
                store_ub_cases.append({
                    "key": tiling_key + 1000,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": strategy,
                    "batch_dims": batch_dims,
                    "axis": axis,
                    "store_area": 1,
                    "is_params_align": False,
                    "is_need_align": False
                })

                # params store in ub and params need align
                # pattern 2000
                store_ub_cases.append({
                    "key": tiling_key + 2000,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": strategy,
                    "batch_dims": batch_dims,
                    "axis": axis,
                    "store_area": 1,
                    "is_params_align": True,
                    "is_need_align": True
                })

        return store_ub_cases

    @staticmethod
    def gen_db_cases(dim_len, base_key, strategy, batch_dims, axis):
        db_cases = []
        for i in range(dim_len):
            for j in range(i, dim_len):
                tiling_key = base_key + i * dim_len + j
                # db
                # pattern 5000
                db_cases.append({
                    "key": tiling_key + 5000,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": strategy,
                    "batch_dims": batch_dims,
                    "axis": axis,
                    "is_db": True,
                    "is_need_align": False
                })

        return db_cases

    @classmethod
    def get_instance(cls, outs, option):  # type: (list[Any], dict[str, Any]) -> "Computation"
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):  # type: () -> list[str]
        return [Pattern.GATHER]

    @classmethod
    def get_supported_soc(cls):  # type: () -> list[str]
        return [DEFAULT]

    def _calc_gather(self, batch_dims, axis, strategy, rank):
        out = self.outs[0] if isinstance(self.outs, (list, tuple)) else self.outs
        shape = util.shape_to_list(out.shape)
        dim_len = len(shape)
        out_dtype = out.dtype

        base_key = self._calc_base_key(rank)

        # skip large params
        params_shape = operation.get_context().get_current_compute().get("_params_shape")
        # abs is intended to get min params shape(assume -1 dims value is 1)
        params_total_size = abs(reduce(lambda x, y: x * y, params_shape)) * DTYPE_BYTE_MAPPING.get(out_dtype)

        total_ub_size = util.get_ub_size()

        cases = []

        # special pattern
        # gather nd will generate too much tiling cases
        if rank == 1:
            if out.dtype in REMOVE_PAD_DTYPE.keys() and (isinstance(shape[-1], Var) or (
                    isinstance(shape[-1], int) and shape[-1] <= REMOVE_PAD_DTYPE.get(out_dtype))):
                cases.extend(self.gen_remov_pad_cases(dim_len, base_key, strategy, batch_dims, axis))

            if params_total_size <= total_ub_size // 2:
                cases.extend(self.gen_store_ub_cases(dim_len, base_key, strategy, batch_dims, axis))

            if isinstance(shape[-1], Var) or (isinstance(shape[-1], int) and shape[-1] != 1):
                cases.extend(self.gen_db_cases(dim_len, base_key, strategy, batch_dims, axis))

        # base tiling case, cover all condition
        for i in range(dim_len):
            for j in range(i, dim_len):
                tiling_key = base_key + i * dim_len + j
                # need align: ub split last dim and last dim not axis
                is_need_align = j < (dim_len - 1)
                cases.append({
                    "key": tiling_key,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": strategy,
                    "batch_dims": batch_dims,
                    "axis": axis,
                    "store_area": 0,
                    "is_need_align": is_need_align
                })
        return cases

    def _calc_base_key(self, rank):
        base_key = 900000000
        # add pattern
        base_key += rank * 10000

        return base_key


def _pre_build(schedules_list):
    def _flatten_sch(_schedules: list):
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                _schedules.extend(sub_schs)
            else:
                _schedules.append(sub_schs)

    def _name_to_int(_var_names):
        new_var_names = []
        for name in _var_names:
            if name[0] != "_":
                continue
            names = name[1:].split("_")
            if names[0] == "params":
                new_var_names.append(10000 + int(names[2]))
            elif names[0] == "indices":
                new_var_names.append(20000 + int(names[2]))
            elif names[0] == "block":
                new_var_names.append(30000 + int(names[2]))
            elif names[0] == "ub":
                new_var_names.append(40000 + int(names[2]))

        return new_var_names

    # set special build cfg
    operation.add_build_arg("double_buffer_non_reuse", True)

    # static buildcfg
    is_close_pass = operation.get_context().get(GatherCompileInfo.STATIC_CLOSE_PASS)
    if operation.get_op_mode() == "static" and is_close_pass:
        operation.add_build_arg("out_of_bound_sync_check", False)

    cpt_computes = operation.get_context().get_computes()

    schedules = []
    _flatten_sch(schedules)

    te_vars_list = []
    op_vars = operation.get_context().get_vars()
    cpt_cores, cpt_ub_size, cpt_params_dtype, cpt_indices_dtype, cpt_gather_type = [], [], [], [], []
    params_ub_num, batch_dims = [], []
    tensor_sizes = {}

    for cpt in cpt_computes:
        cpt_vars = cpt.get_vars()
        for sch_context in cpt.get_schedules():
            if sch_context.get(GatherCompileInfo.FAKE_SCHEDULE):
                continue
            cpt_cores.append(sch_context.get(CompileInfo.CORE_NUM))
            cpt_ub_size.append(sch_context.get(CompileInfo.UB_SIZE))
            cpt_params_dtype.append(sch_context.get(GatherCompileInfo.PARAMS_DTYPE_SIZE))
            cpt_indices_dtype.append(sch_context.get(GatherCompileInfo.INDICES_DTYPE_SIZE))
            cpt_gather_type.append(sch_context.get(GatherCompileInfo.GATHER_TYPE))

            params_ub_num.append(sch_context.get(GatherCompileInfo.PARAMS_UB_NUM))
            batch_dims.append(sch_context.get(GatherCompileInfo.BATCH_DIMS))
            special_pattern = sch_context.get(GatherCompileInfo.SPECIAL_PATTERN)
            params_num = sch_context.get(GatherCompileInfo.PARAMS_NUM)
            indices_num = sch_context.get(GatherCompileInfo.INDICES_NUM)

            sch_vars = sch_context.get_vars()
            te_vars_list.append(op_vars + cpt_vars + sch_vars)

            if special_pattern not in tensor_sizes.keys():
                tensor_sizes[special_pattern] = [params_num, indices_num]

        base_info = [max(cpt_cores), min(cpt_ub_size), max(cpt_gather_type), max(cpt_params_dtype),
                     max(cpt_indices_dtype)]
        operation.add_compile_info_inner(CompileInfo.BASE_INFO, base_info)

        unknown_batch_dims = operation.get_context().get("_unknown_batch_dims")
        if unknown_batch_dims:
            org_batch_dims = 0
        else:
            org_batch_dims = operation.get_context().get("_org_batch_dims")

        custom_info = [min(params_ub_num), max(batch_dims), unknown_batch_dims, org_batch_dims]
        operation.add_compile_info_inner(GatherCompileInfo.CUSTOM_INFO, custom_info)

        operation.add_compile_info_inner(GatherCompileInfo.TENSOR_SIZES, tensor_sizes)

        compile_vars = {}
        for sch, te_vars in zip(schedules, te_vars_list):
            if sch is None:
                continue
            var_names = [x.get_name() for x in te_vars]
            compile_vars[sch.tiling_key] = _name_to_int(var_names)
        operation.add_compile_info_inner("_gather_vars", compile_vars)


@register_build_pointcut(pattern=Pattern.GATHER)
def build_pointcut(func, *args, **kwargs):
    """
    build_pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    _pre_build(args[0])
    func(*args, **kwargs)
