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
Transdata Schedule Remake stage 1 - Tilingcase
"""

# Standard Package
from tbe.common.utils import op_tiling
from tbe.common.utils.errormgr import get_error_message
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.base.operation import get_context, register_build_pointcut
from tbe.dsl.base.operation import get_compile_info, add_compile_info_inner

from tbe.dsl.unify_schedule.util import shape_to_list
from tbe.dsl.unify_schedule.computation import Computation
from tbe.dsl.unify_schedule.constants import Pattern, CompileInfo

from .transdata_graph_info import ComputeGraphInfo
from .transdata_key import TransdataCase, TransdataSplit
from .constants import BLOCK, DEFAULT, FORMAT, BASE_BRANCH


class TransdataComputation(Computation, TransdataSplit):
    """
    Calculate Transdata TilingCase
    """

    def __init__(self, outs, option):
        TransdataSplit.__init__(self, outs, option)
        self.tiling_case_list = []

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TRANSDATA]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @staticmethod
    def apply_ub_info(case_list):
        """
        case_list: contains ubSize that would be used in schedule
        ub_info: index is ub_category, value is ubSize-list in different shape_type
        """
        ub_category_count = 3
        ub_info = [[-1] for i in range(ub_category_count)]
        for i in case_list:
            ub_info[i.ub_category] = i.tensor_ub_size_list

        # deal ub_info: different computation has different ub size
        if "_ub_info" in get_compile_info().keys():
            pre_info = get_compile_info().get("_ub_info")
            for k, v in enumerate(ub_info):
                if pre_info[k] == [-1, ] and v != [-1, ]:
                    pre_info[k] = v
            ub_info = pre_info
        add_compile_info_inner("_ub_info", ub_info)

    def get_sub_pattern(self):
        return get_context().get_current_compute().get("_transdata_category")

    def do_tiling_case(self):
        """
        do tiling case
        """
        outs = list(self.outs) if isinstance(self.outs, (list, tuple)) else [self.outs]
        self.graph_info = ComputeGraphInfo(outs)
        current_compute = get_context().get_current_compute()
        current_compute.add("_compute_graph_info", self.graph_info)

        # while src-tensor is fp32, schedule would reinterpret it by fp16-mode,
        # ori-bit is size of fp32-element.
        self.bit = current_compute.get("_bit")
        self.ori_bit = current_compute.get("_ori_bit")
        self.is_const = current_compute.get("_const_model")

        if self.is_const:
            self.align_size = BLOCK // self.bit
            self.tiling_case_list += self.calc_const_tiling_case()
        else:
            self.align_size = BLOCK // self.ori_bit
            self.tiling_case_list += self.calc_tiling_case()
            self.apply_dynamic_compile_info()

        return self.tiling_case_list

    def calc_tiling_case(self, ):

        def _base_case():
            if self.graph_info.is_forward:
                length = len(list(self.graph_info.output_tensor_set)[0].shape)
            else:
                length = len(list(self.graph_info.transpose_tensor_set)[0].shape)
            result = self.split(length, self.graph_info.permute)
            result = self.base_filter(result)
            return self.base_generation(result)

        def _borrow_n_h_case():
            length = len(self.graph_info.transpose_2_tensor.shape)
            perm = [int(x) for x in self.graph_info.transpose_2_tensor.op.attrs["permute"]]
            result = self.split(length, perm)
            result = self.borrow_filter(result)
            return self.borrow_generation(result)

        return _base_case() if self.graph_info.category in BASE_BRANCH else _borrow_n_h_case()

    def calc_const_tiling_case(self):
        input_tensor = list(self.graph_info.input_tensor_set)[0]
        output_tensor = list(self.graph_info.output_tensor_set)[0]
        input_shape = shape_to_list(input_tensor.shape)
        output_shape = shape_to_list(output_tensor.shape)

        # while in const model, sch known which transdata category,
        # but don't known which shape_type in category,
        # that mean don't known which ub_size should be used
        cst_case = TransdataCase(self.graph_info)
        ComputeGraphInfo.update_tensor_ub_sizes(self.graph_info, cst_case)
        self.apply_info([cst_case], is_const_compile=True)

        inputs = [{"shape": input_shape, "dtype": input_tensor.dtype}]
        outputs = [{"shape": output_shape, "dtype": output_tensor.dtype}]
        run_info = op_tiling.do_op_tiling("Transdata", get_compile_info(), inputs, outputs)

        tiling_data = op_tiling.decode(run_info["tiling_data"], FORMAT)

        cst_case.ub_category = tiling_data["ub_category"]
        cst_case.shape_type = tiling_data["shape_type"]
        cst_case.block_split_idx = tiling_data["block_split_idx"]
        cst_case.ub_split_first_idx = tiling_data["ub_split_first_idx"]
        cst_case.ub_split_second_idx = tiling_data["ub_split_second_idx"]
        cst_case.block_factor = tiling_data["block_factor"]
        cst_case.ub_first_factor = tiling_data["ub_first_factor"]
        cst_case.ub_second_factor = tiling_data["ub_second_factor"]
        cst_case.transpose_work = tiling_data["transpose_work"]
        cst_case.avoid_bank_conflict = tiling_data["avoid_bank_conflict"]

        self.apply_info([cst_case], is_const_compile=False)
        block_dims = get_compile_info().get(CompileInfo.CONST_BLOCK_DIMS)
        if block_dims is None:
            block_dims = {}
            add_compile_info_inner(CompileInfo.CONST_BLOCK_DIMS, block_dims)
        block_dims[str(cst_case.tiling_key)] = tiling_data["block_dim"]
        return [cst_case, ]

    def apply_dynamic_compile_info(self):
        if not get_compile_info():
            dict_args = {"errCode": "E90003", "detailed_cause": "pre_compile_info is null"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        self.apply_info(self.tiling_case_list)

    def apply_info(self, case_list, is_const_compile=False):
        self.apply_common_info(is_const_compile)
        self.apply_ub_info(case_list)

    def apply_common_info(self, is_const_compile):
        is_forward = int(self.graph_info.is_forward)
        align_size = self.align_size
        core_num = get_soc_spec("CORE_NUM")
        is_const = int(self.is_const)
        is_const_compile = int(is_const_compile)
        common_info = [is_forward, align_size, core_num, is_const, is_const_compile]
        add_compile_info_inner("_common_info", common_info)


@register_build_pointcut(pattern=Pattern.TRANSDATA)
def build_pointcut(func, *args, **kwargs):
    func(*args, **kwargs)
