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
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import add_compile_info_inner

from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.constants import CompileInfo
from tbe.dsl.unify_schedule.constants import TransdataCategory

from tbe.dsl.unify_schedule.util import shape_to_list
from tbe.dsl.unify_schedule.computation import Computation
from tbe.common.utils import op_tiling
from tbe.common.utils.errormgr import get_error_message

from .transdata_graph_info import ComputeGraphInfo

BLOCK = 32
DEFAULT = "default"
FORWARD = "forward"
BACKWARD = "backward"
CONST_KEY = 123

CATEGORY_MAP_UB = {
    TransdataCategory.GENERAL_FORWARD: 0,
    TransdataCategory.GENERAL_BACKWARD: 0,
}


class TransdataComputation(Computation):
    """
    Calculate Transdata TilingCase
    """

    def __init__(self, outs, option):
        self.outs = outs
        self.option = option

        self.is_const = None
        self.graph_info = None
        self.unknown_dims = None
        self.pad_factor = None
        self.align_size = None
        self.tiling_case_list = []

    def get_sub_pattern(self):
        return get_context().get_current_compute().get("_transdata_category")

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TRANSDATA]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def do_tiling_case(self):
        """
        do tiling case
        """
        return self.main_proc()

    def main_proc(self):
        outs = list(self.outs) if isinstance(self.outs, (list, tuple)) else [self.outs]
        self.graph_info = ComputeGraphInfo(outs)
        current_compute = get_context().get_current_compute()
        current_compute.add("_compute_graph_info", self.graph_info)

        self.is_const = current_compute.get("_const_model")
        self.unknown_dims = current_compute.get("_unknown_dims")
        self.pad_factor = current_compute.get("_pad_factor")
        self.align_size = int(BLOCK // DTYPE_BYTE_MAPPING[list(self.graph_info.transpose_tensor_set)[0].dtype])

        if self.is_const:
            self.tiling_case_list += self.calc_const_tiling_case()
        else:
            self.tiling_case_list += self.calc_tiling_case()
            self.apply_dynamic_compile_info()

        return self.tiling_case_list

    @staticmethod
    def normal_split(_length, _perm):
        """
        Return all split cases no matter sch support or not.
        Eg: Input:[A,B,C,D,E], output:[E,D,C,B,A], perm: [4,3,2,1,0]
        split_i : ub split in input
        split_o : ub split in output
        split_b : block split in output
        """
        _out = []
        for split_i in range(_length - 1, -1, -1):
            # axises in ub by split input that base on output
            input_axis_inner = {_perm.index(x) for x in range(split_i + 1, _length, 1)}
            for split_o in range(_length - 1, -1, -1):
                # axises in ub by split output
                output_axis_inner = set(range(split_o + 1, _length, 1))
                if split_o in input_axis_inner or _perm.index(split_i) in output_axis_inner:
                    continue
                output_axis_inner = output_axis_inner.union(input_axis_inner)
                axis_outer = set(range(_length)).difference(output_axis_inner)
                for split_b in axis_outer:
                    _out.append([split_b, _perm.index(split_i), split_o])
        return _out

    @staticmethod
    def filter_split(_inputs, _info):
        """
        Return legal cases from all split cases
        1. not split to split c0 and c1 toghther
        2. not support to split last c0
        3. maybe not existed c1 and c0
        """

        def split_c1_c0(_input):
            for c1c0 in _info.c1c0_idx_base_on_output:
                if c1c0[0] in _input and c1c0[1] in _input:
                    return True
            return False

        def split_last_c0(_input):
            if _info.c1c0_idx_base_on_output[-1][-1] in _input:
                return True
            return False

        # filter
        if _info.c1c0_idx_base_on_output:
            _out = []
            for single_input in _inputs:
                if not split_c1_c0(single_input) and not split_last_c0(single_input):
                    _out.append(single_input)
        else:
            _out = _inputs
        return _out

    @staticmethod
    def get_key(case):

        def _check(idx, _value):
            rule = [range(2), [2, 3], range(9), range(100), range(9), range(9), range(9), ]
            name = ["db", "is_forward", "ub_category", "shape_type", "block_split_idx",
                    "ub_split_first_idx", "ub_split_second_idx"]
            if _value not in rule[idx]:
                dict_args = {"errCode": "E90001", "detailed_cause": " %s should in %s, but is %d"
                                                                    % (name[idx], str(rule[idx]), _value)}
                raise RuntimeError(dict_args, get_error_message(dict_args))

        is_forward = 2 if case.sch_type == FORWARD else 3
        pos = (case.db, is_forward, case.ub_category, case.shape_type, case.block_split_idx,
               case.ub_split_first_idx, case.ub_split_second_idx)
        val = (10 ** 9, 10 ** 8, 10 ** 7, 10 ** 5, 10 ** 4, 10 ** 3, 10 ** 2)
        key = 0
        for k, v in enumerate(pos):
            _check(k, v)
            key += v * val[k]
        case.tiling_key = key

    @staticmethod
    def create_case(inputs, info):
        result = []
        for item in inputs:
            # 0: use storage_align
            # 1: use common_align
            for shape_type in range(2):
                case = TransdataTilingCase(info)
                case.block_split_idx = item[0]
                case.ub_split_first_idx = item[1]
                case.ub_split_second_idx = item[2]
                case.shape_type = shape_type
                TransdataComputation.get_key(case)
                ComputeGraphInfo.update_tensor_ub_sizes(info, case)
                result.append(case)
        return result

    def calc_tiling_case(self, ):

        def _forward_case_0():
            length = len(list(self.graph_info.output_tensor_set)[0].shape)
            result = self.normal_split(length, self.graph_info.permute)
            result = self.filter_split(result, self.graph_info)
            result = self.create_case(result, self.graph_info)
            return result

        def _backward_case_0():
            # assume split in transpose_tensor, real split in output_tensor
            length = len(list(self.graph_info.transpose_tensor_set)[0].shape)
            result = self.normal_split(length, self.graph_info.permute)
            result = self.filter_split(result, self.graph_info)
            result = self.create_case(result, self.graph_info)
            return result

        if self.graph_info.category in [TransdataCategory.GENERAL_FORWARD,
                                        TransdataCategory.GENERAL_BACKWARD]:
            return _forward_case_0() if self.graph_info.is_forward else _backward_case_0()
        else:
            dict_args = {"errCode": "E90001", "detailed_cause": " TilingCases not match the "
                                                                "transdata_category that in [general.forward, general.backward], but "
                                                                "is %s" % self.graph_info.category}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def calc_const_tiling_case(self):
        input_tensor = list(self.graph_info.input_tensor_set)[0]
        output_tensor = list(self.graph_info.output_tensor_set)[0]
        input_shape = shape_to_list(input_tensor.shape)
        output_shape = shape_to_list(output_tensor.shape)
        dtype = input_tensor.dtype

        # while in const model, sch known which transdata category,
        # but don't known which shape_type in category,
        # that mean don't known which ub_size should be used
        cst_case = TransdataTilingCase(self.graph_info)
        ComputeGraphInfo.update_tensor_ub_sizes(self.graph_info, cst_case)
        self.apply_info([cst_case], is_const_compile=True)

        inputs = [{"shape": input_shape, "dtype": dtype}]
        outputs = [{"shape": output_shape, "dtype": dtype}]
        run_info = op_tiling.do_op_tiling("Transdata", get_compile_info(), inputs, outputs)

        _format = {"ub_category": "int", "shape_type": "int", "block_split_idx": "int",
                   "ub_split_first_idx": "int", "ub_split_second_idx": "int",
                   "block_factor": "int", "ub_first_factor": "int", "ub_second_factor": "int",
                   "block_dim": "int"}
        tiling_data = op_tiling.decode(run_info["tiling_data"], _format)

        cst_case.ub_category = tiling_data["ub_category"]
        cst_case.shape_type = tiling_data["shape_type"]
        cst_case.block_split_idx = tiling_data["block_split_idx"]
        cst_case.ub_split_first_idx = tiling_data["ub_split_first_idx"]
        cst_case.ub_split_second_idx = tiling_data["ub_split_second_idx"]
        cst_case.block_factor = tiling_data["block_factor"]
        cst_case.ub_first_factor = tiling_data["ub_first_factor"]
        cst_case.ub_second_factor = tiling_data["ub_second_factor"]
        cst_case.tiling_key = CONST_KEY

        self.apply_info([cst_case], is_const_compile=False)
        block_dims = get_compile_info().get(CompileInfo.CONST_BLOCK_DIMS)
        if block_dims is None:
            block_dims = {}
            add_compile_info_inner(CompileInfo.CONST_BLOCK_DIMS, block_dims)
        block_dims[str(cst_case.tiling_key)] = tiling_data["block_dim"]
        return [cst_case, ]

    def apply_dynamic_compile_info(self):
        pre_compile_info = get_compile_info()
        if pre_compile_info:
            self.apply_info(self.tiling_case_list)
        else:
            dict_args = {"errCode": "E90001", "detailed_cause": "pre_compile_info is null"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def apply_info(self, case_list, is_const_compile=False):
        self.apply_common_info(is_const_compile)
        self.apply_ub_info(case_list)

    def apply_common_info(self, is_const_compile):
        is_forward = int(self.graph_info.is_forward)
        align_size = self.align_size
        pad_factor = self.pad_factor
        core_num = get_soc_spec("CORE_NUM")
        is_const = int(self.is_const)
        is_const_compile = int(is_const_compile)
        common_info = [is_forward, align_size, pad_factor, core_num, is_const, is_const_compile]
        add_compile_info_inner("_common_info", common_info)
        add_compile_info_inner("_unknown_dims", self.unknown_dims)

    def apply_ub_info(self, case_list):
        # case_list had full info of ubSize
        # index of tensor_list is ub_category
        # value of tensor_list is ub_size_list in different shape_type
        _case_list = sorted(case_list, key=lambda x: x.ub_category)
        tensor_list = [[-1] for i in range(_case_list[-1].ub_category + 1)]
        for _case in _case_list:
            tensor_list[_case.ub_category] = _case.tensor_ub_size_list
        ub_info = tensor_list

        # deal ub_info: different computation has different ub size
        pre_compile_info = get_compile_info()
        if "_ub_info" in pre_compile_info.keys():
            existed_info = pre_compile_info.get("_ub_info")
            if len(existed_info) >= len(ub_info):
                update, base = ub_info, existed_info
            else:
                update, base = existed_info, ub_info
            for k, v in enumerate(update):
                if base[k] == [-1, ]:
                    base[k] = v
            add_compile_info_inner("_ub_info", base)
        else:
            add_compile_info_inner("_ub_info", ub_info)


class TransdataTilingCase:
    """
    TransdataTilingCase
    """

    def __init__(self, graph):
        self.sch_type = FORWARD if graph.is_forward else BACKWARD
        self.block_split_idx = None
        self.ub_split_first_idx = None
        self.ub_split_second_idx = None
        self.block_factor = None
        self.ub_first_factor = None
        self.ub_second_factor = None

        self.tensor_ub_size_list = []
        self.db = 0
        self.shape_type = 0
        self.ub_category = CATEGORY_MAP_UB.get(graph.category, 0)
        self.tiling_key = 2 ** 31 - 1
        self.transdata_category = graph.category


@register_build_pointcut(pattern=Pattern.TRANSDATA)
def build_pointcut(func, *args, **kwargs):
    func(*args, **kwargs)
