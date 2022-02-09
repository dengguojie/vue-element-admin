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
conv2d backprop filter tiling case
"""

import json
import copy
from collections import OrderedDict
from functools import reduce
from itertools import product

from tbe import tvm
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.context import get_context
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import get_context as op_get_context
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.compute.conv2d_backprop_filter_compute import \
    DynamicConv2dBpFilterParams as DynamicParams

from .constants import Pattern
from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import TilingSelection
from .cube_tilingcase import TilingUtils as utils


H_RANGE = 4096
W_RANGE = 4096
N_RANGE = 1000000
W_DELTA = 1
H_LEN = 400
W_LEN = 400
MAX_INT64 = 9223372036854775807
MIN_INT64 = 1
DEFAULT_KERNEL_ID = None


def parse_fuzz_build_range(info_list):
    """
    parse multiple range segment from json string

    Notice
    ----------
    for conv2d_backprop_filter only parse input range

    Parameters
    ----------
    info_list: list support info
        [{
            "inputs": [{
                "index": 0,
                "tensor": [{
                    "range": [
                        [16, 32],
                        [3, 3],
                        [16, 32],
                        [16, 32]
                    ],
                    "shape": [-1, 3, -1, -1]
                }]
            }]
        }]

    Returns
    -------
    range_list: list of 4d range
    """
    invalid = (not isinstance(info_list, list)) or len(info_list) == 0
    if invalid:
        raise RuntimeError("invalid missing support info {}".format(str(info_list)))
    list_size = 4
    range_list = []
    for item in info_list:
        inputs = item.get("inputs")
        invalid = (not isinstance(inputs, list)) or len(inputs) == 0
        if invalid:
            continue
        # >>> start: parse range from index [0] input
        for input_tensor in inputs:
            invalid = (not isinstance(input_tensor, dict)) or input_tensor.get("index") != 0
            if invalid:
                continue
            invalid = (not isinstance(input_tensor.get("tensor"), list)) \
                      or len(input_tensor.get("tensor")) == 0 \
                      or (not isinstance(input_tensor.get("tensor")[0].get("range"), list)) \
                      or len(input_tensor.get("tensor")[0].get("range")) != list_size
            if invalid:
                raise RuntimeError("invalid support info input {}".format(str(input_tensor)))
            input_range = input_tensor.get("tensor")[0].get("range")
            for axis_range in input_range:
                invalid = (not isinstance(axis_range, list)) \
                          or len(axis_range) != 2 \
                          or axis_range[0] < 1 \
                          or axis_range[0] > axis_range[1]
                if invalid:
                    raise RuntimeError("invalid range {}".format(str(axis_range)))
            range_list.append(input_range)
            # <<< end: parse range from index [0] input
    return range_list


def gen_support_info(range_x, ori_tensors):
    """
    kernel list support info part

    Notice
    ------
    only need to set inputs with range

    Parameters
    ----------
    range_x: list
    input x range
    ori_tensors: dict
    orginal vaild tensors

    Returns
    -------
    support_info: dict
    """
    support_info = {}
    # >>> start: generate input shape and range
    inputs = []
    indexes = [0, 2]
    if DynamicParams.op_type in ["Conv2DBackpropFilter"]:
        input_tensores = ["x", "out_backprop"]
    elif DynamicParams.op_type in ["DepthwiseConv2DBackpropFilter"]:
        input_tensores = ["input_fm", "out_backprop"]
    for index, input_tensor in zip(indexes, input_tensores):
        item = {}
        item["index"] = index
        item["tensor"] = []
        tensor_info = {}
        ori_shape = ori_tensors.get(input_tensor).get("ori_shape")
        tensor_info["shape"] = ori_shape
        range_valid = ori_tensors.get(input_tensor).get("range")
        x_format = ori_tensors.get(input_tensor).get("ori_format")
        range_valid = [[0, 0]] * 4
        range_valid[x_format.find("N")] = range_x[0]
        range_valid[x_format.find("C")] = [ori_shape[x_format.find("C")]] * 2
        if input_tensor == "out_backprop":
            range_valid[x_format.find("N")] = [MIN_INT64, MAX_INT64]
            range_valid[x_format.find("H")] = [MIN_INT64, MAX_INT64]
            range_valid[x_format.find("W")] = [MIN_INT64, MAX_INT64]
        else:
            range_valid[x_format.find("H")] = range_x[1]
            range_valid[x_format.find("W")] = range_x[2]
        tensor_info["range"] = range_valid
        item.get("tensor").append(tensor_info)
        inputs.append(copy.deepcopy(item))
    support_info["inputs"] = inputs
    # <<< end: generate input shape and range
    return support_info


def add_covered_shape_range(compile_info):
    """
    tiling_case func for dynamic shape conv2d_backprop_filter

    Parameters
    ----------
    compile_info: dict
    tiling range info

    Returns
    -------
    info_list: dict
    support info and compile info pair
    max_id: int
    last kernel id
    """
    info_list = []
    id_list = list(compile_info["block_dim"].keys())
    id_list.sort()
    max_id = id_list[-1]
    # >>> start: add compute var for op tiling
    te_vars = []
    for cpt in op_get_context().get_computes():
        te_vars += cpt.get_vars()
    var_list = [var.get_name() for var in te_vars]
    # <<< end: add compute var for op tiling
    for kernel_id, _ in compile_info["block_dim"].items():
        new_compile = compile_info.copy()
        # >>> start: keep only one record
        for key, value in new_compile.items():
            if isinstance(value, dict):
                value = {} if value.get(kernel_id) is None else {kernel_id: value[kernel_id]}
                new_compile[key] = value
        # <<< end: keep only one record
        new_compile["kernelId"] = kernel_id
        new_compile["_vars"] = {kernel_id: var_list}
        range_x = new_compile["repo_range"].get(kernel_id) or new_compile["cost_range"].get(kernel_id)
        new_range = [range_x[:2], range_x[2:4], range_x[4:6]]
        new_support = gen_support_info(new_range, DynamicParams.ori_tensors)
        info_list.append({"supportInfo": new_support, "compileInfo": new_compile})
    return info_list, max_id


@register_build_pointcut(pattern=Pattern.CONV2D_BACKPROP_FILTER)
def build_pointcut_conv2d_bp_filter(func, *args, **kwargs):
    """
    kernel info process before build

    Notice
    ------
    kernel_info: dict with support info and compile info
        {
            "supportInfo": {
                "inputs": [{
                    "index": 0,
                    "tensor": [{
                        "range": [
                            [16, 32],
                            [3, 3],
                            [64, 128],
                            [64, 128]
                        ],
                        "shape": [-1, 3, -1, -1]
                    }]
                }]
            },
            "compileInfo": {
                "_pattern": "Conv2d_backporop_filter",
                "tiling_type": "dynamic_tiling",
                "repo_seeds": {},
                "repo_range": {},
                "cost_range": {
                    1: [16, 32, 64, 128, 64, 128]
                },
                "block_dim": {
                    1: 16
                },
                "_vars": {
                    1: ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]
                }
            }
        }

    Parameters
    ----------
    func: funtions
        build process
    args: list
        function input args
    kwargs: dict
        function input args and value

    Returns
    -------
    None
    """
    fuzz_build = get_context().get_build_type() == "fuzzily_build"
    if fuzz_build: # set kernel info
        info_list, max_id = add_covered_shape_range(get_compile_info())
        get_context().add_build_json_result("kernelList", info_list)
        get_context().add_build_json_result("maxKernelId", max_id)
    func(*args, **kwargs)


@register_tiling_case(pattern=Pattern.CONV2D_BACKPROP_FILTER)
def calc_conv2dbp_filter(outs, option=None):
    """
    tiling_case func for dynamic shape conv2d_bp_filter

    Parameters
    ----------
    option : option

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    var_names = ('batch', 'fmap_h', 'fmap_w')
    fuzz_build = get_context().get_build_type() == "fuzzily_build"
    tgt_list = []
    tgt_area = {}
    info = DynamicParams.tiling_info_dict
    shape_dict = {"batch": info.get("B_shape")[0],
                  "fmap_h": info.get("B_shape")[2],
                  "fmap_w": info.get("B_shape")[3]}
    for var_name in var_names:
        if DynamicParams.is_binary_flag:
            tgt_area[var_name] = (1, None)
        elif get_te_var(var_name):
            tgt_area[var_name] = tuple(get_te_var(var_name).get_bound())
        else:
            tgt_area[var_name] = (int(shape_dict.get(var_name)), int(shape_dict.get(var_name)))
    tgt_list.append(tgt_area)
    max_id = DEFAULT_KERNEL_ID
    if fuzz_build: # parse input range
        # generate tgt_area by format
        ori_tensors = DynamicParams.ori_tensors
        op_type = DynamicParams.op_type
        if op_type == "DepthwiseConv2DBackpropFilter":
            dedx_index = "input_fm"
        else:
            dedx_index = "x"
        invalid = (not isinstance(ori_tensors, dict)) \
                  or (not isinstance(ori_tensors.get(dedx_index), dict)) \
                  or (not isinstance(ori_tensors.get("out_backprop"), dict))
        if invalid:
            raise RuntimeError("can't get input from para_dict")
        input_format = ori_tensors.get(dedx_index).get("ori_format")
        pos_list = [input_format.find("N"),
                    input_format.find("H"),
                    input_format.find("W")]
        # te fusion make sure that each range is within the range request
        range_str = get_context().get_addition("missing_support_info")
        range_list = []
        if len(range_str) > 0:
            range_list = parse_fuzz_build_range(json.loads(range_str))
        if len(range_list) > 0:
            tgt_list.clear()
            for item in range_list:
                fuzz_area = {}
                for var, index in zip(var_names, pos_list):
                    fuzz_area[var] = tuple(item[index])
                tgt_list.append(fuzz_area)
        # >>> start: get kernel id
        kernel_id = get_context().get_addition("max_kernel_id")
        valid = isinstance(kernel_id, int) and kernel_id > -2
        if valid:
            max_id = kernel_id + 1

    tiling_cases = []
    total_info = {}
    for tgt in tgt_list:
        new_info = copy.deepcopy(info)
        tiling_op = Conv2dBpFilterTiling(new_info, DynamicParams.var_map)
        selector = TilingSelection(tiling_op, max_id)
        tiling_cases += selector.calc_tiling(tgt, var_names)
        # >>> start: gather compile_info process
        if fuzz_build:
            tgt_nhw = []
            for var_name in var_names:
                tgt_nhw.extend(tgt[var_name])
            current_info = get_compile_info().copy()
            id_list = list(current_info["block_dim"].keys())
            id_list.sort()
            max_id = id_list[-1] + 1
            # >>> start: make sure range is within tgt_nhw
            for range_key in ["repo_range", "cost_range"]:
                valid = isinstance(current_info.get(range_key), dict)
                if valid:
                    for kernel_id, range_x in current_info[range_key].items():
                        new_range = []
                        for index, dim_value in enumerate(range_x):
                            if index in (0, 2, 4):
                                new_range.append(tgt_nhw[index] if dim_value < tgt_nhw[index] else dim_value)
                            else:
                                new_range.append(tgt_nhw[index] if dim_value > tgt_nhw[index] else dim_value)
                        current_info[range_key][kernel_id] = new_range
            # <<< end: make sure range is within tgt_nhw
            if total_info:
                # >>> start: add new dict info
                for key, value in current_info.items():
                    need_update = isinstance(total_info.get(key), dict) \
                                  and isinstance(value, dict)
                    if need_update:
                        new_item = total_info.get(key)
                        new_item.update(value)
                        total_info[key] = new_item
                        add_compile_info(key, total_info.get(key))
            else:
                total_info = current_info
                # <<< end: add new dict info
        # <<< end: gather compile_info process
    return tiling_cases


class Conv2dBpFilterTiling(CubeTilingOp):
    """
    get_tiling class for dynamic shape conv2d_bp_filter
    """
    def __init__(self, tiling_info, var_map):
        super().__init__(tiling_info, var_map)
        self.a_info = self.tiling_info['A_shape']
        self.b_info = self.tiling_info['B_shape']
        self.c_info = self.tiling_info['C_shape']
        self._get_calc_info()
        self.key = 'B_shape'
        self.op_type = 'conv2d_bp_filter'
        self.var_map = var_map
        self.is_binary_flag = DynamicParams.is_binary_flag
        self.attrs = DynamicParams.attrs
        op_get_context().add("_use_cache_tiling", self.is_binary_flag)

    def get_repo_tiling(self):
        """
        get tiling from repository

        Returns
        -------
        tiling: shape and tiling retrieved from repository
        """
        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling in tiling_list:
            self._set_padding_list(tiling['B_shape'][2],
                                   tiling['B_shape'][3])
            if tiling['pad'] == self.cur_pads:
                self._get_attach_flag(tiling)
                res_list.append(tiling)
        return res_list

    def get_costmodel_tiling(self, shape):
        """
        get tiling using cost model

        Parameters
        ----------
        shape: specified shape to get tiling

        Returns
        -------
        tiling: tiling retrieved by cost model
        """

        if 'batch' in self.var_map:
            self.a_info[0] = shape if isinstance(shape, int) else shape[0]
            self.b_info[0] = shape if isinstance(shape, int) else shape[0]
        if 'fmap_h' in self.var_map:
            self.b_info[2] = shape[1]
        if 'fmap_w' in self.var_map:
            self.b_info[3] = shape[2]
        self._set_padding_list(self.b_info[2], self.b_info[3])
        self.tiling_info['padl'] = self.cur_pads[0]
        self.tiling_info['padr'] = self.cur_pads[1]
        self.tiling_info['padu'] = self.cur_pads[2]
        self.tiling_info['padd'] = self.cur_pads[3]
        self.a_info[2] = self._get_output_h(self.b_info[2])
        self.a_info[3] = self._get_output_w(self.b_info[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"

        if self.a_info[2] != 1 and self.a_info[3] == 1:
            self.tiling_info['padr'] += (self.c_info[3] - 1) * self.tiling_info['dilationW'] + 1
            self.tiling_info['strideW'] = self.b_info[3] + self.cur_pads[2] + self.cur_pads[3]

        cost_tiling = get_tiling(self.tiling_info)
        tiling = self._check_and_set_default_tiling(cost_tiling[0])
        return tiling

    def _check_and_set_default_tiling(self, tiling_in):
        """
        get tiling using cost model

        Parameters
        ----------
        shape: tiling retrieved by cost model

        Returns
        -------
        tiling: tiling retrieved by cost model
        """

        if tiling_in.get("tiling").get("AL0_matrix")[2] == 32:
            tiling_in = {"tiling": self.get_default_tiling(), "A_shape": self.a_info,
                        "B_shape":self.b_info, "C_shape": self.c_info}
        self._get_attach_flag(tiling_in)
        return tiling_in

    def get_tiling_range(self, tiling, fmap_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        fmap_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

        n_i, _, h_i, w_i, _ = fmap_shape

        if not tiling["BL1_shape"]:
            # fully load in BL1, covering lower region
            return [1, n_i, 1, h_i, 1, w_i]

        self._set_padding_list(h_i, w_i)

        # get min value
        hi_min = max(self.k_h - self.cur_pads[2] - self.cur_pads[3], 1)
        wi_min = max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1)

        hi_max = h_i
        cur_w_size = w_i
        cur_n_size = n_i

        # check tiling covering itself situation
        if not self._check_tiling_match(tiling, cur_w_size, h_i, cur_n_size) \
            or h_i > H_RANGE or w_i > W_RANGE:
            return [0, 0, 0, 0, 0, 0]

        # searching down-ward for w_min
        while self._check_tiling_match(tiling, cur_w_size, h_i, cur_n_size) and \
            cur_w_size > max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1) and \
            not (tiling.get("w_one_flag") == 1 and self._get_output_w(cur_w_size) == 1):
            wi_min = cur_w_size
            cur_w_size = cur_w_size - W_DELTA

        # searching up-ward for w_max
        cur_w_size = w_i
        wi_max = cur_w_size
        while self._check_tiling_match(tiling, cur_w_size, h_i, cur_n_size) \
            and cur_w_size <= W_RANGE and tiling.get("w_one_flag") == 1:
            wi_max = cur_w_size
            cur_w_size = cur_w_size + W_DELTA

        perf_wi_min = max(wi_min, w_i - W_LEN)
        perf_wi_max = min(wi_max, w_i + W_LEN)

        # searching down-ward for h_min based on w_min
        perf_hi_min = max(hi_min, h_i - H_LEN)
        cur_h_size = h_i
        while self._check_tiling_match(tiling, perf_wi_min, cur_h_size, cur_n_size) \
            and cur_h_size > \
            max(self.k_h - self.cur_pads[2] - self.cur_pads[3], 1):
            hi_min = cur_h_size
            cur_h_size = cur_h_size - W_DELTA
        perf_hi_min = max(hi_min, h_i - H_LEN)

        # searching up-ward for h_max based on w_max
        cur_h_size = h_i
        while self._check_tiling_match(tiling, perf_wi_max, cur_h_size, cur_n_size) \
                and cur_h_size <= H_RANGE:
            hi_max = cur_h_size
            cur_h_size = cur_h_size + W_DELTA
        perf_hi_max = min(hi_max, h_i + H_LEN)

        # searching down-ward for n_min based on w_min and h_min
        ni_min = cur_n_size
        while self._check_tiling_match(tiling, perf_wi_min, perf_hi_min, cur_n_size) \
            and cur_n_size >= 1:
            ni_min = cur_n_size
            cur_n_size = cur_n_size - W_DELTA

        cur_n_size = n_i
        ni_max = cur_n_size
        # searching up-ward for n_max based on w_max and h_max
        dynamic_l0a_attach = tiling.get("dynamic_l0a_attach")
        dynamic_l0b_attach = tiling.get("dynamic_l0b_attach")
        dynamic_al1_attach = tiling.get("dynamic_al1_attach")
        dynamic_bl1_attach = tiling.get("dynamic_bl1_attach")
        attach_set = set([dynamic_l0a_attach, dynamic_l0b_attach, dynamic_al1_attach, dynamic_bl1_attach])
        if len(attach_set) == 1 and "dw_cc" in attach_set:
            ni_max = -1
        else:
            while self._check_tiling_match(tiling, perf_wi_max, perf_hi_max, cur_n_size) \
                and cur_n_size <= N_RANGE:
                ni_max = cur_n_size
                cur_n_size = cur_n_size + W_DELTA

        if perf_wi_min > perf_wi_max:
            return [0, 0, 0, 0, 0, 0]

        perf_range = [ni_min, ni_max, perf_hi_min, perf_hi_max, perf_wi_min, perf_wi_max]
        perf_range = [int(v) for v in perf_range]
        return perf_range

    def get_batch_range(self, tiling, fmap_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        fmap_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

        n_i, _, h_i, w_i, _ = fmap_shape
        self._set_padding_list(h_i, w_i)
        h_o = self._get_output_h(h_i)
        w_o = self._get_output_w(w_i)
        if tiling.get("w_one_flag") == 2:
            w_o *= 2
        dy_shape = n_i, self.a_info[1], h_o, w_o, utils.CUBE_SIZE
        block_dim_batch = tiling.get("block_dim")[0]

        ni_min = 1
        ni_max = N_RANGE

        full_k_info = self._check_full_k(tiling, dy_shape)
        full_k_in_l0a = full_k_info.get("full_k_l0a")
        full_k_in_l0b = full_k_info.get("full_k_l0b")
        grads_l1_tiling_nparts = full_k_info.get("grads_l1_tiling_nparts")
        fmap_l1_tiling_nparts = full_k_info.get("fmap_l1_tiling_nparts")
        batch_num_sc = utils.icd(n_i, block_dim_batch)

        # based on l0_attach and l1_attach
        if (full_k_in_l0a <= 0 and full_k_in_l0b <= 0) \
            and (grads_l1_tiling_nparts[0] != 1 \
            and fmap_l1_tiling_nparts[0] != 1):
            # batch wont influence attach flag
            return [ni_min, ni_max]
        # attach flag with different batch situation
        elif batch_num_sc == 1:
            return [ni_min, block_dim_batch]
        return [block_dim_batch + 1, ni_max]

    def assembly_case(self, tiling, coverage, cnt):
        """
        Configure dict of tiling strategy and coverage

        Parameters
        ----------
        tiling: dict, tiling from repository or cost model

        coverage: list of tuple, coverage of tiling

        cnt: serial number of tiling

        Returns
        -------
        dict: describe a tiling strategy
        """
        var_range = OrderedDict()
        if not self.is_binary_flag:
            if 'batch' in self.var_map:
                var_range['batch'] = (utils.trans_to_int(coverage[0]), utils.trans_to_int(coverage[1]))
            if 'fmap_h' in self.var_map or 'fmap_w' in self.var_map:
                x_h_low, x_h_high = utils.trans_to_int(coverage[2]), utils.trans_to_int(coverage[3])
                x_w_low, x_w_high = utils.trans_to_int(coverage[4]), utils.trans_to_int(coverage[5])
                self._set_padding_list(x_h_low, x_w_low)
                dedy_h_low = self._get_output_h(x_h_low)
                dedy_w_low = self._get_output_w(x_w_low)
                if x_h_high and x_w_high:
                    self._set_padding_list(x_h_high, x_w_high)
                dedy_h_high = self._get_output_h(x_h_high)
                dedy_w_high = self._get_output_w(x_w_high)

                var_range['fmap_h'] = (x_h_low, x_h_high)
                var_range['fmap_w'] = (x_w_low, x_w_high)
                var_range['dedy_h'] = (dedy_h_low, dedy_h_high)
                var_range['dedy_w'] = (dedy_w_low, dedy_w_high)

        block_dim_multi = tiling["AUB_shape"][0] \
            if tiling["AUB_shape"] else 1
        block_dims = block_dim_multi * reduce(
            lambda x, y: x * y, tiling['block_dim'])
        correct_range_flag = DynamicParams.correct_range_flag

        return {"key": cnt, "tiling_strategy": tiling,
                "var_range": var_range, "block_dim": block_dims,
                "correct_range_flag": correct_range_flag}

    @staticmethod
    def get_default_tiling(w_bound=None):
        """
        get default tiling for unlimited range or special case

        Returns
        -------
        dict: default tiling for conv2d_bp_filter
        """
        return {
            'AUB_shape': [1, 0, 0, 0], 'BUB_shape': None,
            'AL1_shape': [utils.CUBE_SIZE, 1, 1],
            'BL1_shape': [utils.CUBE_SIZE, 1, 1],
            'AL0_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'BL0_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'CL0_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'CUB_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'block_dim': [1, 1, 1, 1],
            'cout_bef_batch_flag': 0,
            'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
            'manual_pingpong_buffer': {
                'AUB_pbuffer': 1, 'BUB_pbuffer': 1,
                'AL1_pbuffer': 1, 'BL1_pbuffer': 1,
                'AL0_pbuffer': 1, 'BL0_pbuffer': 1,
                'CL0_pbuffer': 1, 'CUB_pbuffer': 1,
                'UBG_pbuffer': 1},
            'dynamic_l0a_attach': 'dw_cc',
            'dynamic_l0b_attach': 'dw_cc',
            'dynamic_al1_attach': 'dw_cc',
            'dynamic_bl1_attach': 'dw_cc',
            'bl1_hw_allin_flag': 'dw_cc',
            'w_one_flag': 1,
        }

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.c_info[2:4]
        self.k_cin = self.c_info[1] * self.c_info[4]
        self.k_cout = self.c_info[0]
        self.stride_h, self.stride_w = self.tiling_info["strideH"], \
            self.tiling_info["strideW"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilationH"], \
            self.tiling_info["dilationW"]

        if isinstance(self.tiling_info["padl"], tvm.expr.Expr) or \
            isinstance(self.tiling_info["padu"], tvm.expr.Expr):
            self.pad_mode = "SAME"
            self.cur_pads = [-1, -1, -1, -1]
            for pad in ("padl", "padr", "padu", "padd"):
                self.tiling_info[pad] = -1
        else:
            self.pad_mode = "FIX"
            self.cur_pads = [
                self.tiling_info["padl"], self.tiling_info["padr"],
                self.tiling_info["padu"], self.tiling_info["padd"]
            ]

        self.k_h_dilation = (self.k_h - 1) * self.dilate_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilate_w + 1

    def _set_padding_list(self, cur_h, cur_w):
        """
        get padding list in cur dx shape
        """

        if self.pad_mode == "SAME":
            pad_h = max(utils.align(cur_h, self.stride_h) -
                        self.stride_h + self.k_h_dilation - cur_h, 0)
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
            pad_w = max(utils.align(cur_w, self.stride_w) -
                        self.stride_w + self.k_w_dilation - cur_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            self.cur_pads = [pad_left, pad_right, pad_up, pad_down]

    @staticmethod
    def _get_dw_k(tiling, hw_pad_1, block_dim_hw):
        if tiling["AL0_matrix"]:
            # dw_k equals to ka if L0A needs tiling
            dw_k = tiling["AL0_matrix"][1]
        elif tiling["BL0_matrix"]:
            dw_k = tiling["BL0_matrix"][0]
        else:
            # both fully loaded
            dw_k = hw_pad_1 // block_dim_hw
        return dw_k

    def _get_bound_fmap(self, tiling,
                        width_grads, width_fmap,
                        local_tiling_flag, height_grads):
        """
        get bound info for _get_bound_fmap

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        width_grads, width_fmap : int, size of w

        Returns
        -------
        int, load_length

        """

        # bl1 set storage bound
        # actual load length in k_reduce_axis
        bl1_k = tiling.get("BL1_shape")[0]
        block_dim_hw = tiling.get("AUB_shape")[0] \
            if tiling.get("AUB_shape") else 1
        hw_pad_1 = utils.icd(width_grads * height_grads, utils.CUBE_SIZE)
        flag_fmap_load2d = local_tiling_flag[-2]
        flag_conv1d_case = local_tiling_flag[-3]

        # load2d instructions refer to data_mov with raw lens
        if flag_fmap_load2d:
            return bl1_k

        dw_k = self._get_dw_k(tiling, hw_pad_1, block_dim_hw)

        hw_single_core_factor = utils.icd(utils.icd(hw_pad_1, dw_k), block_dim_hw)
        hw_single_core_factor = hw_single_core_factor * dw_k * utils.CUBE_SIZE

        if bl1_k <= width_grads:
            # tiling load lens less then width_grads, need to load a full line
            if flag_conv1d_case:
                return tiling["BL1_shape"][0]
            hw_single_core_factor = utils.icd(hw_pad_1, block_dim_hw) * \
                                    utils.CUBE_SIZE
            hw_single_core_factor = utils.align(hw_single_core_factor,
                                                dw_k * width_grads * utils.CUBE_SIZE)
            # if res_data exists then need to load 2 lines
            ho_len = 1 if ((width_grads % bl1_k == 0 and \
                            (hw_single_core_factor < width_grads or hw_single_core_factor % width_grads == 0))
                            or height_grads == 1) else 2
        else:
            # load3d instructions refer to load extra lines with pad/stride/filter
            if (bl1_k % width_grads == 0 and \
                    hw_single_core_factor % width_grads == 0) \
                    or (bl1_k > width_grads * height_grads):
                # full line could load without extra lines
                additional_rows = 0
            elif bl1_k * 2 % width_grads == 0 or \
                bl1_k % width_grads == 1:
                # every 2 load3d covered only 1 extra line
                additional_rows = 1
            else:
                # other situations need 2 extra lines in case
                additional_rows = 2
            ho_len = bl1_k // width_grads + additional_rows

        if flag_conv1d_case:
            bl1_hi = 1
            kbl1_data = utils.icd(hw_pad_1 * utils.CUBE_SIZE, block_dim_hw)
            if tiling.get("BL1_shape"):
                kbl1_data = tiling["BL1_shape"][0]
            bl1_wi = (kbl1_data - 1) * self.stride_w + self.k_w
            bl1_k_full = bl1_hi * bl1_wi
        else:
            hi_max = self.k_h + (ho_len - 1) * self.stride_h
            bl1_k_full = width_fmap * hi_max

        return bl1_k_full

    def _check_tiling_match(self, tiling, current_w, current_h, current_n):
        """

        check whether this tiling matches the shape

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        bool, True: match
            False: do not match

        """

        n_i, w_i, h_i = current_n, current_w, current_h
        self._set_padding_list(h_i, w_i)
        h_o = self._get_output_h(h_i)
        w_o = self._get_output_w(w_i)
        if tiling.get("w_one_flag") == 2:
            w_o *= 2
        howo_align = utils.align(h_o * w_o, utils.FP16_K)

        dy_shape = n_i, self.a_info[1], h_o, w_o, utils.CUBE_SIZE
        fmap_shape = n_i, self.b_info[1], h_i, w_i, utils.CUBE_SIZE

        # flag check
        constraint_flag = self._get_attach_flag_detail(tiling,
                                                         dy_shape, fmap_shape)
        local_tiling_flag = (constraint_flag.get('l0a_attach'),
                             constraint_flag.get('l0b_attach'),
                             constraint_flag.get('al1_attach'),
                             constraint_flag.get('bl1_attach'),
                             constraint_flag.get('flag_bl1k_less_than_wo'),
                             constraint_flag.get('bl1_hw_allin_flag'),
                             constraint_flag.get('batch_num_sc'),
                             constraint_flag.get('flag_conv1d_case'),
                             constraint_flag.get('flag_fmap_load2d'),
                             constraint_flag.get('k_atomic_add_len'),)
        seed_tiling_flag = (tiling["dynamic_l0a_attach"],
                            tiling["dynamic_l0b_attach"],
                            tiling["dynamic_al1_attach"],
                            tiling["dynamic_bl1_attach"],
                            tiling["flag_bl1k_less_than_wo"],
                            tiling["bl1_hw_allin_flag"],
                            tiling["batch_num_sc"],
                            tiling["flag_conv1d_case"],
                            tiling["flag_fmap_load2d"],
                            tiling["k_atomic_add_len"],)

        for index, flag in enumerate(seed_tiling_flag[:-2]):
            if flag != "dw_cc" and flag != local_tiling_flag[index]:
                return False

        # align tiling["BL1_shape"] for k_h * k_w
        tiling["BL1_shape"][1] = utils.align(
            tiling["BL1_shape"][1] * tiling["BL0_matrix"][1],
            self.k_h * self.k_w) // tiling["BL0_matrix"][1]

        # get K axis length in al1
        bl1_bound = self._get_bound_fmap(tiling,
                                         w_o, w_i, local_tiling_flag, h_o)

        # fmap size in L1 ( K * N * db * 2byte)
        fmap_l1_size = utils.FP16_SIZE * bl1_bound * \
                       tiling['BL1_shape'][1] * \
                       tiling["BL0_matrix"][1] // (self.k_h * self.k_w) * \
                       utils.FP16_N * \
                       tiling['manual_pingpong_buffer']['BL1_pbuffer']

        # grad size
        if tiling["AL1_shape"]:
            # tiling size in L1 ( M * K * db * 2byte)
            al1_m = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * \
                    utils.FP16_M
            grad_l1_size = utils.FP16_SIZE * tiling["AL1_shape"][0] * \
                           al1_m * \
                           tiling['manual_pingpong_buffer']['AL1_pbuffer']
        else:
            # fully load in AL1
            al1_m = self.k_cout
            grad_l1_size = utils.FP16_SIZE * howo_align * al1_m

        return int(fmap_l1_size + grad_l1_size) <= utils.L1BUFFER

    def _get_output_h(self, h_i):
        if not h_i:
            return None
        return max(1, (h_i + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) // self.stride_h + 1)

    def _get_output_w(self, w_i):
        if not w_i:
            return None
        return max(1, (w_i + self.cur_pads[0] + self.cur_pads[1] - self.dilate_w *
                (self.k_w - 1) - 1) // self.stride_w + 1)

    def _check_full_k(self, tiling, dy_shape):
        """
        set flag whether axis K is fully loaded in L0A and L0B
        return:
        -------
        full_k_l0a: 1 or 0, 1 means K is fully loaded in L0A
        full_k_l0b: 1 or 0, 1 means K is fully loaded in L0B
        """

        # if k is fully load in BL1 and
        # there is multi load in N1 and N1 in BL1
        # isn't aligned to kernel_height*kernel_width, then align to it
        _, _, grads_height, grads_width, _ = dy_shape
        hw_mad_1 = utils.icd(grads_height * grads_width, utils.FP16_K)

        fmap_channel_1 = utils.icd(self.k_cin, utils.CUBE_SIZE)
        fkk = fmap_channel_1 * self.k_h * self.k_w
        c1_grads = utils.icd(self.k_cout, utils.CUBE_SIZE)
        block_dim_hw = tiling.get("AUB_shape")[0] \
            if tiling.get("AUB_shape") else 1

        block_dim_cout = tiling.get("block_dim")[2]
        block_dim_cin = tiling.get("block_dim")[1]

        if tiling.get("BL1_shape"):
            tiling["BL1_shape"][1] = utils.align(
                tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1],
                self.k_h * self.k_w) // tiling.get("BL0_matrix")[1]

        # whether axis K is fully loaded in L0A and L0B
        # excluding axis batch
        dw_k = self._get_dw_k(tiling, hw_mad_1, block_dim_hw)
        hw_single_core_factor = utils.icd(utils.icd(hw_mad_1, dw_k), block_dim_hw) * dw_k
        if tiling.get("BL1_shape")[0] <= grads_width:
            hw_single_core_factor = utils.align(utils.icd(hw_mad_1, block_dim_hw), dw_k * grads_width)
        full_k_l0a = 1 \
            if not tiling["AL0_matrix"] \
            else tiling["AL0_matrix"][1] // hw_single_core_factor
        full_k_l0b = 1 \
            if not tiling["BL0_matrix"] \
            else tiling["BL0_matrix"][0] // hw_single_core_factor

        dw_tiling_factor = [tiling["CL0_matrix"][0], tiling["CL0_matrix"][1]]
        dw_tiling_nparts = \
            [utils.icd(fkk // block_dim_cin, dw_tiling_factor[0]),
             utils.icd(utils.icd(c1_grads, dw_tiling_factor[1]), block_dim_cout)]

        if tiling["AL1_shape"]:  # if grads needs tiling in L1
            if len(tiling["AL1_shape"]) == 1:  # but no C_1 tiling info
                tiling["AL1_shape"] = tiling["AL1_shape"] + [1]
            # nparts K1 in L1, nparts M1 in L1
            grads_l1_tiling_nparts = [
                utils.icd(hw_single_core_factor,
                           (tiling["AL1_shape"][0] // utils.CUBE_SIZE)),
                dw_tiling_nparts[1] // tiling["AL1_shape"][1]]
        else:
            grads_l1_tiling_nparts = [1, 1]

        if tiling["BL1_shape"]:  # if fmap needs tiling in L1
            if len(tiling["BL1_shape"]) == 1:  # but no fkk tiling info
                tiling["BL1_shape"] = \
                    tiling["BL1_shape"] + [1]  # tiling fkk=1
            # DDR to L1 [nparts K1, nparts N1]
            fmap_l1_tiling_nparts = [
                utils.icd(hw_single_core_factor,
                           (tiling["BL1_shape"][0] // utils.CUBE_SIZE)),
                dw_tiling_nparts[0] // tiling["BL1_shape"][1]]
        else:
            fmap_l1_tiling_nparts = [1, 1]

        return {"full_k_l0a": full_k_l0a, "full_k_l0b": full_k_l0b, \
               "grads_l1_tiling_nparts": grads_l1_tiling_nparts, "fmap_l1_tiling_nparts": fmap_l1_tiling_nparts}

    def _get_attach_flag(self, tiling_extend):
        """
        tiling_extend: tiling with "A_shape", "B_shape", "C_shape"
        """

        tiling = tiling_extend["tiling"]
        dy_shape = tiling_extend["A_shape"]
        fmap_shape = tiling_extend["B_shape"]
        w_one_flag = 2 if dy_shape[2] != 1 and dy_shape[3] == 1 else 1
        if w_one_flag == 2:
            dy_shape = (dy_shape[0], dy_shape[1], dy_shape[2], dy_shape[3] * 2, dy_shape[4])
        constraint_flag = self._get_attach_flag_detail(tiling, dy_shape, fmap_shape)

        tiling.update({
            "dynamic_l0a_attach": constraint_flag.get('l0a_attach'),
            "dynamic_l0b_attach": constraint_flag.get('l0b_attach'),
            "dynamic_al1_attach": constraint_flag.get('al1_attach'),
            "dynamic_bl1_attach": constraint_flag.get('bl1_attach'),
            "flag_bl1k_less_than_wo":constraint_flag.get('flag_bl1k_less_than_wo'),
            "bl1_hw_allin_flag": constraint_flag.get('bl1_hw_allin_flag'),
            "batch_num_sc": constraint_flag.get('batch_num_sc'),
            "flag_conv1d_case": constraint_flag.get('flag_conv1d_case'),
            "flag_fmap_load2d": constraint_flag.get('flag_fmap_load2d'),
            "k_atomic_add_len": constraint_flag.get('k_atomic_add_len'),
            "w_one_flag": w_one_flag})

    def _get_attach_flag_detail(self, tiling, dy_shape, fmap_shape):
        l0a_attach = None
        l0b_attach = None
        al1_attach = None
        bl1_attach = None
        bl1_hw_allin_flag = False
        flag_conv1d_case = False
        flag_fmap_load2d = False
        k_atomic_add_len = -1

        batch = dy_shape[0]
        flag_bl1k_less_than_wo = tiling.get('BL1_shape')[0] <= dy_shape[3]
        block_dim_batch = tiling.get("block_dim")[0]
        block_dim_hw = tiling.get("AUB_shape")[0] if tiling.get("AUB_shape") else 1
        batch_num_sc = utils.icd(batch, block_dim_batch)

        height_all_one = dy_shape[2] == 1 and fmap_shape[2] == 1 \
            and self.k_h == 1
        width_all_one = dy_shape[3] == 1 and fmap_shape[3] == 1 \
            and self.k_w == 1

        # conv1d_split_w
        flag_conv1d_case = height_all_one and not width_all_one

        # load2d check
        flag_fmap_load2d = height_all_one and width_all_one

        full_k_info = self._check_full_k(tiling, dy_shape)
        full_k_in_l0a = full_k_info.get("full_k_l0a")
        full_k_in_l0b = full_k_info.get("full_k_l0b")
        grads_l1_tiling_nparts = full_k_info.get("grads_l1_tiling_nparts")
        fmap_l1_tiling_nparts = full_k_info.get("fmap_l1_tiling_nparts")
        l0a_attach, l0b_attach = self._get_l0_attach(
            tiling, batch_num_sc, full_k_in_l0a, full_k_in_l0b)
        al1_attach, bl1_attach = self._get_l1_attach(
            tiling, batch_num_sc, grads_l1_tiling_nparts, fmap_l1_tiling_nparts)

        bl1_hw_allin_flag = self._get_bl1_hw_allin_flag(
            tiling, fmap_l1_tiling_nparts)

        fmap_hw_align = utils.align(fmap_shape[2] * fmap_shape[3], 16)
        k_atomic_add_len = utils.align(
            utils.icd(fmap_hw_align, block_dim_hw), 16)

        constraint_flag = {
            "l0a_attach": l0a_attach,
            "l0b_attach": l0b_attach,
            "al1_attach": al1_attach,
            "bl1_attach": bl1_attach,
            "flag_bl1k_less_than_wo": flag_bl1k_less_than_wo,
            "bl1_hw_allin_flag": bl1_hw_allin_flag,
            "batch_num_sc": batch_num_sc == 1,
            "flag_conv1d_case": flag_conv1d_case,
            "flag_fmap_load2d": flag_fmap_load2d,
            "k_atomic_add_len": k_atomic_add_len,
        }
        return constraint_flag

    @staticmethod
    def _get_bl1_hw_allin_flag(tiling, fmap_l1_tiling_nparts):
        if tiling["BL1_shape"]:
            if fmap_l1_tiling_nparts[0] == 1:
                return True
        else:
            return True
        return False

    @staticmethod
    def _get_l0_attach(tiling, batch_num_sc, full_k_in_l0a, full_k_in_l0b):
        l0a_attach = None
        l0b_attach = None

        if tiling["AL0_matrix"]:
            l0a_attach = "dw_ddr" if batch_num_sc == 1 and full_k_in_l0a > 0 \
                else "dw_cc"

        if tiling["BL0_matrix"]:
            l0b_attach = "dw_ddr" if batch_num_sc == 1 and full_k_in_l0b > 0 \
                else "dw_cc"
        return l0a_attach, l0b_attach

    @staticmethod
    def _get_l1_attach(tiling, batch_num_sc, grads_l1_tiling_nparts, fmap_l1_tiling_nparts):
        al1_attach = None
        bl1_attach = None

        if tiling["AL1_shape"]:
            # if axis K needs split, then attach to dw_cc, else attach to dw_ddr
            al1_attach = "dw_cc" if grads_l1_tiling_nparts[0] != 1 or \
                batch_num_sc != 1 else "dw_ddr"

        if tiling["BL1_shape"]:
            # if axis K needs split, then attach to dw_cc else attach to dw_ddr
            bl1_attach = "dw_cc" if fmap_l1_tiling_nparts[0] != 1 or \
                batch_num_sc != 1 else "dw_ddr"
        return al1_attach, bl1_attach

    def _check_invalid_choice(self, choice):
        ## Drop invalid choices
        # 1) a_l1 full_load or a_l1 full_k, b_l1 full_load or full_k => a_kl1 = a_bkl1;
        invalid_choice = (
            choice[4] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and
            choice[5] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and
            choice[3] != utils.ATTACH_FULL_LOAD
        )

        # 2) a_l1 full_load or full_k, b_l1 k_split => a_kl1 > b_kl1 & reorder_l1_mn = 1;
        invalid_choice |= (
            choice[4] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and
            choice[5] == utils.ATTACH_LESS and
            (choice[3] != utils.ATTACH_EQUAL or choice[9] != 1)
        )

        # 3) a_l1 k_split, b_l1 full_load or full_k => a_kl1 < b_kl1 & reorder_l1_mn = 0;
        invalid_choice |= (
            choice[4] == utils.ATTACH_LESS and
            choice[5] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and
            (choice[3] != utils.ATTACH_LESS or choice[9] != 0)
        )

        # 4) a_l1 k_split, b_l1 k_split => reorder_l1_mn = 0
        invalid_choice |= (
            choice[4] == utils.ATTACH_LESS and
            choice[5] == utils.ATTACH_LESS and
            choice[9] != 0
        )

        # 5) a_l1 full_load, b_l1 full_load => reorder_l1_mn = 0
        invalid_choice |= (
            choice[4] == utils.ATTACH_FULL_LOAD and
            choice[5] == utils.ATTACH_FULL_LOAD and
            choice[9] != 0
        )

        # 6) a_l1@l0c or b_l1@l0c or min_kl1_cmp_kl0 = 1 => reorder_l0_mn = 0
        invalid_choice |= (
            (choice[4] == utils.ATTACH_LESS or
            choice[5] == utils.ATTACH_LESS or
            choice[6] == 1) and
            choice[10] == 1
        )

        # 7) a_l1 full_load => a_l1_pb off
        invalid_choice |= (
            choice[4] == utils.ATTACH_FULL_LOAD and
            choice[0] != utils.DB_OFF
        )

        # 8) b_l1 full_load => b_l1_pb off
        invalid_choice |= (
            choice[5] == utils.ATTACH_FULL_LOAD and
            choice[1] != utils.DB_OFF
        )

        return invalid_choice

    def get_cache_tiling(self):
        '''
        Generate tiling cases based on combinations of all attach flags.
        ---------------------------------------------
        al1 @l0c and bl1 @l0c                       |
        al1 @l0c and bl1 @ddr                       |
        al1 @l0c and bl1 full load                  |
        al1 @ddr and bl1 @l0c                       |
        al1 @ddr and bl1 @ddr                       |
        al1 @ddr and bl1 full load                  |
        al1 full load and bl1 @l0c                  |
        al1 full load and bl1 @ddr                  |
        al1 full load and bl1 full load             |

        Returns
        ----------
        tiling_cases: list of all tiling templates.
        '''
        # get cache_tiling
        cache_tiling_all = {}
        (al1_pb, bl1_pb, l0c_pb, abkl1_attach, al1_attach_flag, bl1_attach_flag,
        min_kl1_cmp_kl0, aub_multi_flag, bub_multi_flag, reorder_l1_mn, reorder_l0_mn) = (
            [utils.DB_OFF, utils.DB_ON], [utils.DB_OFF, utils.DB_ON], [utils.DB_OFF, utils.DB_ON],
            [utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL, utils.ATTACH_LESS],
            [utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL, utils.ATTACH_LESS],
            [utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL, utils.ATTACH_LESS],
            [0, 1],
            [utils.ABUB_NOT_FULL_LOAD, utils.ABUB_INNER_FULL_LOAD, utils.ABUB_FULL_LOAD],
            [utils.ABUB_NOT_FULL_LOAD, utils.ABUB_INNER_FULL_LOAD, utils.ABUB_FULL_LOAD],
            [0, 1],
            [0, 1])

        l1_choice = list(
            product(al1_pb, bl1_pb, l0c_pb, abkl1_attach, al1_attach_flag, bl1_attach_flag, min_kl1_cmp_kl0,
                    aub_multi_flag, bub_multi_flag, reorder_l1_mn, reorder_l0_mn))

        for choice in l1_choice:
            cache_tiling = {
                'block_dim': [-1, -1, -1, 1],
                'AL0_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'BL0_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'CL0_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'CUB_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'BUB_shape': [-1, -1, 1, 1],
                'AL1_shape': [-1, -1, 1, 1], 'BL1_shape': [-1, -1, 1, 1],
                'AUB_shape': [-1, -1, 1, 1],
                'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'batch_bef_group_flag': 0,
                'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
                'AUB_channel_wise_flag': None, 'BUB_channel_wise_flag': None, 'CUB_channel_wise_flag': None,
                'manual_pingpong_buffer': {'AUB_pbuffer': utils.DB_ON, 'BUB_pbuffer': utils.DB_ON,
                'AL1_pbuffer': utils.DB_ON, 'BL1_pbuffer': utils.DB_ON,
                'AL0_pbuffer': utils.DB_ON, 'BL0_pbuffer': utils.DB_ON, 'CL0_pbuffer': utils.DB_ON,
                'CUB_pbuffer': utils.DB_ON, 'UBG_pbuffer': utils.DB_OFF},
                'attach_at_flag': {'cub_attach_flag': utils.ATTACH_LESS,
                'cl0_attach_flag': utils.ATTACH_LARGE, 'al0_attach_flag': utils.ATTACH_LESS,
                'bl0_attach_flag': utils.ATTACH_LESS,
                'al1_attach_flag': -1, 'bl1_attach_flag': -1, 'aub_attach_flag': utils.ATTACH_LESS,
                'abkl1_attach_flag': -1, 'aub_multi_flag': -1, 'bub_multi_flag': -1},
                'reorder_l1_mn': -1,
                'reorder_l0_mn': -1,
            }

            if self._check_invalid_choice(choice):
                continue

            cache_tiling.get('manual_pingpong_buffer')['AL1_pbuffer'] = choice[0]
            cache_tiling.get('manual_pingpong_buffer')['BL1_pbuffer'] = choice[1]
            cache_tiling.get('manual_pingpong_buffer')['CL0_pbuffer'] = choice[2]
            cache_tiling.get('attach_at_flag')['abkl1_attach_flag'] = choice[3]
            cache_tiling.get('attach_at_flag')['al1_attach_flag'] = choice[4]
            cache_tiling.get('attach_at_flag')['bl1_attach_flag'] = choice[5]
            cache_tiling.get('attach_at_flag')['min_kl1_cmp_kl0'] = choice[6]
            cache_tiling.get('attach_at_flag')['aub_multi_flag'] = choice[7]
            cache_tiling.get('attach_at_flag')['bub_multi_flag'] = choice[8]
            cache_tiling['reorder_l1_mn'] = choice[9]
            cache_tiling['reorder_l0_mn'] = choice[10]

            name = reduce(lambda x, y: 5*x + y, choice)
            cache_tiling_all[name] = [[], cache_tiling, []]
            tiling_cases = [self.assembly_case(v[1], v[0], k) for k, v in cache_tiling_all.items()]

        return tiling_cases
