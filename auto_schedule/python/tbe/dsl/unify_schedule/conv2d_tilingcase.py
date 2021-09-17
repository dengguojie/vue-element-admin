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
conv2d tiling case
"""
import copy
import math
import json
from collections import OrderedDict

from tbe.tvm.expr import Expr
from tbe.tvm import schedule as tvm
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.context import get_context

from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import get_context as op_get_context

from tbe.dsl.compute.conv_compute import ConvParam
from tbe.dsl.static_schedule.conv_schedule import CceConvOp
from tbe.dsl.static_schedule.conv_schedule import reget_tensor_list
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from .conv2d_schedule import get_op_tensor_map
from .cube_tilingcase import TilingSelection
from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import TilingUtils as utils
from .constants import Pattern

CUBE_INFO = {'type_size':{'int8': 1, 'float16': 2, 'float32': 4, 'int32': 4},
             'reduce_k0':{'int8': 32, 'float16': 16, 'float32': 8, 'int32': 8}}

TILING_REPO_MODE = 0
TILING_COST_MODE = 1


def parse_fuzz_build_range(info_list):
    """
    parse multiple range segment from json string

    Notice
    ----------
    for conv2d only parse input range

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
    range_list = []
    if invalid:
        return range_list
    list_size = 4
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


def gen_support_info(range_x, para_dict):
    """
    kernel list support info part

    Notice
    ------
    only need to set inputs with range

    Parameters
    ----------
    range_x: list
         input x range
    para_dict: dict
        contains orginal vaild tensors

    Returns
    -------
    support_info: dict
    """
    support_info = {}
    # >>> start: generate input shape and range
    inputs = []
    ori_tensors = para_dict.get("ori_tensors")
    item = {}
    item["index"] = 0
    item["tensor"] = []
    tensor_info = {}
    ori_shape = ori_tensors.get("inputs").get("ori_shape")
    tensor_info["shape"] = ori_shape
    range_valid = ori_tensors.get("inputs").get("range")
    x_format = ori_tensors.get("inputs").get("ori_format")
    range_valid = [[0, 0]] * 4
    range_valid[x_format.find("N")] = range_x[0]
    range_valid[x_format.find("C")] = [ori_shape[x_format.find("C")]] * 2
    range_valid[x_format.find("H")] = range_x[1]
    range_valid[x_format.find("W")] = range_x[2]
    tensor_info["range"] = range_valid
    item["tensor"].append(tensor_info)
    inputs.append(item)
    support_info["inputs"] = inputs
    # <<< end: generate input shape and range
    return support_info


def add_covered_shape_range(compile_info):
    """
    tiling_case func for dynamic shape conv2d

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
    for kernel_id, block_id in compile_info["block_dim"].items():
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
        new_support = gen_support_info(new_range, ConvParam.para_dict)
        info_list.append({"supportInfo": new_support, "compileInfo": new_compile})
    return info_list, max_id


@register_build_pointcut(pattern=Pattern.CONV2D)
def build_pointcut_conv2d(func, *args, **kwargs):
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
                "_pattern": "Convolution",
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


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.CONV2D)
def calc_conv2d(outs, option=None):
    """
    tiling_case func for dynamic shape conv2d

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    var_names = ["batch_n", "fmap_h", "fmap_w"]
    fuzz_build = get_context().get_build_type() == "fuzzily_build"
    conv_info = ConvParam.tiling_info_dict
    tgt_list = []
    tgt_area = {}
    shape_dict = {"batch_n": conv_info.get("a_shape")[0],
                  "fmap_h": conv_info.get("a_shape")[2],
                  "fmap_w": conv_info.get("a_shape")[3]}
    for var_name in var_names:
        if get_te_var(var_name):
            tgt_area[var_name] = tuple(get_te_var(var_name).get_bound())
        else:
            tgt_area[var_name] = (int(shape_dict.get(var_name)), int(shape_dict.get(var_name)))
    new_in_range = ConvParam.dynamic_para.get("new_in_range")
    correct_range_flag = ConvParam.dynamic_para.get("correct_range_flag", False)
    if correct_range_flag:
        if tgt_area["fmap_h"][0] != tgt_area["fmap_h"][1]:
            tgt_area["fmap_h"] = tuple(new_in_range[2])
        if tgt_area["fmap_w"][0] != tgt_area["fmap_w"][1]:
            tgt_area["fmap_w"] = tuple(new_in_range[3])
    tgt_list.append(tgt_area)
    # >>> start: generate tgt_area by format
    if fuzz_build:
        ori_tensors = ConvParam.para_dict.get("ori_tensors")
        invalid = (not isinstance(ori_tensors, dict)) \
                  or (not isinstance(ori_tensors.get("inputs"), dict))
        if invalid:
            raise RuntimeError("can't get input from para_dict")
        input_format = ori_tensors["inputs"]["ori_format"]
        pos_list = [input_format.find("N"),
                    input_format.find("H"),
                    input_format.find("W")]
        # te fusion make sure that each range is within the range request
        range_str = get_context().get_addition("missing_support_info")
        range_list = []
        valid = isinstance(range_str, str) and (len(range_str) > 0)
        if valid:
            range_list = parse_fuzz_build_range(json.loads(range_str))
        if len(range_list) > 0:
            tgt_list.clear()
            for item in range_list:
                fuzz_area = {}
                for var, index in zip(var_names, pos_list):
                    fuzz_area[var] = tuple(item[index])
                tgt_list.append(fuzz_area)
    # <<< end: generate tgt_area by format

    cce_conv_op = CceConvOp()
    op_info = get_op_tensor_map(outs)
    schedule = tvm.create_schedule(
                [res.op for res in outs if res not in op_info])
    if ConvParam.convbn1_flag:
        res_out = outs[-1]
    else:
        res_out = outs[0]
    tiling_dict = cce_conv_op.schedule(res_out, outs, [schedule], convbn1_flag=ConvParam.convbn1_flag,
                                       tilingdict_flag=True)
    tiling_dict["dynamic_shape_flag"] = True
    add_compile_info("fmap_c1", ConvParam.dim_map["fmap_5hd_shape"][1])

    tiling_cases = []
    total_info = {}
    cnt = None
    if fuzz_build:
        # >>> start: get kernel id
        kernel_id = get_context().get_addition("max_kernel_id")
        valid = isinstance(kernel_id, int) and kernel_id > -2
        if valid:
            cnt = kernel_id + 1
        # <<< end: get kernel id
    for tgt in tgt_list:
        tiling_op = Conv2dTiling(tiling_dict.copy(), ConvParam.dynamic_para, res_out)
        seletor = TilingSelection(tiling_op, cnt)
        tiling_cases += seletor.calc_tiling(tgt, var_names)
        cnt = next(seletor.seed_cnt)
        # >>> start: gather compile_info process
        if fuzz_build:
            tgt_nhw = []
            for var_name in var_names:
                tgt_nhw.extend(tgt[var_name])
            current_info = get_compile_info().copy()
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
                    need_update = isinstance(total_info.get(key), dict) and isinstance(value, dict)
                    if need_update:
                        new_item = total_info[key]
                        new_item.update(value)
                        total_info[key] = new_item
                        add_compile_info(key, total_info[key])
            else:
                total_info = current_info
                # <<< end: add new dict info
        # <<< end: gather compile_info process
    return tiling_cases


class Conv2dTiling(CubeTilingOp):
    def __init__(self, tiling_info, dynamic_para, res):
        super().__init__(tiling_info, None, dynamic_para.get("var_map"))
        self.a_info = tiling_info['a_shape']
        self.a_5hd_info = tiling_info['placeholder_fmap_5hd_shape']
        self.b_info = tiling_info['b_shape']
        self.c_info = tiling_info['c_shape']
        self._get_calc_info()
        self.key = 'A_shape'
        self.op_type = "conv2d"
        self.w_type = tiling_info.get("b_dtype")
        self._quant_fusion_muti_groups_in_cl0 = False
        self._l1_fusion_type = ConvParam.fusion_para.get("l1_fusion_type")
        self._input_memory_type = ConvParam.fusion_para.get("input_memory_type")
        if ConvParam.para_dict["cout1_opt"] % 2 == 1 and ConvParam.para_dict["group_opt"] > 1 and \
           ("virtual_res" in res.op.name or res.dtype == "int8"):
            self._quant_fusion_muti_groups_in_cl0 = True
        self._tiling_type = TILING_REPO_MODE

    def get_repo_tiling(self):
        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling in tiling_list:
            t_h, t_w = self.get_output_h(tiling["A_shape"][2]), \
                self.get_output_w(tiling["A_shape"][3])
            if t_h == tiling["C_shape"][2] and t_w == tiling["C_shape"][3]:
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

        if "batch_n" in self.var_map:
            self.a_info[0] = shape if isinstance(shape, int) else shape[0]
            self.c_info[0] = shape if isinstance(shape, int) else shape[0]
        if "fmap_h" in self.var_map:
            self.a_info[2] = shape[1]
            self.c_info[2] = self.get_output_h(self.a_info[2])
        if "fmap_w" in self.var_map:
            self.a_info[3] = shape[2]
            self.c_info[3] = self.get_output_w(self.a_info[3])
        if self.pad_mode == "VAR":
            if self.op_type == "conv2d":
                self.tiling_info["pad"] = self._calc_pads(shape[1], shape[2])
            else:
                self.tiling_info["pad"] = self._calc_pads(shape[0], shape[1])
        self.tiling_info["tiling_type"] = "cost_model_tiling"
        self._tiling_type = TILING_COST_MODE

        tiling = get_tiling(self.tiling_info)[0]
        return tiling

    def get_default_tiling(self, w_lower_bound=1):
        """
        get default tiling

        Parameters
        ----------
        w_lower_bound: the min value of w when dynamic w

        Returns
        -------
        default tiling
        """

        def _handle_block_dim():
            """
            handle block_dim
            """
            tiling["block_dim"] = [1, 1, 1, 1]
            device_core_num = get_soc_spec("CORE_NUM")
            if (self.a_info[0] > 1) and (device_core_num > 1):
                if self.a_info[0] <= device_core_num:
                    tiling["block_dim"][0] = self.a_info[0]
                else:
                    for core_num in range(device_core_num, 0, -1):
                        if self.a_info[0] % core_num == 0:
                            break
                    tiling["block_dim"][0] = core_num
            else:
                tiling["block_dim"][0] = 1

        tiling = {}

        if self._quant_fusion_muti_groups_in_cl0:
            tiling_n = ConvParam.para_dict["cout1_opt"]
            group_cl0 = 2
        else:
            tiling_n = 2
            group_cl0 = 1
        tiling_m = 1
        tiling_k = 1

        reduce_k0 = CUBE_INFO["reduce_k0"][self.w_type]
        tiling["AL1_shape"] = [self.b_info[2] * self.b_info[3] * reduce_k0, 1, 1, 1]
        tiling["BL1_shape"] = None
        tiling["AL0_matrix"] = [tiling_m, tiling_k, utils.FP16_M, reduce_k0, 1, 1]
        tiling["BL0_matrix"] = [tiling_k, tiling_n, utils.FP16_N, reduce_k0, 1, 1]
        tiling["CL0_matrix"] = [tiling_n, tiling_m, utils.FP16_M, utils.FP16_N, 1, group_cl0]
        tiling["CUB_matrix"] = [tiling_n, tiling_m, utils.FP16_M, utils.FP16_N, 1, 1]
        tiling["AUB_shape"] = tiling["AL1_shape"]
        tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                            'BL1_pbuffer': 1,
                                            'AL0_pbuffer': 1,
                                            'BL0_pbuffer': 1,
                                            'CL0_pbuffer': 1,
                                            'CUB_pbuffer': 1,
                                            'UBG_pbuffer': 1,
                                            'AUB_pbuffer': 1}
        tiling["A_overhead_opt_flag"] = False
        tiling["B_overhead_opt_flag"] = False
        tiling["CUB_channel_wise_flag"] = True
        tiling["n_bef_batch_flag"] = False
        _handle_block_dim()
        if self._l1_fusion_type == 1:
            tiling["AL1_shape"] = []
        if self._input_memory_type[0] == 1:
            tiling["AL1_shape"] = []

        return tiling

    def get_tiling_range(self, tiling_in, a_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        a_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """
        tiling = self._preprocess_tiling(tiling_in)
        _, _, fmap_h, fmap_w, _ = a_shape

        paras = {
            "var_map": self.var_map,
            "k_h": self.k_h_dilation,
            "k_w": self.k_w_dilation,
            "pad_mode": self.pad_mode,
            "pads": self.cur_pads
        }
        n_range_min, n_range_max = self.get_batch_range(a_shape[0], paras)
        tiling_range = [n_range_min, n_range_max]
        # check tiling covering itself situation
        if not self.check_tiling_match(tiling, fmap_w, fmap_h) or fmap_h > utils.NHW_MAX or fmap_w > utils.NHW_MAX:
            if self._tiling_type == TILING_COST_MODE:
                raise RuntimeError("current cost tiling exceed L1_Buffer, input_shape is {}".format(a_shape),
                                   "tiling is {}".format(tiling))
            return tiling_range + [0, 0, 0, 0]
        h_range_min, h_range_max = self.get_h_range(fmap_h, tiling, paras)
        tiling_range += [h_range_min, h_range_max]
        w_range_min, w_range_max = self.get_w_range(fmap_h, fmap_w, tiling, paras)
        tiling_range += [w_range_min, w_range_max]

        if not tiling.get("AL1_shape"):
            return tiling_range

        h_o = self.get_output_h(fmap_h)
        w_o = self.get_output_w(fmap_w)

        # modify range for curv performance line
        if utils.icd(utils.icd(utils.icd(h_o * w_o, tiling["block_dim"][2]), utils.FP16_M),
                     tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]:
            range_max = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * utils.FP16_M * tiling["block_dim"][2]
            perf_ho = self.get_output_h(h_range_max)
            perf_wo = self.get_output_w(w_range_max)
            if perf_ho * perf_wo > range_max:
                range_inc = int((math.sqrt((h_o + w_o) ** 2 - 4 * (h_o * w_o - range_max)) - (h_o + w_o)) / 2)
                perf_ho_max = h_o + range_inc
                perf_wo_max = w_o + range_inc
                perf_hi_max_rev = self._get_input_h(perf_ho_max)
                perf_wi_max_rev = self._get_input_w(perf_wo_max)
                perf_hi_max = min(h_range_max, perf_hi_max_rev)
                perf_wi_max = min(w_range_max, perf_wi_max_rev)
                tiling_range[3], tiling_range[5] = perf_hi_max, perf_wi_max

        return tiling_range

    def assembly_case(self, tiling, coverage, cnt):
        var_range = OrderedDict()
        if self.var_map.get("batch_n") is not None:
            var_range['batch_n'] = (coverage[0], coverage[1])

        if self.var_map.get("fmap_h") is not None:
            var_range['fmap_h'] = (coverage[2], coverage[3])
            var_range['ho'] = (self.get_output_h(var_range['fmap_h'][0]),
                               self.get_output_h(var_range['fmap_h'][1]))

        if self.var_map.get("fmap_w") is not None:
            var_range['fmap_w'] = (coverage[4], coverage[5])
            var_range['wo'] = (self.get_output_w(var_range['fmap_w'][0]),
                               self.get_output_w(var_range['fmap_w'][1]))
        correct_range_flag = ConvParam.dynamic_para.get("correct_range_flag", False)

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range,
                "correct_range_flag": correct_range_flag}

    def _get_al1_bound(self, tiling, curent_w, curent_h):
        """
        get al1 bound info

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        int, al1_load_length (al1_bound) in M axis

        """
        # shape info
        h_i, w_i = curent_h, curent_w
        out_w = self.get_output_w(w_i)

        zero_padding = False
        if self.pad_mode == "VAR":
            pad_h = utils.align(h_i, self.stride_h) - self.stride_h + self.k_h_dilation - h_i
            pad_w = utils.align(w_i, self.stride_w) - self.stride_w + self.k_w_dilation - w_i
            zero_padding = pad_h <= 0 and pad_w <= 0
        else:
            zero_padding = sum(self.cur_pads) == 0
        strideh_opti_flag = self.k_h == 1 and self.stride_h > 1 and zero_padding

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)

        # M axis theorically loading length in al1
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * tiling['AL1_shape'][1]

        # load2d instructions refer to data_mov with raw lens
        if (self.pad_mode == "VAR" or sum(self.cur_pads) == 0) \
            and (self.stride_h * self.stride_w == 1) \
                and (self.k_h * self.k_w == 1) and self.w_type == "float16":
            al1_m_data = min(al1_m_data, h_i * w_i)
            return al1_m_data

        # load3d instructions refer to load extra lines with pad/stride/filter
        if al1_m_data % out_w == 0:
            # full line could load without extra lines
            extend_h = 0
        elif (al1_m_data * 2) % out_w == 0:
            # every 2 load3d covered only 1 extra line
            extend_h = 1
        else:
            # other situations need 2 extra lines in case
            extend_h = 2
        # consistent with tiling processing logic
        l1_ho = min((al1_m_data // out_w + extend_h), self.get_output_h(h_i))

        # calculate input lines (hi) from output lines (ho)
        if not strideh_opti_flag:
            li_hi = min(self.k_h + (l1_ho - 1) * self.stride_h, h_i)
        else:
            li_hi = self.k_h + (l1_ho - 1)

        return li_hi * w_i

    def check_tiling_match(self, tiling, current_w, current_h):
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
        if not tiling.get("AL1_shape"):
            return True

        # get M axis length in al1
        al1_bound = self._get_al1_bound(tiling, current_w, current_h)

        # fmap size in L1 ( M * K * db * 2byte)
        type_size = CUBE_INFO["type_size"][self.w_type]
        reduce_k0 = CUBE_INFO["reduce_k0"][self.w_type]
        fmap_l1_size = type_size * al1_bound * tiling['AL1_shape'][0] * \
            reduce_k0 * tiling['manual_pingpong_buffer']['AL1_pbuffer']

        # filter size
        if tiling['BL1_shape'] is None:
            # not using BL1
            filter_l1_size = 0
        elif len(tiling['BL1_shape']) == 0:
            # fully load in BL1
            filter_l1_size = type_size * self.k_cout * self.k_cin * self.k_h * \
                self.k_w / tiling['block_dim'][1]
        else:
            # fmap size in L1 ( K * N * db * 2byte)
            filter_l1_size = type_size * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                reduce_k0 * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']
        if ConvParam.conv1d_split_w_flag:
            return int(filter_l1_size) <= utils.L1BUFFER
        return int(fmap_l1_size) + int(filter_l1_size) <= utils.L1BUFFER

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.a_5hd_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.b_info[2:4]
        self.k_cin = self.b_info[1] * self.b_info[4]
        self.k_cout = self.b_info[0]
        self.stride_h, self.stride_w = self.tiling_info["stride"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilation"]

        self.pad_mode = "FIX"
        # currently, in dynamic_hw, when padding is SAME, pad_mode is "VAR"
        if isinstance(self.tiling_info["pad"][0], Expr) or isinstance(self.tiling_info["pad"][1], Expr) or \
            isinstance(self.tiling_info["pad"][2], Expr) or isinstance(self.tiling_info["pad"][3], Expr):
            self.pad_mode = "VAR"
            self.tiling_info["pad"] = [-1, -1, -1, -1]
        self.cur_pads = self.tiling_info["pad"]

        self.k_h_dilation = (self.k_h - 1) * self.dilate_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilate_w + 1

    def _preprocess_tiling(self, tiling_in):
        """
        preprocess tiling for get tiling range
        """

        tiling = copy.deepcopy(tiling_in)
        if tiling["AL1_shape"]:
            tiling["AL1_shape"][0] = tiling["AL1_shape"][0] // \
                (((self.k_h - 1)*ConvParam.dilate_h + 1)*((self.k_w - 1)*ConvParam.dilate_w + 1)*
                 CUBE_INFO["reduce_k0"][self.w_type])
        if tiling["BL1_shape"]:
            tiling["BL1_shape"][0] = tiling["BL1_shape"][0] // \
                (self.k_h * self.k_w * CUBE_INFO["reduce_k0"][self.w_type])
        return tiling

    def get_output_h(self, h_in):
        """
        calculate output h
        """

        if not h_in:
            return h_in
        if self.pad_mode == "VAR":
            return utils.icd(h_in, self.stride_h)
        return (h_in + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) // self.stride_h + 1

    def get_output_w(self, w_in):
        """
        calculate output w
        """

        if not w_in:
            return w_in
        if self.pad_mode == "VAR":
            return utils.icd(w_in, self.stride_w)
        return (w_in + self.cur_pads[0] + self.cur_pads[1] - self.dilate_w *
                (self.k_w - 1) - 1) // self.stride_w + 1

    def _get_input_h(self, h_out):
        """
        calculate max input h
        """

        if self.pad_mode == "VAR":
            return h_out * self.stride_h
        return h_out * self.stride_h - self.cur_pads[2] - self.cur_pads[3] \
            + self.dilate_h * (self.k_h - 1)

    def _get_input_w(self, w_out):
        """
        calculate max input w
        """

        if self.pad_mode == "VAR":
            return w_out * self.stride_w
        return w_out * self.stride_w - self.cur_pads[0] - self.cur_pads[1] \
            + self.dilate_w * (self.k_w - 1)

    def _calc_pads(self, h_in, w_in):
        """
        calculate pads
        """

        pad_h = utils.align(h_in, self.stride_h) - self.stride_h + \
            self.k_h_dilation - h_in
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = utils.align(w_in, self.stride_w) - self.stride_w + \
            self.k_w_dilation - w_in
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return [pad_left, pad_right, pad_up, pad_down]
