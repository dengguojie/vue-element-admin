#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
extract image patches schedule
"""

import math
from tbe import tvm
from tbe.common import platform as tbe_platform
import te.platform as te_platform
from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info

from . import util
from .constants import Pattern


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_INT8 = 32

    DOUBLE_BUFFER = 2
    FP16_SIZE = 2
    INT8_SIZE = 1
    NEED_UB_SPACE_NUM = 1
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    SIZE_UB = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    LOAD3D_REPEAT_TIME_LIMIT = 255
    DELTA = 0.000001  # aviod div zero, fp32 precision


def _ceil_div(value, block):
    """
    integrate the input value by block
    """
    return (value + block - 1) // block


def _prod(val_list):
    """
    calculate product of val_list
    """
    res = 1
    for val in val_list:
        res = res * val
    return res


@register_schedule(pattern=Pattern.EXTRACT_IMAGE_PATCHES)
def schedule(outs, tiling_case):
    """
    schedule for extract_image_patch dynamic shape
    """
    return ExtractImagePatchesSchedule(outs, tiling_case).do_schedule()


class ExtractImagePatchesSchedule:
    """
    ExtractImagePatchesSchedule
    """
    def __init__(self, outs, tiling_case):
        self.output_res = outs[0]
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _get_tiling_param_cut_howo_col(used_ub_size, lcm_out_w, khkw, cut_h_col, fmap_w, fmap_c0, type_size,
                                       c_in_real, align_block_size):
        """
        get params for tiling
        """
        # cut howo col
        max_vm_ub = (used_ub_size // align_block_size // lcm_out_w + khkw - 1) // (khkw + 1)
        if max_vm_ub > Constant.LOAD3D_REPEAT_TIME_LIMIT:
            max_vm_ub = Constant.LOAD3D_REPEAT_TIME_LIMIT
        max_vm_l1 = Constant.SIZE_L1 // (cut_h_col * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER)
        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1
        if max_vm_ub > 1:
            while c_in_real % max_vm_ub != 0:
                max_vm_ub = max_vm_ub - 1
        # cut howo col, move_rate
        # move_rate limit according to mte2 bound
        move_rate = 1 / khkw
        return max_vm_ub, move_rate

    # 'pylint: disable=too-many-locals,too-many-arguments
    @staticmethod
    def _get_tiling_param_cut_howo_row(khkw, fmap_w, fmap_c0, dilated_kernel_h, dilated_kernel_w, stride_h,
                                       type_size, avg_split_ub_size, cut_w_row, cut_h_row, c_in_real,
                                       align_block_size):
        # cut howo row
        max_vm_ub = avg_split_ub_size // align_block_size // Constant.BLOCK_SIZE // khkw
        max_vm_load3d_limit = Constant.LOAD3D_REPEAT_TIME_LIMIT // khkw
        if max_vm_ub > max_vm_load3d_limit:
            max_vm_ub = max_vm_load3d_limit
        max_vm_l1 = Constant.SIZE_L1 // (cut_h_row * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER)
        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1
        if max_vm_ub > 1:
            while c_in_real % max_vm_ub != 0:
                max_vm_ub = max_vm_ub - 1

        # cut howo row, move_rate
        # move_rate useful move rate while mte2 data move
        double_loaded = dilated_kernel_h // 2 - stride_h
        if double_loaded < 0:
            double_loaded = 0
        slide_dis_h = cut_h_row - dilated_kernel_h + 1
        slide_times_h = slide_dis_h // stride_h + 1
        slide_dis_w = cut_w_row - dilated_kernel_w + 1
        move_rate = slide_dis_w / (slide_times_h * fmap_w) * (1 - double_loaded / cut_h_row)
        return max_vm_ub, move_rate

    # 'pylint: disable=too-many-arguments
    @staticmethod
    def _get_tiling_param_cut_howo_partial_col(out_w, khkw, fmap_w, stride_h, type_size, avg_split_ub_size,
                                               cut_h_row, c_in_real, align_block_size, dilated_kernel_h):
        """
        The function is get tiling param cut howo partial col.
        """
        # cut howo col partially
        c_in_align = _ceil_div(c_in_real, align_block_size) * align_block_size
        max_vm_ub = avg_split_ub_size // (khkw * c_in_align * align_block_size)
        max_vm_load3d_limit = Constant.LOAD3D_REPEAT_TIME_LIMIT // khkw
        if max_vm_ub > max_vm_load3d_limit:
            max_vm_ub = 0

        w_size = fmap_w * c_in_align * type_size * Constant.DOUBLE_BUFFER
        max_vm_l1 = Constant.SIZE_L1 // (dilated_kernel_h * w_size)
        if Constant.SIZE_L1 < (_ceil_div(max_vm_l1 * Constant.BLOCK_SIZE, out_w) + 1) * stride_h * w_size \
                or cut_h_row > stride_h + dilated_kernel_h - 1:
            max_vm_l1 = Constant.SIZE_L1 // (cut_h_row * w_size)

        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1
        cut_hw_up_w = (max_vm_ub * align_block_size + out_w - 1) // out_w * out_w

        # cut howo col partially, move_rate
        # move_rate useful move rate while mte2 data move
        move_rate = max_vm_ub * align_block_size / (cut_hw_up_w + Constant.DELTA)
        return max_vm_ub, move_rate

    @staticmethod
    def _get_tiling_param_cut_howo_min(fmap_w, fmap_c0, type_size, avg_split_ub_size, cut_h_row,
                                       align_block_size):
        # cut howo khkw c, minimum cut
        max_vm_ub = avg_split_ub_size // (1 * align_block_size * Constant.BLOCK_SIZE)
        if max_vm_ub > Constant.LOAD3D_REPEAT_TIME_LIMIT:
            max_vm_ub = Constant.LOAD3D_REPEAT_TIME_LIMIT
        max_vm_l1 = Constant.SIZE_L1 // (cut_h_row * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER)
        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1

        return max_vm_ub

    # 'pylint: disable=too-many-arguments
    def _get_tiling_param(self, setfmatrix_dict, extract_params, used_ub_size, type_size, avg_split_ub_size,
                          align_block_size):
        out_w = extract_params['out_w']
        fmap_shape = extract_params['fmap_shape']
        c_in_real = extract_params["c_in_real"]
        lcm_out_w = extract_params['lcm_out_w']
        cut_h_col = extract_params['cut_h_col']
        cut_w_row = extract_params['cut_w_row']
        cut_h_row = extract_params['cut_h_row']
        dilated_kernel_h = extract_params['dilated_kernel_h']
        dilated_kernel_w = extract_params['dilated_kernel_w']
        fmap_w = fmap_shape[3].value
        fmap_c0 = fmap_shape[4].value
        kernel_h = setfmatrix_dict['conv_kernel_h']
        kernel_w = setfmatrix_dict['conv_kernel_w']
        stride_h = setfmatrix_dict['conv_stride_h']
        khkw = kernel_h * kernel_w

        max_vm_cut_col, move_rate_cut_col = self._get_tiling_param_cut_howo_col(used_ub_size, lcm_out_w, khkw,
                                                                                cut_h_col, fmap_w, fmap_c0,
                                                                                type_size, c_in_real,
                                                                                align_block_size)

        max_vm_cut_row, move_rate_cut_row = \
            self._get_tiling_param_cut_howo_row(khkw, fmap_w, fmap_c0, dilated_kernel_h, dilated_kernel_w, stride_h,
                                                type_size, avg_split_ub_size, cut_w_row, cut_h_row, c_in_real,
                                                align_block_size)

        max_vm_cut_col_p, move_rate_cut_col_p = \
            self._get_tiling_param_cut_howo_partial_col(out_w, khkw, fmap_w, stride_h, type_size, avg_split_ub_size,
                                                        cut_h_row, c_in_real, align_block_size, dilated_kernel_h)

        max_vm_cut_min = self._get_tiling_param_cut_howo_min(fmap_w, fmap_c0, type_size, avg_split_ub_size, cut_h_row,
                                                             align_block_size)
        return [max_vm_cut_col, max_vm_cut_row, max_vm_cut_col_p, max_vm_cut_min, move_rate_cut_col, move_rate_cut_row,
                move_rate_cut_col_p]

    def _cal_multi_core_factor(self, m, n, m_list, n_list):
        """
        Return the cut factors for multicore axis.
        """

        m_list = list(set(m_list))
        n_list = list(set(n_list))

        m_list.sort(reverse=True)
        n_list.sort(reverse=True)

        min_cycle_num = m * n
        core_m, core_n = m_list[-1], n_list[-1]

        for i in m_list:
            for j in n_list:
                if i * j > self.device_core_num:
                    continue
                tmp_cycle_num = _ceil_div(m, i) * _ceil_div(n, j)
                if tmp_cycle_num < min_cycle_num:
                    min_cycle_num = tmp_cycle_num
                    core_m, core_n = i, j
                break
        return core_m, core_n

    def _cal_multi_core_factor_3d(self, m, n, l, m_list, n_list, l_list):
        """
        Return the cut factors for multicore axis.
        """
        m_list = list(set(m_list))
        n_list = list(set(n_list))
        l_list = list(set(l_list))

        m_list.sort(reverse=True)
        n_list.sort(reverse=True)
        l_list.sort(reverse=True)

        min_cycle_num = m * n * l
        core_m, core_n, core_l = m_list[-1], n_list[-1], l_list[-1]

        for i in m_list:
            for j in n_list:
                if i * j > self.device_core_num:
                    continue
                for k in l_list:
                    if i * j * k > self.device_core_num:
                        continue
                    tmp_cycle_num = _ceil_div(m, i) * _ceil_div(n, j) * _ceil_div(l, k)
                    if tmp_cycle_num < min_cycle_num:
                        min_cycle_num = tmp_cycle_num
                        core_m, core_n, core_l = i, j, k
                    break
        return core_m, core_n, core_l

    @staticmethod
    def _get_dma_split_factor(dma_split_axis_id, out_shape, out_w, kernel_w, align_block_size):
        """
        get split factor
        """
        split_eles = _prod(out_shape[dma_split_axis_id:])
        ele_len = _prod(out_shape[dma_split_axis_id + 1:])

        def _could_split_multi_core(val):
            if val * ele_len > 9069:
                return False
            tail_len = split_eles % (val * ele_len)
            return (tail_len > align_block_size) or (val * ele_len > align_block_size and tail_len == 0)

        if _could_split_multi_core(out_shape[dma_split_axis_id]):
            return out_shape[dma_split_axis_id], True, True

        if dma_split_axis_id == 1 and _could_split_multi_core(out_w):  # howo
            return out_w, True, True

        if dma_split_axis_id == 2 and _could_split_multi_core(kernel_w):  # khkw
            return kernel_w, True, True

        for val in range(align_block_size, out_shape[dma_split_axis_id], align_block_size):
            if _could_split_multi_core(val):
                return val, (out_shape[dma_split_axis_id] % val == 0), True

        return 1, False, False

    def _split_multi_core_32b_not_aligned(self, sch, multi_core_factor, dma_split_axis_id, dma_split_factor,
                                          workspace_res):
        """
        split multi core, when 32B is not aligned
        """
        res_axis_list = list(self.output_res.op.axis).copy()
        workspace_axis_list = list(workspace_res.op.axis).copy()

        res_bind_axis_list = [0 for _ in range(dma_split_axis_id)]
        workspace_bind_axis_list = [0 for _ in range(dma_split_axis_id)]
        for i in range(dma_split_axis_id):
            workspace_bind_axis_list[i], workspace_axis_list[i] = sch[workspace_res].split(workspace_axis_list[i],
                                                                                        factor=multi_core_factor[
                                                                                        i])
            res_bind_axis_list[i], res_axis_list[i] = sch[self.output_res].split(res_axis_list[i],
                                                                                    factor=multi_core_factor[i])
        # 32B not align data copy
        res_axis_list[dma_split_axis_id], dma_copy_axis = sch[self.output_res].split(
            res_axis_list[dma_split_axis_id],
            factor=dma_split_factor)

        sch[self.output_res].reorder(*(res_bind_axis_list + res_axis_list[:dma_split_axis_id] +
                                        [dma_copy_axis] + res_axis_list[dma_split_axis_id + 1:]))
        sch[workspace_res].reorder(*(workspace_bind_axis_list + workspace_axis_list))

        res_bind_axis = sch[self.output_res].fuse(*(res_bind_axis_list))
        workspace_bind_axis = sch[workspace_res].fuse(*(workspace_bind_axis_list))

        return [[res_bind_axis], res_axis_list, [workspace_bind_axis], workspace_axis_list, dma_copy_axis]

    @staticmethod
    def _add_compile_info(avg_split_ub_size, res_ub_num, align_block_size, workspace_output, workspace_filter,
                          workspace_c):
        add_compile_info("ubSize", Constant.SIZE_UB)
        add_compile_info("coreNum", util.get_core_num())
        add_compile_info("avgSplitUbSize", avg_split_ub_size)
        add_compile_info("resUbNum", res_ub_num)
        add_compile_info("alignBlockSize", align_block_size)
        add_compile_info("workspaceOutput", int(workspace_output))
        add_compile_info("workspaceFilter", int(workspace_filter))
        add_compile_info("workspaceC", int(workspace_c))

    # 'pylint: disable=too-many-statements,too-many-branches,too-many-locals,too-many-lines
    def do_schedule(self):
        """
        :param res: the multi-results in the operator
        :param sch: schedule list
        """
        sch = tvm.create_schedule(self.output_res.op)
        sch.tiling_key = self._tiling_key
        compile_info = get_compile_info()
        multi_core_factor_0 = operation.get_context().get("multi_core_factor_0")

        setfmatrix_map = self.output_res.op.attrs['setfmatrix_dict']
        setfmatrix_dict = {}
        for key, value in setfmatrix_map.items():
            if hasattr(value, "value"):
                setfmatrix_dict[key] = value.value
            else:
                setfmatrix_dict[key] = value

        extract_map = self.output_res.op.attrs['extract_params']
        extract_params = {}
        for key, value in extract_map.items():
            if hasattr(value, "value"):
                extract_params[key] = value.value
            else:
                extract_params[key] = value

        out_h = extract_params.get('out_h')
        out_w = extract_params.get('out_w')
        fmap_shape = extract_params.get('fmap_shape')
        c_in_real = extract_params.get("c_in_real")
        fmap_n = fmap_shape[0]
        sch.set_var_range(fmap_n, 2, 513)
        sch.set_var_range(multi_core_factor_0, 3, 20)
        fmap_c1 = fmap_shape[1].value
        fmap_h = fmap_shape[2].value
        fmap_w = fmap_shape[3].value
        fmap_c0 = fmap_shape[4].value
        kernel_h = setfmatrix_dict.get('conv_kernel_h')
        kernel_w = setfmatrix_dict.get('conv_kernel_w')
        dilate_h = setfmatrix_dict.get('conv_dilation_h')
        dilate_w = setfmatrix_dict.get('conv_dilation_w')
        stride_h = setfmatrix_dict.get('conv_stride_h')
        stride_w = setfmatrix_dict.get('conv_stride_w')

        ub_res = self.output_res.op.input_tensors[0]
        workspace_res = ub_res.op.input_tensors[0]
        merge_co_ub = workspace_res.op.input_tensors[0]
        merge_hw_ub = merge_co_ub.op.input_tensors[0]
        transpose_ub = merge_hw_ub.op.input_tensors[0]
        split_c1_ub = transpose_ub.op.input_tensors[0]
        fmap_fractal = split_c1_ub.op.input_tensors[0]
        fmap_im2col = fmap_fractal.op.input_tensors[0]
        fmap_in_l1 = fmap_im2col.op.input_tensors[0]

        sch[fmap_in_l1].set_scope(tbe_platform.scope_cbuf)
        sch[fmap_im2col].set_scope(tbe_platform.scope_cbuf)
        sch[fmap_fractal].set_scope(tbe_platform.scope_ubuf)
        sch[split_c1_ub].set_scope(tbe_platform.scope_ubuf)
        sch[transpose_ub].set_scope(tbe_platform.scope_ubuf)
        sch[merge_hw_ub].set_scope(tbe_platform.scope_ubuf)
        sch[merge_co_ub].set_scope(tbe_platform.scope_ubuf)
        sch[workspace_res].set_scope(tbe_platform.scope_gm)
        sch[ub_res].set_scope(tbe_platform.scope_ubuf)

        workspace_output = workspace_res.shape[1]
        workspace_filter = workspace_res.shape[2]
        workspace_c = workspace_res.shape[3]

        dtype_input = ub_res.dtype
        if dtype_input in ('int8', 'uint8'):
            align_block_size = Constant.BLOCK_SIZE_INT8
            type_size = Constant.INT8_SIZE
        else:
            align_block_size = Constant.BLOCK_SIZE
            type_size = Constant.FP16_SIZE

        out_hw_up16 = ((out_h * out_w - 1) // Constant.BLOCK_SIZE + 1) * Constant.BLOCK_SIZE
        dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilate_w + 1
        lcm_out_w = Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) * out_w
        cut_h_col = (Constant.BLOCK_SIZE // math.gcd(out_w,
                                                     Constant.BLOCK_SIZE) - 1) * stride_h + 1 + dilated_kernel_h // 2
        if cut_h_col > fmap_h:
            cut_h_col = fmap_h
        # `cut_h_col while cut_hw = Constant.BLOCK_SIZE`
        cut_w_row_s = (Constant.BLOCK_SIZE - 1) * stride_w + 1
        cut_h_row_s = max(stride_h, (((cut_w_row_s - 1) // fmap_w + 1) - 1) * stride_h + 1)
        cut_w_row = cut_w_row_s + dilated_kernel_w - 1
        cut_h_row = cut_h_row_s + dilated_kernel_h - 1
        if lcm_out_w > out_hw_up16:
            lcm_out_w = out_hw_up16

        extract_params['lcm_out_w'] = lcm_out_w
        extract_params['cut_h_col'] = cut_h_col
        extract_params['cut_w_row'] = cut_w_row
        extract_params['cut_h_row'] = cut_h_row
        extract_params['dilated_kernel_h'] = dilated_kernel_h
        extract_params['dilated_kernel_w'] = dilated_kernel_w

        sch[ub_res].buffer_align((1, 1), (1, 1), (1, 1), (1, align_block_size))
        sch[fmap_im2col].buffer_align((1, 1), (out_w, out_w), (1, 1), (1, 1), (1, 1), (1, align_block_size))
        sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE), (1, align_block_size))

        used_ub_size = Constant.SIZE_UB // type_size // Constant.DOUBLE_BUFFER
        avg_split_ub_size = used_ub_size // Constant.NEED_UB_SPACE_NUM
        howo = out_h * out_w
        khkw = kernel_h * kernel_w
        c_out = khkw * fmap_c1 * fmap_c0

        out_shape = [fmap_n, howo, khkw, c_in_real]
        self.device_core_num = util.get_core_num()

        def _get_multi_core_factor(dma_split_axis_id, tiling_factor):
            """
            get multi core split factor
            """
            multi_core_factor = out_shape.copy()
            if dma_split_axis_id == 0:
                return multi_core_factor
            if dma_split_axis_id == 1:
                multi_core_factor[0] = multi_core_factor_0
                return multi_core_factor

            if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                howo_align = Constant.BLOCK_SIZE
            elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                howo_align = lcm_out_w
            else:
                howo_align = howo

            def _get_core_factor(multi_core_factor, core_n, core_howo):
                multi_core_factor[0] = max(_ceil_div(128, core_n), tiling_factor[0])
                multi_core_factor[1] = _ceil_div(max(_ceil_div(out_shape[1], core_howo), tiling_factor[1]),
                                                    howo_align) * howo_align
                return multi_core_factor

            pre_core_n, pre_core_howo = [1], [1]
            for i in range(1, self.device_core_num + 1):
                multi_core_factor = _get_core_factor(out_shape.copy(), i, i)
                pre_core_n.append(_ceil_div(128, multi_core_factor[0]))
                pre_core_howo.append(_ceil_div(out_shape[1], multi_core_factor[1]))

            core_n, core_howo = self._cal_multi_core_factor(_ceil_div(128, tiling_factor[0]),
                                                            _ceil_div(out_shape[1], tiling_factor[1]),
                                                            pre_core_n, pre_core_howo)
            multi_core_factor = _get_core_factor(out_shape.copy(), core_n, core_howo)

            return multi_core_factor

        def _split_multi_core_32b_align(tiling_factor):
            """
            split multi core, when 32B is aligned
            """
            if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                howo_align = Constant.BLOCK_SIZE
            else:
                howo_align = lcm_out_w

            def _get_core_factor(multi_core_factor, core_n, core_howo, core_c):
                multi_core_factor[0] = _ceil_div(128, core_n)
                multi_core_factor[1] = _ceil_div(max(_ceil_div(out_shape[1], core_howo), tiling_factor[1]),
                                                    howo_align) * howo_align
                multi_core_factor[3] = _ceil_div(max(_ceil_div(out_shape[3], core_c), tiling_factor[3]),
                                                    align_block_size) * align_block_size
                return multi_core_factor

            pre_core_n, pre_core_c, pre_core_howo = [1], [1], [1]
            for i in range(1, self.device_core_num + 1):
                multi_core_factor = _get_core_factor(out_shape.copy(), i, i, i)
                pre_core_n.append(_ceil_div(128, multi_core_factor[0]))
                pre_core_howo.append(_ceil_div(out_shape[1], multi_core_factor[1]))
                pre_core_c.append(_ceil_div(out_shape[3], multi_core_factor[3]))

            core_n, core_c, core_howo = self._cal_multi_core_factor_3d(_ceil_div(128, tiling_factor[0]),
                                                                    _ceil_div(out_shape[3], tiling_factor[3]),
                                                                    _ceil_div(out_shape[1], tiling_factor[1]),
                                                                    pre_core_n, pre_core_c, pre_core_howo)
            multi_core_factor = _get_core_factor(out_shape.copy(), core_n, core_howo, core_c)

            res_axis_list = list(self.output_res.op.axis).copy()
            res_bind_axis_list = [0 for _ in res_axis_list]
            for i, _ in enumerate(res_bind_axis_list):
                res_bind_axis_list[i], res_axis_list[i] = sch[self.output_res].split(res_axis_list[i],
                                                                                     factor=multi_core_factor[i])
            sch[self.output_res].reorder(*(res_bind_axis_list + res_axis_list))
            res_bind_axis = sch[self.output_res].fuse(*(res_bind_axis_list))

            return [res_bind_axis], res_axis_list

        # 'pylint: disable=too-many-branches,too-many-lines
        def _schedule_32b_not_aligned(dma_split_axis_id, dma_split_factor, allow_multi_core, reg_mov=True):
            """
            schedule, when 32B is not aligned
            """
            n_factor = 1
            howo_factor = howo
            khkw_factor = khkw
            c_factor = c_in_real
            tiling_param = self._get_tiling_param(setfmatrix_dict, extract_params, used_ub_size,
                                                  type_size, avg_split_ub_size, align_block_size)

            max_vm_cut_col, max_vm_cut_row, max_vm_cut_col_p, _, move_rate_cut_col, move_rate_cut_row, \
            move_rate_cut_col_p = tiling_param
            move_rate = 0
            if max_vm_cut_col > 0:
                move_rate = move_rate_cut_col
            if move_rate < move_rate_cut_row and max_vm_cut_row > 0:
                move_rate = move_rate_cut_row
            if move_rate < move_rate_cut_col_p and max_vm_cut_col_p > 0:
                move_rate = move_rate_cut_col_p

            if lcm_out_w * c_out <= avg_split_ub_size and khkw * fmap_c1 <= Constant.LOAD3D_REPEAT_TIME_LIMIT \
                    and max_vm_cut_col > 0 and max_vm_cut_row > 0 \
                    and Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                max_v = avg_split_ub_size // lcm_out_w // c_out
                if lcm_out_w * max_v < howo:
                    # if True cut n howo else only cut n
                    howo_factor = lcm_out_w * max_v
            elif move_rate == move_rate_cut_col and max_vm_cut_col > 0:
                # cut howo col
                howo_factor = lcm_out_w
                khkw_factor = 1
                c_factor = align_block_size
            elif move_rate == move_rate_cut_row and max_vm_cut_row > 0:
                # cut howo row
                howo_factor = Constant.BLOCK_SIZE
                khkw_factor = khkw
                c_factor = align_block_size
            elif move_rate == move_rate_cut_col_p and max_vm_cut_col_p > 0:
                # cut howo col partially
                howo_factor = Constant.BLOCK_SIZE * max_vm_cut_col_p
                c_factor = c_in_real
                khkw_factor = khkw
            else:
                # cut howo khkw c
                howo_factor = Constant.BLOCK_SIZE
                khkw_factor = 1
                c_factor = align_block_size

            tiling_factor = [n_factor, howo_factor, khkw_factor, c_factor]

            if reg_mov:
                reg_mov_ub = sch.cache_write(self.output_res, tbe_platform.scope_ubuf)
            if allow_multi_core:
                multi_core_factor = _get_multi_core_factor(dma_split_axis_id, tiling_factor)
            else:
                multi_core_factor = out_shape.copy()

            split_multi_core_axis_list = self._split_multi_core_32b_not_aligned(sch, multi_core_factor,
                                                                                dma_split_axis_id,
                                                                                dma_split_factor, workspace_res)
            res_bind_list, res_axis_list, workspace_bind_list, workspace_axis_list, dma_copy_axis = \
                split_multi_core_axis_list

            workspace_res_n_outer, workspace_res_n_inner = sch[workspace_res].split(workspace_axis_list[0],
                                                                                    factor=tiling_factor[0])
            workspace_res_howo_outer, workspace_res_howo_inner = sch[workspace_res].split(workspace_axis_list[1],
                                                                                          factor=tiling_factor[1])
            workspace_res_khkw_outer, workspace_res_khkw_inner = sch[workspace_res].split(workspace_axis_list[2],
                                                                                          factor=tiling_factor[2])
            workspace_res_c1_inner_outer, workspace_res_c1_inner = sch[workspace_res].split(workspace_axis_list[3],
                                                                                            factor=align_block_size)
            workspace_res_c1_outer, workspace_res_c1_inner_outer = \
                sch[workspace_res].split(workspace_res_c1_inner_outer,
                                         factor=max(tiling_factor[3] // align_block_size, 1))

            workspace_axis_outer_list = [workspace_res_n_outer, workspace_res_howo_outer, workspace_res_khkw_outer,
                                         workspace_res_c1_outer]
            workspace_axis_inner_list = [workspace_res_n_inner, workspace_res_c1_inner_outer,
                                         workspace_res_howo_inner, workspace_res_khkw_inner, workspace_res_c1_inner]

            if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                compute_at_id = 0
            elif Constant.SIZE_L1 >= cut_h_row * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                    and move_rate != move_rate_cut_col:
                compute_at_id = 1
            elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                    and move_rate == move_rate_cut_col:
                compute_at_id = 1
            elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER \
                    and move_rate == move_rate_cut_col:
                compute_at_id = 1
                workspace_c_out_outer, workspace_axis_outer_list[3] = sch[workspace_res].split(
                    workspace_axis_outer_list[3], factor=1)
                workspace_bind_list.append(workspace_c_out_outer)
            else:
                compute_at_id = 2
                workspace_c_out_outer, workspace_axis_outer_list[3] = sch[workspace_res].split(
                    workspace_axis_outer_list[3], factor=1)
                workspace_bind_list.append(workspace_c_out_outer)
                workspace_axis_outer_list[2], _ = sch[workspace_res].split(
                    workspace_axis_outer_list[2], factor=max(kernel_w // tiling_factor[2], 1))

            sch[workspace_res].reorder(
                *(workspace_bind_list + workspace_axis_outer_list + workspace_axis_inner_list))

            sch[fmap_im2col].compute_at(sch[workspace_res], workspace_axis_outer_list[compute_at_id])
            sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_axis_outer_list[compute_at_id])

            sch[merge_co_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
            sch[merge_hw_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
            sch[transpose_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
            sch[split_c1_ub].compute_at(sch[workspace_res], workspace_axis_outer_list[3])
            sch[fmap_fractal].compute_at(sch[workspace_res], workspace_axis_outer_list[3])

            sch[ub_res].compute_at(sch[self.output_res], res_axis_list[dma_split_axis_id])
            if reg_mov:
                sch[reg_mov_ub].compute_at(sch[self.output_res], res_axis_list[dma_split_axis_id])

            block = tvm.thread_axis("blockIdx.x")
            sch[self.output_res].bind(res_bind_list[0], block)
            sch[workspace_res].bind(workspace_bind_list[0], block)

            sch[split_c1_ub].compute_inline()
            sch[merge_co_ub].compute_inline()

            sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], te_platform.DMA_COPY)
            sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0], te_platform.SET_FMATRIX, setfmatrix_dict)
            sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], te_platform.IM2COL)
            sch[split_c1_ub].emit_insn(split_c1_ub.op.axis[0], te_platform.DMA_COPY)

            if dtype_input in ("int8", "uint8"):
                sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], te_platform.DMA_COPY)
                sch[merge_hw_ub].emit_insn(merge_hw_ub.op.axis[0], te_platform.DMA_COPY)
            else:
                sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], te_platform.insn_cmd.ADDVS)
                sch[merge_hw_ub].emit_insn(merge_hw_ub.op.axis[0], te_platform.insn_cmd.ADDVS)

            sch[merge_co_ub].emit_insn(merge_co_ub.op.axis[0], te_platform.DMA_COPY)
            sch[workspace_res].emit_insn(workspace_axis_inner_list[0], te_platform.DMA_COPY)
            sch[ub_res].emit_insn(ub_res.op.axis[0], te_platform.DMA_COPY)
            if reg_mov:
                if c_in_real == 1 and dtype_input not in ('int8', 'uint8'):
                    sch[reg_mov_ub].emit_insn(reg_mov_ub.op.axis[0], te_platform.REDUCE_SUM)
                else:
                    sch[reg_mov_ub].emit_insn(reg_mov_ub.op.axis[0], te_platform.DATA_MOV)
            sch[self.output_res].emit_insn(dma_copy_axis, te_platform.DMA_PADDING)

        if c_in_real % align_block_size == 0:
            n_factor = 1
            howo_factor = howo
            khkw_factor = khkw
            c_factor = c_in_real
            max_v = fmap_c1
            tiling_param = self._get_tiling_param(setfmatrix_dict, extract_params, used_ub_size,
                                                  type_size, avg_split_ub_size, align_block_size)

            max_vm_cut_col, max_vm_cut_row, max_vm_cut_col_p, max_vm_cut_min, move_rate_cut_col, move_rate_cut_row, \
            move_rate_cut_col_p = tiling_param

            move_rate = 0
            if max_vm_cut_col > 0:
                move_rate = move_rate_cut_col
            if move_rate < move_rate_cut_row and max_vm_cut_row > 0:
                move_rate = move_rate_cut_row
            if move_rate < move_rate_cut_col_p and max_vm_cut_col_p > 0:
                move_rate = move_rate_cut_col_p
            split_khkw_mode = False
            if lcm_out_w * c_out <= avg_split_ub_size and khkw * fmap_c1 <= Constant.LOAD3D_REPEAT_TIME_LIMIT \
                    and max_vm_cut_col > 0 and max_vm_cut_row > 0 \
                    and Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                max_v = avg_split_ub_size // lcm_out_w // c_out
                if lcm_out_w * max_v < howo:
                    # if True cut n howo else only cut n
                    howo_factor = lcm_out_w * max_v
            elif move_rate == move_rate_cut_col and max_vm_cut_col > 0:
                # cut howo col
                howo_factor = lcm_out_w
                max_v = max_vm_cut_col
                khkw_factor = 1
                c_factor = align_block_size * max_v
            elif move_rate == move_rate_cut_row and max_vm_cut_row > 0:
                # cut howo row
                howo_factor = Constant.BLOCK_SIZE
                khkw_factor = khkw
                max_v = max_vm_cut_row
                c_factor = align_block_size * max_v
            elif move_rate == move_rate_cut_col_p and max_vm_cut_col_p > 0:
                # cut howo col partially
                howo_factor = Constant.BLOCK_SIZE * max_vm_cut_col_p
                c_factor = c_in_real
                khkw_factor = khkw
                max_v = fmap_c1
            else:
                # cut howo khkw c
                howo_factor = Constant.BLOCK_SIZE
                max_v = max_vm_cut_min
                if max_v == 0:
                    max_v = 1
                    split_khkw_mode = True
                # The instruction parameter is uint8 type.
                if khkw * max_v >= 256:
                    max_v = max(255 // khkw, 1)
                khkw_factor = 1
                c_factor = align_block_size * max_v

            tiling_factor = [n_factor, howo_factor, khkw_factor, c_factor]
            res_bind_list, res_axis_list = _split_multi_core_32b_align(tiling_factor)
            res_ub_num = _ceil_div(_ceil_div(c_in_real, align_block_size) * align_block_size, c_in_real) + 1
            res_n_outer, res_n_inner = sch[self.output_res].split(res_axis_list[0], factor=tiling_factor[0])
            res_howo_outer, res_howo_inner = sch[self.output_res].split(res_axis_list[1], factor=tiling_factor[1])
            res_khkw_outer, res_khkw_inner = sch[self.output_res].split(res_axis_list[2], factor=tiling_factor[2])
            res_c_inner_outer, res_c_inner = sch[self.output_res].split(res_axis_list[3], factor=align_block_size)
            res_c_outer, res_c_outer_inner = sch[self.output_res].split(res_c_inner_outer,
                                                                        factor=tiling_factor[3] // align_block_size)

            res_axis_outer_list = [res_n_outer, res_howo_outer, res_khkw_outer, res_c_outer]

            if Constant.SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER:
                compute_at_id = 0
            elif Constant.SIZE_L1 >= cut_h_row * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                    and move_rate != move_rate_cut_col:
                compute_at_id = 1
            elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * Constant.DOUBLE_BUFFER \
                    and move_rate == move_rate_cut_col:
                compute_at_id = 1
            elif Constant.SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER \
                    and move_rate == move_rate_cut_col:
                compute_at_id = 1
                res_c_out_outer, res_axis_outer_list[3] = sch[self.output_res].split(res_axis_outer_list[3], factor=1)
                res_bind_list.append(res_c_out_outer)
            elif Constant.SIZE_L1 >= cut_h_row_s * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER \
                    and split_khkw_mode:
                compute_at_id = 2
                res_axis_outer_list[2], _ = sch[self.output_res].split(res_axis_outer_list[2],
                                                                            factor=max(kernel_w // khkw_factor, 1))
                res_c_out_outer, res_axis_outer_list[3] = sch[self.output_res].split(res_axis_outer_list[3], factor=1)
                res_bind_list.append(res_c_out_outer)
            else:
                compute_at_id = 3

            sch[self.output_res].reorder(*(res_bind_list + res_axis_outer_list +
                            [res_n_inner, res_c_outer_inner, res_howo_inner, res_khkw_inner, res_c_inner]))

            sch[fmap_im2col].compute_at(sch[self.output_res], res_axis_outer_list[compute_at_id])
            sch[fmap_in_l1].compute_at(sch[self.output_res], res_axis_outer_list[compute_at_id])
            sch[transpose_ub].compute_at(sch[self.output_res], res_axis_outer_list[3])
            sch[fmap_fractal].compute_at(sch[self.output_res], res_axis_outer_list[3])

            sch[workspace_res].compute_inline()
            sch[ub_res].compute_inline()
            sch[merge_co_ub].compute_inline()
            sch[merge_hw_ub].compute_inline()
            sch[split_c1_ub].compute_inline()

            block = tvm.thread_axis("blockIdx.x")
            sch[self.output_res].bind(res_bind_list[0], block)

            sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], te_platform.DMA_COPY)
            sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0], te_platform.SET_FMATRIX, setfmatrix_dict)
            sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], te_platform.IM2COL)
            sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], te_platform.DMA_COPY)
            sch[self.output_res].emit_insn(res_n_inner, te_platform.DMA_COPY)
        else:
            out_shape_len = len(out_shape)
            dma_split_i = 0
            prod = 1
            for i in range(out_shape_len - 1, -1, -1):
                prod = prod * out_shape[i]
                if prod > align_block_size:
                    dma_split_i = i
                    break

            res_ub_num = _ceil_div(_ceil_div(c_in_real, align_block_size) * align_block_size, c_in_real) + 1

            for i in range(min(1, dma_split_i), dma_split_i + 1):
                dma_split_factor, align_split, allow_multi_core = self._get_dma_split_factor(i, out_shape, out_w,
                                                                                             kernel_w,
                                                                                             align_block_size)
                if align_split or i == dma_split_i:
                    _schedule_32b_not_aligned(i, dma_split_factor, allow_multi_core, reg_mov=(i != out_shape_len - 1))
                    break

        sch[fmap_in_l1].double_buffer()
        sch[fmap_im2col].double_buffer()
        sch[fmap_fractal].double_buffer()
        sch[transpose_ub].double_buffer()
        sch[ub_res].double_buffer()
        self._add_compile_info(avg_split_ub_size, res_ub_num, align_block_size, workspace_output, workspace_filter,
                               workspace_c)
        return sch