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
Tool function of conv2d maxpool fusion schedule.
"""
from tbe.dsl.static_schedule.conv_schedule_util import ceil_div
from tbe import tvm


POOLING_STRIDE = 2
POOLING_WINDOW = 3
POOLING_2_2_WINDOW = 2

def maxpool_tensor_buffertile(sch, fmap_row_major, al1, al0, cl0, cub, res,
                              m_dim, stride_h, kernel_h, out_width,
                              fusion_mode, pooling_tiling_param, pooling_padding, conv_param,
                              singlecore_out2al1_loopm_axis, al12al0_loopm_axis, cl0_mo,
                              ):
    """
    Buffertile for maxpool fusion.
    """
    def al1_buffer_tile():
        """
        Buffer tile of al1.
        """
        al1_before_offset_bound_first = block_tile * ceil_div(pooling_al1_nparts, m_dim) \
            * pooling_al1_factor * POOLING_STRIDE * stride_h - conv_param.padding[0]

        al1_before_offset_bound = al1_before_offset_bound_first + \
            singlecore_out2al1_loopm_axis * pooling_al1_factor * POOLING_STRIDE * stride_h + stride_h - \
            stride_h * pooling_padding[0]

        extend_al1_first = int(kernel_h + pooling_al1_factor * POOLING_STRIDE * stride_h)

        extend_al1 = int(kernel_h + pooling_al1_factor * POOLING_STRIDE * stride_h) - stride_h

        al1_before_offset_bound_first -= stride_h * pooling_padding[0] * block_tile

        if out_width % 16 != 0:
            al1_before_offset_bound -= (align_16_nums * stride_h)
            al1_before_offset_bound_first -= (align_16_nums * stride_h)
            extend_al1 += (align_16_nums * stride_h)
            extend_al1_first += (align_16_nums * stride_h)

        sch[al1].buffer_tile(
            (None, None),
            (None, None),
            (tvm.select(singlecore_out2al1_loopm_axis.var == 0,
                        al1_before_offset_bound_first,
                        al1_before_offset_bound),
             tvm.select(singlecore_out2al1_loopm_axis.var == 0,
                        extend_al1_first,
                        extend_al1)),
            (None, None),
            (None, None),
        )

    def row_major_buffer_tile():
        """
        Buffer tile of fmap_row_major.
        """
        fcol_before_offset_bound_first = block_tile * ceil_div(pooling_al1_nparts, m_dim) * \
            pooling_al1_factor * POOLING_STRIDE * out_width

        fcol_before_offset_bound = singlecore_out2al1_loopm_axis * pooling_al1_factor * \
            POOLING_STRIDE * out_width + out_width

        fcol_before_offset_bound += (fcol_before_offset_bound_first - pooling_padding[0] * out_width)

        fcol_before_offset_bound_first -= block_tile * pooling_padding[0] * out_width

        extend_fmap_col_before_first = int(out_width * ((pooling_al1_factor - 1) * POOLING_STRIDE + POOLING_WINDOW))

        extend_fmap_col_before = int(out_width * pooling_al1_factor * POOLING_STRIDE)

        if out_width % 16 != 0:
            extend_fmap_col_before += (align_16_nums * out_width)
            extend_fmap_col_before_first += (align_16_nums * out_width)
            fcol_before_offset_bound -= align_16_nums * out_width
            fcol_before_offset_bound_first -= align_16_nums * out_width

        sch[fmap_row_major].buffer_tile(
            (None, None),
            (None, None),
            (tvm.select(singlecore_out2al1_loopm_axis.var == 0,
                        fcol_before_offset_bound_first,
                        fcol_before_offset_bound),
             tvm.select(singlecore_out2al1_loopm_axis.var == 0,
                        extend_fmap_col_before_first, extend_fmap_col_before)),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        )

    def al0_buffer_tile():
        """
        Buffer tile of al0.
        """
        fmap_col_offset_bound_first = ceil_div(pooling_al1_nparts, m_dim) * pooling_al1_factor * \
            POOLING_STRIDE * out_width

        fmap_col_offset_bound_first = block_tile * fmap_col_offset_bound_first

        fmap_col_offset_bound = (singlecore_out2al1_loopm_axis * pooling_al1_factor + al12al0_loopm_axis) * \
            POOLING_STRIDE * out_width

        fmap_col_offset_bound += (out_width + fmap_col_offset_bound_first - pooling_padding[0] * out_width)

        fmap_col_offset_bound_first -= block_tile * pooling_padding[0] * out_width

        fmap_col_offset_bound_first //= 16

        fmap_col_offset_bound //= 16

        sch[al0].buffer_tile(
            (None, None),
            (None, None),
            (tvm.select(
                singlecore_out2al1_loopm_axis.var == 0,
                tvm.select(al12al0_loopm_axis.var == 0,
                           fmap_col_offset_bound_first + cl0_mo * pooling_mc_cub,
                           fmap_col_offset_bound + cl0_mo * pooling_mc_cub),
                fmap_col_offset_bound + cl0_mo * pooling_mc_cub), pooling_mc_cub),
            (None, None),
            (None, None),
            (None, None),
        )

    def cub_buffer_tile():
        """
        Buffer tile of cub.
        """
        offset_bound_first = block_tile * ceil_div(pooling_al1_nparts, m_dim) * \
            pooling_al1_factor * POOLING_STRIDE * out_width

        offset_bound = ((singlecore_out2al1_loopm_axis * pooling_al1_factor + al12al0_loopm_axis) * \
            POOLING_STRIDE + 1) * out_width

        offset_bound += (offset_bound_first - pooling_padding[0] * out_width)

        offset_bound_first -= (block_tile * pooling_padding[0] * out_width)

        first_time_row = POOLING_WINDOW * out_width

        other_time_row = POOLING_STRIDE * out_width

        if out_width % 16 != 0:
            first_time_row = ceil_div(first_time_row, 16) * 16 + 16
            other_time_row = ceil_div(other_time_row, 16) * 16 + 16
            offset_bound_first = offset_bound_first // 16 * 16
            offset_bound = offset_bound // 16 * 16

        sch[cub].buffer_tile(
            (None, None),
            (None, None),
            (tvm.select(
                singlecore_out2al1_loopm_axis.var == 0,
                tvm.select(al12al0_loopm_axis.var == 0, offset_bound_first,
                           offset_bound), offset_bound),
             tvm.select(
                 singlecore_out2al1_loopm_axis.var == 0,
                 tvm.select(al12al0_loopm_axis.var == 0, first_time_row,
                            other_time_row), other_time_row)),
            (None, None),
        )
        sch[cl0].buffer_tile(
            (None, None),
            (None, None),
            (None, None),
            (tvm.select(
                singlecore_out2al1_loopm_axis.var == 0,
                tvm.select(al12al0_loopm_axis.var == 0, offset_bound_first,
                           offset_bound), offset_bound),
             tvm.select(
                 singlecore_out2al1_loopm_axis.var == 0,
                 tvm.select(al12al0_loopm_axis.var == 0, first_time_row,
                            other_time_row), other_time_row)),
            (None, None),
            (None, None),
            (None, None),
        )

    def row_major_2_2_buffer_tile():
        """
        Pooling 2*2 row_major buffer tile.
        """
        fcol_before_offset_bound_first = block_tile * ceil_div(pooling_al1_nparts, m_dim) \
            * pooling_al1_factor * POOLING_STRIDE * out_width \
            + singlecore_out2al1_loopm_axis * pooling_al1_factor * POOLING_STRIDE * out_width

        if singlecore_out2al1_loopm_axis.var != 0 or block_tile.var != 0:
            fcol_before_offset_bound_first \
            -= pooling_padding[0] * out_width
        extend_fmap_col_before_first = \
            int(out_width * ((pooling_al1_factor - 1)
                          * POOLING_STRIDE + POOLING_2_2_WINDOW))
        if out_width % 16 != 0:
            extend_fmap_col_before_first += (align_16_nums * out_width)
            fcol_before_offset_bound_first -= align_16_nums * out_width

        sch[fmap_row_major].buffer_tile(
            (None, None),
            (None, None),
            (fcol_before_offset_bound_first, extend_fmap_col_before_first),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        )

    def al1_2_2_buffer_tile():
        """
        Pooling 2*2 al buffer tile.
        """
        al1_before_offset_bound_first = block_tile * ceil_div(pooling_al1_nparts, m_dim) \
            * pooling_al1_factor * POOLING_STRIDE * stride_h \
            - conv_param.padding[0] \
            + singlecore_out2al1_loopm_axis * pooling_al1_factor * POOLING_STRIDE * stride_h

        if singlecore_out2al1_loopm_axis.var != 0 or block_tile.var != 0:
            al1_before_offset_bound_first -= stride_h * pooling_padding[0]

        extend_al1_first = int(kernel_h + pooling_al1_factor * POOLING_STRIDE * stride_h - stride_h)
        if out_width % 16 != 0:
            al1_before_offset_bound_first -= (align_16_nums * stride_h)
            extend_al1_first += (align_16_nums * stride_h)

        sch[al1].buffer_tile(
            (None, None),
            (None, None),
            (al1_before_offset_bound_first, extend_al1_first),
            (None, None),
            (None, None),
        )

    align_16_nums = ceil_div(16, out_width)
    block_tile = pooling_tiling_param["block_tile"]
    pooling_al1_factor = pooling_tiling_param["pooling_al1_factor"]
    pooling_al1_nparts = pooling_tiling_param["pooling_al1_nparts"]
    pooling_mc_cub = pooling_tiling_param["pooling_mc_cub"]

    if fusion_mode == "3*3":
        cub_buffer_tile()
        al0_buffer_tile()
        row_major_buffer_tile()
        al1_buffer_tile()

        sch[res].partition(singlecore_out2al1_loopm_axis, ((0, 0), ))
        sch[res].partition(al12al0_loopm_axis, ((0, 0), ))

    if fusion_mode == "2*2":
        row_major_2_2_buffer_tile()
        al1_2_2_buffer_tile()
