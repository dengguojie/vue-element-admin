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
conv3d backprop input general schedule.
"""
from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from tbe.dsl.compute import util as compute_util
from tbe.dsl.compute import cube_util


_NUM_3 = 3
_DEFAULT_TILING_FLAG = 32
_FUSION_NODE_WHITELIST = [
    "conv3d_backprop_input_dy_filling",
    "conv3d_backprop_input_w_l1",
    "conv3d_backprop_input_dy_l1_s1",
    "conv3d_backprop_input_c_ub_vn",
    "conv3d_backprop_input_dy_vn"]


def general_schedule(tensor, sch_list, tiling_case=None, var_range=None):  # pylint:disable=R0914, R0915
    """
    auto_schedule for cce AI-CORE.
    For now, only one convolution operation is supported.

    Parameters
    ----------
    tensor: tvm.tensor

    sch_list: use sch_list[0] to return conv schedule

    tiling_case: fix tiling for dynamic shape

    var_range: var_range for dynamic shape

    Returns
    -------
    True for sucess, False for no schedule
    """
    def _get_var_map(var_range):
        var_map = {}
        if var_range is None:
            return var_map
        var_name = ["batch_n", "dedy_d", "dedy_h", "dedy_w"]
        for var in var_name:
            if var in var_range:
                var_map[var] = var_range[var]
        return var_map

    def _cub_process():
        cub_tiling_mc_factor_m0 = cub_tiling_mc_factor * cub_tiling_m0
        cddr_n_outer, cddr_n_for_cub = sch[c_ddr].split(n_after_multicore, factor=cub_tiling_nc_factor)
        cddr_m_outer, cddr_m_for_cub = sch[c_ddr].split(m_after_multicore, factor=cub_tiling_mc_factor_m0)
        sch[c_ddr].reorder(cddr_n_outer, cddr_m_outer,
                           cddr_n_for_cub, cddr_m_for_cub)
        return cddr_n_outer, cddr_m_outer, cddr_n_for_cub

    def _l0c_procees():
        cddr_n_outer_outer, cddr_n_outer_inner = sch[c_ddr].split(cddr_n_outer,
                                                                  factor=cl0_tiling_nc // cub_tiling_nc_factor)
        cddr_m_outer_outer, cddr_m_outer_inner = sch[c_ddr].split(cddr_m_outer,
                                                                  factor=cl0_tiling_mc // cub_tiling_mc_factor)
        sch[c_ddr].reorder(cddr_n_outer_outer, cddr_m_outer_outer,
                           cddr_n_outer_inner, cddr_m_outer_inner)
        al1_at_ddr_m_outer, al1_at_ddr_m_inner = sch[c_ddr].split(cddr_m_outer_outer, factor=al1_tiling_m)
        bl1_at_ddr_n_outer, bl1_at_ddr_n_inner = sch[c_ddr].split(cddr_n_outer_outer, factor=bl1_tiling_n)
        batch_outer, batch_inner = sch[c_ddr].split(batch_after_multicore, factor=1)
        c_ddr_deep_outer, c_ddr_deep_inner = sch[c_ddr].split(d_after_multicore, factor=cddr_deep_factor)
        sch[c_ddr].reorder(
            c_ddr_deep_outer, al1_at_ddr_m_outer, batch_inner,
            bl1_at_ddr_n_outer, bl1_at_ddr_n_inner, al1_at_ddr_m_inner,
            c_ddr_deep_inner)
        col_at_ddr_axis = al1_at_ddr_m_inner
        return (
            batch_outer, al1_at_ddr_m_outer, batch_inner, bl1_at_ddr_n_outer,
            bl1_at_ddr_n_inner, col_at_ddr_axis, cddr_m_outer_inner, cddr_m_outer_outer,
            c_ddr_deep_outer, c_ddr_deep_inner)

    def _l0a_and_l0b_process():
        c_col_n_outer, c_col_n_inner = sch[c_col].split(c_col.op.axis[3], factor=bl0_tiling_nb)
        c_col_m_outer, c_col_m_inner = sch[c_col].split(c_col.op.axis[4], factor=al0_m_factor)
        c_col_deep_outer, c_col_deep_inner = sch[c_col].split(c_col.op.axis[2], factor=1)
        if kd_reduce_flag:  # pylint:disable=R1705
            reduce_axis_kd, reduce_axis_k1, reduce_axis_k0 = c_col.op.reduce_axis
            reduce_axis_kd_outer, reduce_axis_kd_inner = sch[c_col].split(reduce_axis_kd, factor=kd_factor)
            c_col_k_outer, c_col_k_inner = sch[c_col].split(reduce_axis_k1, factor=al0_tiling_ka)
            sch[c_col].reorder(c_col_k_outer, c_col_m_outer,
                               reduce_axis_kd_outer, c_col_n_outer,
                               c_col_deep_outer, reduce_axis_kd_inner,
                               c_col_deep_inner, c_col_n_inner, c_col_m_inner,
                               c_col.op.axis[5], c_col_k_inner, reduce_axis_k0)
            return (
                c_col_deep_outer, c_col_deep_inner, c_col_k_outer,
                c_col_m_outer, c_col_n_outer, reduce_axis_kd,
                reduce_axis_kd_outer, reduce_axis_kd_inner)
        else:
            reduce_axis_k1, reduce_axis_k0 = c_col.op.reduce_axis
            c_col_k_outer, c_col_k_inner = sch[c_col].split(reduce_axis_k1, factor=al0_tiling_ka)
            sch[c_col].reorder(c_col_k_outer, c_col_m_outer, c_col_n_outer,
                               c_col_deep_outer, c_col_deep_inner,
                               c_col_n_inner, c_col_m_inner, c_col.op.axis[5],
                               c_col_k_inner, reduce_axis_k0)
            return (
                c_col_deep_outer, c_col_deep_inner,
                c_col_k_outer, c_col_m_outer,
                c_col_n_outer)

    def _al1_and_bl1_process():
        if kd_reduce_flag:
            reduce_axis_kd_outer_outer, reduce_axis_kd_outer_inner = sch[c_col].split(reduce_axis_kd_outer, kd_tiling_l1_factor)
        if k_al1_factor > k_bl1_factor:
            factor_outer, factor_inner = k_al1_factor // k_bl1_factor, k_bl1_factor
            c_col_k_outer_outer, c_col_k_outer_inner = sch[c_col].split(c_col_k_outer, factor=factor_inner)
            c_col_k_outer_outer_outer, c_col_k_outer_outer_inner = sch[c_col].split(c_col_k_outer_outer,
                                                                                    factor=factor_outer)
            bl1_at_l0c_axis, al1_at_l0c_axis = c_col_k_outer_outer_inner, c_col_k_outer_outer_outer
            if kd_reduce_flag:
                sch[c_col].reorder(
                    reduce_axis_kd_outer_outer,
                    c_col_k_outer_outer_outer,
                    c_col_k_outer_outer_inner,
                    c_col_k_outer_inner,
                    reduce_axis_kd_outer_inner, c_col_m_outer)
            else:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)
        else:
            factor_outer, factor_inner = k_bl1_factor // k_al1_factor, k_al1_factor
            c_col_k_outer_outer, c_col_k_outer_inner = sch[c_col].split(c_col_k_outer, factor=factor_inner)
            c_col_k_outer_outer_outer, c_col_k_outer_outer_inner = sch[c_col].split(c_col_k_outer_outer,
                                                                                    factor=factor_outer)
            bl1_at_l0c_axis, al1_at_l0c_axis = c_col_k_outer_outer_outer, c_col_k_outer_outer_inner
            if kd_reduce_flag:
                sch[c_col].reorder(
                    reduce_axis_kd_outer_outer,
                    c_col_k_outer_outer_outer,
                    c_col_k_outer_outer_inner,
                    c_col_k_outer_inner,
                    reduce_axis_kd_outer_inner,
                    c_col_m_outer)
            else:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)
        reduce_axis_serial = [c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
                              c_col_k_outer_inner]
        return reduce_axis_serial, bl1_at_l0c_axis, al1_at_l0c_axis

    def _aub_process():
        aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
        if aub_tiling_k > al1_tiling_k:
            aub_tiling_k = al1_tiling_k
            tiling["AUB_shape"][0] = al1_tiling_k
        if aub_tiling_m > cl0_tiling_mc * cl0_tiling_m0:
            aub_tiling_m = cl0_tiling_mc * cl0_tiling_m0
            tiling["AUB_shape"][1] = aub_tiling_m

        aub_tiling_k_factor, aub_tiling_m_factor = aub_tiling_k // (kernel_h * kernel_w * 16), aub_tiling_m
        _, _, _, _, aub_w, _ = cube_util.shape_to_list(a_filling.shape)
        a_l1_k_outer, a_l1_k_inner = sch[a_l1].split(sch[a_l1].op.axis[2], factor=aub_tiling_k_factor)
        a_l1_h_outer, a_l1_h_inner = sch[a_l1].split(sch[a_l1].op.axis[3], factor=aub_tiling_m_factor)
        a_l1_w_outer, a_l1_w_inner = sch[a_l1].split(sch[a_l1].op.axis[4], factor=aub_w)
        sch[a_l1].reorder(a_l1_k_outer, a_l1_h_outer, a_l1_w_outer,
                          sch[a_l1].op.axis[0], sch[a_l1].op.axis[1],
                          a_l1_k_inner, a_l1_h_inner, a_l1_w_inner)
        if var_map and (stride_h > 1 or stride_w > 1):
            sch[a_zero].buffer_tile((None, None), (None, None), (None, None),
                                    (None, aub_tiling_m_factor),
                                    (None, None), (None, None))
            sch[a_vn].buffer_tile((None, None), (None, None), (None, None),
                                  (None, aub_tiling_m_factor),
                                  (None, None), (None, None))
        return a_l1_h_outer

    def _multi_core():  # pylint:disable=R0914
        block_dim = tiling['block_dim']
        group_dim = tiling['g_dim']
        batch_dim, n_dim, m_dim, d_dim = block_dim
        blocks = group_dim * batch_dim * n_dim * m_dim * d_dim

        if blocks != 1:
            multicore_batch, batch_outer_inner = sch[c_ddr].split(batch_outer, nparts=batch_dim)
            if "dedy_d" in var_map:
                d_dim_correct = tvm.select(tvm.floormod(tvm.floordiv(dy_depth, al0_tiling_dfactor), d_dim) == 0,
                                           d_dim, 1)
                multicore_d, c_ddr_deep_outer_inner = sch[c_ddr].split(c_ddr_deep_outer, nparts=d_dim)
            else:
                d_factor = compute_util.int_ceil_div(c_ddr_deep_outer_value, d_dim)
                multicore_d, c_ddr_deep_outer_inner = sch[c_ddr].split(c_ddr_deep_outer, factor=d_factor)
                # d_dim may be larger than uesd
                tiling['block_dim'][-1] = compute_util.int_ceil_div(c_ddr_deep_outer_value, d_factor)
            # split n axis
            multicore_n, bl1_at_ddr_n_outer_inner = sch[c_ddr].split(bl1_at_ddr_n_outer, nparts=n_dim)
            # split m axis
            multicore_m, al1_at_ddr_m_outer_inner = sch[c_ddr].split(al1_at_ddr_m_outer, nparts=m_dim)
            # split g axis
            multicore_group, g_axis_inner = sch[c_ddr].split(g_axis, nparts=group_dim)
            # reorder
            sch[c_ddr].reorder(
                multicore_group,
                multicore_batch, multicore_d,
                multicore_n, multicore_m, g_axis_inner,
                batch_outer_inner,
                c_ddr_deep_outer_inner,
                bl1_at_ddr_n_outer_inner,
                al1_at_ddr_m_outer_inner)

            out_fused = sch[c_ddr].fuse(multicore_group,
                                        multicore_batch,
                                        multicore_d,
                                        multicore_n,
                                        multicore_m)
            out_fused_out, _ = sch[c_ddr].split(out_fused, nparts=blocks)
            bind_out, _ = sch[c_ddr].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_ddr].bind(bind_out, blockidx)
        else:
            batch_outer_outer, batch_outer_inner = sch[c_ddr].split(batch_outer, nparts=1)
            bind_out, _ = sch[c_ddr].split(batch_outer_outer, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_ddr].bind(bind_out, blockidx)
            g_axis_inner = g_axis
            c_ddr_deep_outer_inner = c_ddr_deep_outer
            bl1_at_ddr_n_outer_inner = bl1_at_ddr_n_outer
            al1_at_ddr_m_outer_inner = al1_at_ddr_m_outer
        return (batch_outer_inner, c_ddr_deep_outer_inner, bl1_at_ddr_n_outer_inner, al1_at_ddr_m_outer_inner,
                g_axis_inner, blockidx, blocks)

    def _tiling_check():
        _tiling_check_none()
        _tiling_check_value()
        _tiling_check_factor()
        _tiling_check_pbuffer()
        if stride_h == 1 and stride_w == 1 and aub_fusion_flag == False:
            if tiling.get("AUB_shape") is not None:
                cube_err.raise_err_specific("conv3d_backprop_input",
                    'stride = 1 but AUB_shape is not None.')

        if tiling.get("BL0_matrix") == [] and tiling.get("BL1_shape") != []:
            cube_err.raise_err_specific("conv3d_backprop_input", "BL0 full load but BL1 not!")

    def _tiling_check_value():
        if tiling.get("BL0_matrix"):
            if al0_tiling_ka != bl0_tiling_kb:
                cube_err.raise_err_specific("conv3d_backprop_input", "in BL0_matrix, ka != kb")

            if bl0_tiling_nb != cl0_tiling_nc:
                cube_err.raise_err_specific("conv3d_backprop_input", "in BL0_matrix, nb != nc.")

        if al0_tiling_ma != cl0_tiling_mc:
            cube_err.raise_err_specific("conv3d_backprop_input", "ma != mc.")

    def _tiling_check_none():
        if ((tiling.get("AL1_shape") is None) or
            (tiling.get("BL1_shape") is None) or
            (tiling.get("CUB_matrix") is None)):
            cube_err.raise_err_three_paras(
                'E62305', 'conv3d_backprop_input', 'AL1_shape/BL1_shape/CUB_matrix', 'not None', 'None')

        if ((tiling.get("AL0_matrix") is None) or
            (tiling.get("BL0_matrix") is None) or
            (tiling.get("CL0_matrix") is None)):
            cube_err.raise_err_three_paras(
                'E62305', 'conv3d_backprop_input', 'AL0_matrix/BL0_matrix/CL0_matrix', 'not None', 'None')

        if (tiling["BUB_shape"] is None
            or tiling["BUB_shape"][0] is None
            or tiling["BUB_shape"][0] == 0):
            tiling["g_dim"] = 1
        else:
            tiling["g_dim"] = tiling["BUB_shape"][0]

    def _tiling_check_factor():
        if (kernel_w * kernel_h * cout1_g) % al0_tiling_ka != 0:
            cube_err.raise_err_three_paras(
                'E62305', 'conv3d_backprop_input', 'Co1*Hk*Wk % ka', '0',
                str((kernel_w * kernel_h * cout1_g) % al0_tiling_ka))

        if al1_tiling_k % al0_tiling_ka != 0:
            cube_err.raise_err_three_paras(
                'E62305', 'conv3d_backprop_input', 'k_AL1 % ka', '0', str(al1_tiling_k % al0_tiling_ka))

        if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
            if bl1_tiling_k % bl0_tiling_kb != 0:
                cube_err.raise_err_three_paras(
                    'E62305', 'conv3d_backprop_input', 'k_BL1 % kb', '0', str(bl1_tiling_k % bl0_tiling_kb))

        if cl0_tiling_nc % cub_tiling_nc_factor != 0:
            cube_err.raise_err_three_paras(
                    'E62305', 'conv3d_backprop_input', 'nc % nc_factor', '0',
                    str(cl0_tiling_nc % cub_tiling_nc_factor))

        if tiling.get("BL1_shape"):
            if al1_tiling_k > bl1_tiling_k and al1_tiling_k % bl1_tiling_k != 0:
                cube_err.raise_err_specific("conv3d_backprop_input", "k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.")

            if bl1_tiling_k > al1_tiling_k and bl1_tiling_k % al1_tiling_k != 0:
                cube_err.raise_err_specific("conv3d_backprop_input", "k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.")

    def _tiling_check_pbuffer():
        if stride_h > 1 or stride_w > 1:
            pb_flag = [
                aub_pbuffer, al1_pbuffer, bl1_pbuffer, al0_pbuffer,
                bl0_pbuffer, l0c_pbuffer, cub_pbuffer
            ]
            pb_list = [
                'aub_pbuffer', 'al1_pbuffer', 'bl1_pbuffer', 'al0_pbuffer',
                'bl0_pbuffer', 'l0c_pbuffer', 'cub_pbuffer'
            ]
            for flag, name in zip(pb_flag, pb_list):
                if flag not in (1, 2):
                    cube_err.raise_err_three_paras(
                    'E62305', 'conv3d_backprop_input', name, '1 or 2', str(flag))

    def _fetch_tensor_info(var_map):  # pylint:disable=R0914, R0915
        tensor_attr = {}

        if mean_flag:
            sch[mean_matrix_init].set_scope(tbe_platform.scope_ubuf)
            sch[mean_matrix_mul].set_scope(tbe_platform.scope_ubuf)
            sch[mean_matrix_fp16].set_scope(tbe_platform.scope_ubuf)

        stride_d = c_ub_exact_hw.op.attrs["stride_d"].value
        group_dict = c_ub_exact_hw.op.attrs["group_dict"]
        c_ub_vn = tensor_map.get("c_ub_vn")
        c_fill_zero = tensor_map.get("c_fill_zero")
        c_ub = tensor_map.get("c_ub")
        output_shape = cube_util.shape_to_list(c_ub_exact_hw.op.attrs["output_shape"])
        sch[c_fill_zero].set_scope(tbe_platform_info.scope_ubuf)
        sch[c_ub_vn].set_scope(tbe_platform_info.scope_ubuf)
        c_col = tensor_map.get("c_col")
        a_col = tensor_map.get("a_col")
        b_col = tensor_map.get("b_col")
        b_l1 = tensor_map.get("b_l1")
        sch[b_l1].set_scope(tbe_platform_info.scope_cbuf)
        b_ddr = b_l1.op.input_tensors[0]  # weight in ddr
        kernel_d, kernel_h, kernel_w = list(i.value for i in c_ub_exact_hw.op.attrs["kernels"])
        a_col_before = a_col.op.input_tensors[0]  # im2col_row_major in L1
        dilation = list(i.value for i in a_col_before.op.attrs["dilation"])
        padding = cube_util.shape_to_list(a_col_before.op.attrs["padding"])
        padding_var = a_col_before.op.attrs["padding_var"]
        weight_out_var = a_col_before.op.attrs["width_out_var"]

        tensor_map['c_fill_zero'] = c_fill_zero
        tensor_map['c_ub_vn'] = c_ub_vn
        tensor_map['c_ub'] = c_ub
        tensor_map['c_col'] = c_col
        tensor_map['a_col'] = a_col
        tensor_map['b_col'] = b_col
        tensor_map['b_l1'] = b_l1
        tensor_map['b_ddr'] = b_ddr
        tensor_map['a_col_before'] = a_col_before
        tensor_map['dilation'] = dilation

        # stride > 1
        if "a_filling" in tensor_map.keys():
            a_l1 = tensor_map.get("a_l1")
            a_filling = tensor_map.get("a_filling")
            a_zero = tensor_map.get("a_zero")

            stride_h, stride_w = list(i.value for i in a_filling.op.attrs["stride_expand"])
            if not var_map:
                a_ddr = a_filling.op.input_tensors[0]  # dEdY in ddr
            else:
                a_vn = a_l1.op.input_tensors[0]
                a_ddr = a_filling.op.input_tensors[0]
        else:
            a_l1 = a_col_before.op.input_tensors[0]
            a_ddr = a_l1.op.input_tensors[0]  # dEdY in ddr
            stride_h = 1
            stride_w = 1
        if var_map:
            if "dedy_d" in var_map:
                sch.set_var_range(a_ddr.shape[1], *var_range.get("dedy_d"))
                sch.set_var_range(output_shape[1], *var_range.get("dedx_d"))
            if "dedy_h" in var_map:
                sch.set_var_range(a_ddr.shape[3], *var_range.get("dedy_h"))
                sch.set_var_range(output_shape[3], *var_range.get("dedx_h"))
            if "dedy_w" in var_map:
                sch.set_var_range(a_ddr.shape[4], *var_range.get("dedy_w"))
                sch.set_var_range(output_shape[4], *var_range.get("dedx_w"))
            if "batch_n" in var_map:
                sch.set_var_range(a_ddr.shape[0], *var_range.get("batch_n"))
                sch.set_var_range(output_shape[0], *var_range.get("batch_n"))
            dy_l1_attr = a_l1.op.attrs
            if 'info_dy_h' in dy_l1_attr:
                info_dy_h = dy_l1_attr["info_dy_h"]
                info_dy_h_value = dy_l1_attr["info_dy_h_value"]
                sch.set_var_value(info_dy_h, info_dy_h_value)
                sch.set_var_range(info_dy_h, 1, None)

            if 'info_dy_w' in dy_l1_attr:
                info_dy_w = dy_l1_attr["info_dy_w"]
                info_dy_w_value = dy_l1_attr["info_dy_w_value"]
                sch.set_var_value(info_dy_w, info_dy_w_value)
                sch.set_var_range(info_dy_w, 1, None)

            info_padding_up, info_padding_bottom, info_padding_left, info_padding_right = padding_var
            padding_up, padding_bottom, padding_left, padding_right = padding

            sch.set_var_value(info_padding_up, padding_up)
            sch.set_var_value(info_padding_bottom, padding_bottom)
            sch.set_var_value(info_padding_left, padding_left)
            sch.set_var_value(info_padding_right, padding_right)

            sch.set_var_range(info_padding_up, 0, None)
            sch.set_var_range(info_padding_bottom, 0, None)
            sch.set_var_range(info_padding_left, 0, None)
            sch.set_var_range(info_padding_right, 0, None)

            sch.set_var_value(weight_out_var, output_shape[-2])
            sch.set_var_range(weight_out_var, 1, None)

        tensor_map['a_ddr'] = a_ddr
        tensor_attr['stride_w'] = stride_w
        tensor_attr['stride_h'] = stride_h
        # dataflow management
        sch[b_col].set_scope(tbe_platform_info.scope_cb)
        if stride_h == 1 and stride_w == 1:
            sch[a_l1].set_scope(tbe_platform_info.scope_cbuf)
            tensor_map['a_l1'] = a_l1
            if aub_fusion_flag:
                tensor_map['a_filling'] = tensor_map['elewise_mul_before'] if not mean_flag else tensor_map['a_ub_mul']
        else:
            if aub_fusion_flag:
                tensor_map['a_ub'] = tensor_map['elewise_mul_before'] if not mean_flag else tensor_map['a_ub_mul']
                if var_map:
                    sch[a_vn].set_scope(tbe_platform.scope_ubuf)
            else:
                if not var_map:
                    a_ub = sch.cache_read(a_ddr, tbe_platform_info.scope_ubuf, [a_filling])
                    tensor_map['a_ub'] = a_ub
                else:
                    sch[a_vn].set_scope(tbe_platform_info.scope_ubuf)
            # generate a_zero in ub
            sch[a_zero].set_scope(tbe_platform_info.scope_ubuf)
            sch[a_filling].set_scope(tbe_platform_info.scope_ubuf)
            # dma : a_filling ub------>L1
            sch[a_l1].set_scope(tbe_platform_info.scope_cbuf)

        sch[a_col_before].set_scope(tbe_platform_info.scope_cbuf)
        sch[a_col].set_scope(tbe_platform_info.scope_ca)

        sch[c_col].set_scope(tbe_platform_info.scope_cc)
        sch[c_ub].set_scope(tbe_platform_info.scope_ubuf)
        tensor_attr['padding'] = padding
        tensor_attr['padding_var'] = padding_var
        tensor_attr['output_shape'] = output_shape
        tensor_attr['stride_d'] = stride_d
        tensor_attr['kernel_d'] = kernel_d
        tensor_attr['kernel_h'] = kernel_h
        tensor_attr['kernel_w'] = kernel_w

        return tensor_attr, group_dict

    def _tiling_l0_process():
        if tiling.get("BL0_matrix"):
            bl0_tiling_kb, bl0_tiling_nb, _, _, _, bl0_tiling_kd = tiling.get("BL0_matrix")
        else:
            bl0_tiling_group, bl0_tiling_kd, bl0_tiling_kb, bl0_tiling_nb, _, _ = list(i.value for i in b_col.shape)
            bl0_tiling_nb = bl0_tiling_nb // n_dim
        return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_kd

    def _tiling_l1_process():
        if tiling.get("AL1_shape"):
            al1_tiling_k, al1_tiling_m, _, _ = tiling.get("AL1_shape")
            if (al1_tiling_k == kernel_h * kernel_w * cout1_g * al1_co0 and al1_tiling_m == compute_util.int_ceil_div(
                c_l0c_hw, (tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0]*cl0_tiling_mc))):
                tiling["AL1_shape"] = []
        else:
            # batch = 1 other axes full load
            al1_tiling_k = kernel_h * kernel_w * cout1_g * al1_co0
            al1_tiling_m = compute_util.int_ceil_div(c_l0c_hw, (tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0] *
                                                     cl0_tiling_mc * m_dim))
        if tiling.get("BL1_shape"):
            bl1_tiling_k, bl1_tiling_n, _, bl1_tiling_kdparts = tiling.get("BL1_shape")
        else:
            bl1_tiling_k = kernel_h * kernel_w * bl1_co0 * bl1_co1
            bl1_tiling_n = bl1_k1 // (kernel_h * kernel_w * cl0_tiling_nc) // n_dim
            bl1_tiling_kdparts = 1
        return (al1_tiling_k, al1_tiling_m, bl1_tiling_k,
                bl1_tiling_n, bl1_tiling_kdparts)

    def _reorder_management():
        reorder_flag = False
        if k_al1_factor != 1 and k_bl1_factor == 1:
            reorder_flag = True
        if tiling['AL1_shape'] != [] and tiling['BL1_shape'] != [] and k_al1_factor == 1 and k_bl1_factor == 1:
            if tiling['AL1_shape'][1] > tiling['BL1_shape'][1]:
                reorder_flag = True
        if tiling['AL1_shape'] != [] and tiling['BL1_shape'] != [] and k_al1_factor != 1 and k_bl1_factor != 1:
            if tiling['AL1_shape'][1] > tiling['BL1_shape'][1]:
                reorder_flag = True
        if reorder_flag:
            sch[c_ddr].reorder(bl1_at_ddr_n_outer,
                               al1_at_ddr_m_outer, batch_inner)

    def _get_dfactor():
        estimate_d = (b_ddr_kd - 2 + al0_tiling_dfactor + stride_d - 1) // stride_d + 1
        d_factor = min(estimate_d, dy_depth)
        if b_ddr_kd == stride_d:
            d_factor = max(d_factor - 1, 1)
        return d_factor

    def _dfactor_dynamic(var_map):
        ext = (al0_tiling_dfactor - 1 + stride_d - 1) // stride_d
        b_factor = min(bl1_tiling_kdparts*bl0_tiling_kd, b_ddr_kd)
        estimate_d = (b_factor - 1 + stride_d - 1) // stride_d + ext + 1
        if "dedy_d" in var_map:
            d_factor = tvm.min(estimate_d, dy_depth)
        else:
            d_factor = min(estimate_d, dy_depth)
        return d_factor

    def _get_h_l1(howo_size):
        left = 0
        right = 0
        max_dis = 0

        for x in range(1, compute_util.int_ceil_div(cddr_h * cddr_w, howo_size)):
            m_length = x * howo_size
            right = m_length // cddr_w
            distance = right - left + 1 if (m_length % cddr_w != 0) else right - left
            if max_dis < distance:
                max_dis = distance
            left = right

        h_l1 = min(kernel_h - 1 + max_dis, ho_l1)
        if cddr_h * cddr_w <= howo_size:
            h_l1 = ho_l1

        return h_l1

    def _check_exceed_l1_buffer(howo_size):
        c0_size = 16
        n_dim = tiling['block_dim'][1]
        if not tiling['BL1_shape']:
            b_l1_size = b_ddr_n1 * b_ddr_k1 * b_ddr_k0 * b_ddr_n0 // n_dim // real_g * 2
        elif (bl1_tiling_k == kernel_h * kernel_w * bl1_co1 * bl1_co0 and
              kd_factor * kd_tiling_l1_factor == b_ddr_kd):
            b_l1_size = b_ddr_kd * bl1_tiling_k * bl1_tiling_n * cl0_tiling_nc * c0_size * 2
        else:
            b_l1_k = bl1_tiling_n * cl0_tiling_nc * kernel_h * kernel_w
            b_l1_n = bl1_tiling_k // (kernel_h * kernel_w) // c0_size
            b_l1_size = kd_factor * kd_tiling_l1_factor * b_l1_n * b_l1_k * c0_size**2 * 2

        d_factor = _get_dfactor()
        h_l1 = _get_h_l1(howo_size)
        dy_l1_size = d_factor * dy_cout1 * h_l1 * wo_l1 * c0_size * 2

        if (dy_l1_size + b_l1_size) > tbe_platform_info.get_soc_spec("L1_SIZE"):
            return True

        return False

    def _check_exceed_ub_buffer():
        c0_size = 16
        aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
        aub_tiling_k_factor, aub_tiling_m_factor = aub_tiling_k // (kernel_h * kernel_w * 16), aub_tiling_m
        d_factor = _get_dfactor()

        dedy_ub_size = (d_factor * aub_tiling_k_factor * dy_w * c0_size * 2 *
                        compute_util.int_ceil_div(aub_tiling_m_factor, stride_h))
        dy_filing_size = d_factor * aub_tiling_k_factor * aub_tiling_m_factor * (dy_w * stride_w) * c0_size * 2
        c_ub_size = cub_tiling_nc_factor * cub_tiling_mc_factor * c0_size**2 * cub_pbuffer * 2
        c_ub_size = c_ub_size * (cub_fused_num + 1)
        ub_size = tbe_platform_info.get_soc_spec("UB_SIZE")
        if (dedy_ub_size * (aub_fused_num + 1) + dy_filing_size + c_ub_size) > ub_size:
            return True

        return False

    def _check_exceed_buffer(howo_size):
        if _check_exceed_l1_buffer(howo_size):
            return True

        if stride_h > 1 or stride_w > 1 or aub_fusion_flag:
            if _check_exceed_ub_buffer():
                return True

        return False

    def _do_compute_at():
        m_dim = tiling['block_dim'][2]
        howo_out = cddr_h * cddr_w
        howo_deep_outer = compute_util.int_ceil_div(howo_out, m_dim)
        howo_m_outer = al1_tiling_m * al0_tiling_ma * al0_tiling_m0

        if mean_flag:
            sch[mean_matrix_init].compute_at(sch[compute_at_buffer[0]], compute_at_axis[0])
            sch[mean_matrix_mul].compute_at(sch[compute_at_buffer[0]], compute_at_axis[0])
            sch[mean_matrix_fp16].compute_at(sch[compute_at_buffer[0]], compute_at_axis[0])

        if var_map:
            sch[a_l1].compute_at(sch[c_col], al1_at_l0c_axis)
            sch[a_col_before].compute_at(sch[c_col], al1_at_l0c_axis)
        elif (not tiling['AL1_shape'] and not
            _check_exceed_buffer(howo_deep_outer)):
            sch[a_l1].compute_at(sch[c_ddr], c_ddr_deep_outer)
            sch[a_col_before].compute_at(sch[c_ddr], c_ddr_deep_outer)
        elif (al1_tiling_k == kernel_h * kernel_w * cout1_g * al1_co0 and
              not _check_exceed_buffer(howo_m_outer)):
            sch[a_l1].compute_at(sch[c_ddr], al1_at_ddr_m_outer)
            sch[a_col_before].compute_at(sch[c_ddr], al1_at_ddr_m_outer)
        else:
            sch[a_l1].compute_at(sch[c_col], al1_at_l0c_axis)
            sch[a_col_before].compute_at(sch[c_col], al1_at_l0c_axis)

        # bl1_compute_at
        if not tiling['BL1_shape']:
            sch[b_l1].compute_at(sch[c_ddr], batch_outer)
        elif (bl1_tiling_k == kernel_h * kernel_w * bl1_co0 * bl1_co1 and
              kd_factor * kd_tiling_l1_factor == b_ddr_kd):
            sch[b_l1].compute_at(sch[c_ddr], bl1_at_ddr_n_outer)
        else:
            sch[b_l1].compute_at(sch[c_col], bl1_at_l0c_axis)
        sch[c_ub].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_fill_zero].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_ub_vn].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_col].compute_at(sch[c_ddr], col_at_ddr_axis)
        sch[a_col].compute_at(sch[c_col], c_col_m_outer)
        if not tiling['BL0_matrix'] and not tiling['BL1_shape']:
            sch[b_col].compute_at(sch[c_ddr], c_ddr_deep_outer)
        else:
            sch[b_col].compute_at(sch[c_col], c_col_n_outer)
        if stride_h > 1 or stride_w > 1:
            sch[a_filling].compute_at(sch[a_l1], a_l1_h_outer)
            sch[a_zero].compute_at(sch[a_l1], a_l1_h_outer)
            if var_map:
                sch[a_vn].compute_at(sch[a_l1], a_l1_h_outer)
            else:
                sch[a_ub].compute_at(sch[a_l1], a_l1_h_outer)

    def _fused_double_buffer():
        if aub_pbuffer == 2:
            for tensor in aub_body_tensors:
                sch[tensor].double_buffer()

            for tensor in aub_input_tensors:
                sch[tensor].double_buffer()

        if cub_pbuffer == 2:
            for tensor in cub_body_tensors:
                sch[tensor].double_buffer()
        
            for tensor in cub_input_tensors:
                sch[tensor].double_buffer()

    def _double_buffer():
        if stride_h > 1 or stride_w > 1:
            if aub_pbuffer == 2:
                sch[a_filling].double_buffer()
                sch[a_zero].double_buffer()
                if var_map:
                    sch[a_vn].double_buffer()
                else:
                    sch[a_ub].double_buffer()

        if al1_pbuffer == 2:
            sch[a_l1].double_buffer()

        if bl1_pbuffer == 2:
            sch[b_l1].double_buffer()

        if al0_pbuffer == 2:
            sch[a_col].double_buffer()

        if bl0_pbuffer == 2:
            sch[b_col].double_buffer()

        if l0c_pbuffer == 2:
            sch[c_col].double_buffer()

        if cub_pbuffer == 2:
            sch[c_ub].double_buffer()
            sch[c_fill_zero].double_buffer()
            sch[c_ub_vn].double_buffer()

        _fused_double_buffer()

    def _default_tiling():
        tiling = {}
        # defaut value 16
        k0_size = tbe_platform.CUBE_MKN[a_ddr.dtype]["mac"][1]
        k_al1 = kernel_h * kernel_w * k0_size

        if stride_h > 1 or stride_w > 1 or aub_fusion_flag:
            tiling["AUB_shape"] = [kernel_h * kernel_w * k0_size, 1, 1, 1]
            tiling["BUB_shape"] = None
        else:
            tiling["AUB_shape"] = None
            tiling["BUB_shape"] = None

        tiling["AL1_shape"] = [k_al1, 1, 1, 1]
        tiling["BL1_shape"] = [k0_size, 1, 1, 1]
        tiling["AL0_matrix"] = [1, 1, 16, k0_size, 1, 1]
        tiling["BL0_matrix"] = [1, 1, 16, k0_size, 1, 1]
        tiling["CL0_matrix"] = [1, 1, 16, 16, 1, 1]
        tiling["CUB_matrix"] = [1, 1, 16, 16, 1, 1]
        tiling["block_dim"] = [1, 1, 1, 1]
        tiling["n_bef_batch_flag"] = 0
        tiling["n_bef_group_flag"] = 0
        tiling["batch_bef_group_flag"] = 0
        tiling["A_overhead_opt_flag"] = 0
        tiling["B_overhead_opt_flag"] = 0
        tiling["AUB_channel_wise_flag"] = None
        tiling["BUB_channel_wise_flag"] = None
        tiling["CUB_channel_wise_flag"] = None
        tiling["manual_pingpong_buffer"] = {
            'AUB_pbuffer': 1,
            'BUB_pbuffer': 1,
            'AL1_pbuffer': 1,
            'BL1_pbuffer': 1,
            'AL0_pbuffer': 1,
            'BL0_pbuffer': 1,
            'CL0_pbuffer': 1,
            'CUB_pbuffer': 1,
            'UBG_pbuffer': 1,
        }
        return tiling

    def _emit_insn_process():
        sch[b_l1].emit_insn(sch[b_l1].op.axis[1], "dma_copy")
        sch[b_col].emit_insn(sch[b_col].op.axis[3], "dma_copy")

        if mean_flag:
            sch[mean_matrix_init].emit_insn(mean_matrix_init.op.axis[-1], "vector_dup")
            sch[mean_matrix_mul].emit_insn(mean_matrix_mul.op.axis[-1], "vector_auto")
            sch[mean_matrix_fp16].emit_insn(mean_matrix_fp16.op.axis[0], "vector_auto")
            sch[mean_matrix_fp16].reused_by(mean_matrix_mul)

        if stride_h > 1 or stride_w > 1:
            if aub_fusion_flag == False and not var_map:
                sch[a_ub].emit_insn(sch[a_ub].op.axis[0], "dma_copy")
            afill_n, afill_d, afill_c, afill_h, afill_w, _ = sch[a_filling].op.axis
            afill_w_out, afill_w_inner = sch[a_filling].split(
                afill_w, factor=stride_w)
            if not var_map:
                sch[a_filling].reorder(
                    afill_w_inner,
                    afill_n,
                    afill_d,
                    afill_c,
                    afill_h,
                    afill_w_out)
            else:
                sch[a_filling].reorder(
                    afill_w_inner,
                    afill_h,
                    afill_n,
                    afill_d,
                    afill_c,
                    afill_w_out)
            sch[a_filling].unroll(afill_w_inner)
            sch[a_filling].reused_by(a_zero)
            sch[a_zero].emit_insn(sch[a_zero].op.axis[0], "vector_dup")
            if var_map:
                sch[a_filling].emit_insn(afill_n, "dma_copy")
            else:
                sch[a_filling].emit_insn(afill_n, "vector_muls")

        if not var_map:
            setfmatrix_dict = {"conv_kernel_h": kernel_h,
                               "conv_kernel_w": kernel_w,
                               "conv_padding_top": padu,
                               "conv_padding_bottom": padd,
                               "conv_padding_left": padl,
                               "conv_padding_right": padr,
                               "conv_stride_h": 1,
                               "conv_stride_w": 1,
                               "conv_fm_c": cout1_g * a_l1.shape[5],
                               "conv_fm_h": a_l1.shape[3],
                               "conv_fm_w": a_l1.shape[4],
                               "conv_dilation_h": dilation_h,
                               "conv_dilation_w": dilation_w}
            row_major_tag = 'set_fmatrix'
            load3d_tag = 'im2col'
            setfmatrix_dict0 = {}
        else:            
            setfmatrix_dict0 = {
                "set_fmatrix": 0,
                "enable_row_major_vm_desc": 1,
                "conv_kernel_h": kernel_h,
                "conv_kernel_w": kernel_w,
                "conv_padding_top": padu_var,
                "conv_padding_bottom": padd_var,
                "conv_padding_left": padl_var,
                "conv_padding_right": padr_var,
                "conv_stride_h": 1,
                "conv_stride_w": 1,
                "conv_fm_c": cout1_g * a_l1.shape[5],
                "conv_fm_h": a_l1.shape[3],
                "conv_fm_w": a_l1.shape[4],
                "conv_dilation_h": dilation_h,
                "conv_dilation_w": dilation_w
            }
            setfmatrix_dict = {
                "set_fmatrix": 1,
                "enable_row_major_vm_desc": 1,
                "conv_kernel_h": kernel_h,
                "conv_kernel_w": kernel_w,
                "conv_padding_top": padu_var,
                "conv_padding_bottom": padd_var,
                "conv_padding_left": padl_var,
                "conv_padding_right": padr_var,
                "conv_stride_h": 1,
                "conv_stride_w": 1,
                "conv_fm_c": cout1_g * a_l1.shape[5],
                "conv_fm_c1": a_l1.shape[2],
                "conv_fm_h": a_l1.shape[3],
                "conv_fm_w": a_l1.shape[4],
                "conv_fm_c0": a_l1.shape[5],
                "group_flag": 1
            }
            if stride_h != 1 or stride_w != 1:
                sch[a_vn].reused_by(a_filling)
                sch[a_vn].emit_insn(a_vn.op.axis[0], "phony_insn")

            row_major_tag = 'row_major_vm'
            load3d_tag = 'im2col_v2'

        sch[a_l1].emit_insn(sch[a_l1].op.axis[0], "dma_copy", setfmatrix_dict0)
        sch[a_col_before].emit_insn(a_col_before.op.axis[2], row_major_tag, setfmatrix_dict)
        _, a_col_deep_inner = sch[a_col].split(sch[a_col].op.axis[2], factor=1)
        sch[a_col].emit_insn(a_col_deep_inner, load3d_tag, setfmatrix_dict)

        _, n_dim, m_dim, d_dim = tiling['block_dim']
        deep_index = (((blockidx // (n_dim * m_dim)) % d_dim) *
                      ((c_ddr_deep_outer_value + d_dim - 1) // d_dim) +
                      c_ddr_deep_outer) * cddr_deep_factor + c_col_deep_outer

        if kd_reduce_flag:
            axis_kd = reduce_axis_kd_outer * kd_factor + reduce_axis_kd_inner
            mad_dict = {"mad_pattern": 2,
                        "k_outer": [reduce_axis_serial[0],
                                    reduce_axis_serial[1],
                                    reduce_axis_serial[2],
                                    reduce_axis_kd],
                        'k_cond':
                            tvm.all(
                                deep_index + pad_head -
                                tvm.min(a_ddr.shape[1] - 1,
                                        (deep_index + pad_head)//stride_d)
                                * stride_d - axis_kd == 0,
                                reduce_axis_serial[0].var == 0,
                                reduce_axis_serial[1].var == 0,
                                reduce_axis_serial[2].var == 0)}

            mad_dict_stride1 = {"mad_pattern": 2,
                                "k_outer": [reduce_axis_serial[0],
                                            reduce_axis_serial[1],
                                            reduce_axis_serial[2],
                                            reduce_axis_kd],
                                'k_cond':
                                    tvm.all(
                                        axis_kd +
                                        tvm.min(0, dy_depth - 1
                                                - pad_head - deep_index) == 0,
                                        reduce_axis_serial[0].var == 0,
                                        reduce_axis_serial[1].var == 0,
                                        reduce_axis_serial[2].var == 0)}
        else:
            mad_dict_originovrlap = {"mad_pattern": 2,
                                     "k_outer":  [reduce_axis_serial[0],
                                                  reduce_axis_serial[1],
                                                  reduce_axis_serial[2]]}

        if not kd_reduce_flag:
            sch[c_col].emit_insn(
                c_col_deep_inner, "mad", mad_dict_originovrlap)
        elif stride_d == 1:
            sch[c_col].emit_insn(
                c_col_deep_inner, "mad", mad_dict_stride1)
        else:
            sch[c_col].emit_insn(
                c_col_deep_inner, "mad", mad_dict)

        sch[c_ub].reused_by(c_fill_zero, c_ub_vn)
        sch[c_fill_zero].emit_insn(sch[c_fill_zero].op.axis[0], "vector_dup")
        sch[c_ub_vn].emit_insn(sch[c_ub_vn].op.axis[0], "phony_insn")
        sch[c_ub].emit_insn(sch[c_ub].op.axis[0], "dma_copy")
        sch[c_ddr].emit_insn(cddr_n_for_cub, "dma_copy")

    def _redefine_doublebuffer():
        nonlocal al1_pbuffer, bl1_pbuffer, bl0_pbuffer
        if tiling.get("AL1_shape") == []:
            al1_pbuffer = 1
        if tiling.get("BL1_shape") == []:
            bl1_pbuffer = 1
        if tiling.get("BL0_matrix") == []:
            bl0_pbuffer = 1
        return al1_pbuffer, bl1_pbuffer, bl0_pbuffer

    def _get_cub_fuse_num(tensor_map, color_op):
        fuse_num = 0
        for op in color_op.input_ops:
            if "_Before" in op["op"]:
                continue
            fuse_num += 1

        # add c_ub_exact_hw
        fuse_num = fuse_num + 1
        # subtract filters and dx_filing_zero
        fuse_num = fuse_num - 2

        return fuse_num

    def _get_op_infor(color_op):
        tensor_map = {}
        tag_map = {"conv3d_backprop_input_dx_filing_zero": "c_fill_zero",
                   "conv3d_backprop_input_c_ub": "c_ub",
                   "conv3d_backprop_input_c_ub_vn": "c_ub_vn",
                   "conv3d_backprop_input_mad": "c_col",
                   "conv3d_backprop_input_im2col_fractal": "a_col",
                   "conv3d_backprop_input_im2col_fractal_v2": "a_col",
                   "conv3d_backprop_input_w_col": "b_col",
                   "conv3d_backprop_input_im2col_row_major": "a_col_before",
                   "conv3d_backprop_input_w_l1": "b_l1",
                   "conv3d_backprop_input_dy_l1": "a_l1",
                   "conv3d_backprop_input_dy_l1_s1": "a_l1",
                   "conv3d_backprop_input_dy_filling": "a_filling",
                   "conv3d_backprop_input_dy_zero_bp_A_Before": "a_zero",
                   "conv3d_backprop_input_dy_zero": "a_zero",
                   "conv3d_backprop_input_dy_vn": "a_vn",
                   "bias_add_vector": "bias_add_vector",
                   "elewise_binary_mul_bp_A_Before": "elewise_mul_before",
                   "mean_matrix_init_Before": "a_ub_init",
                   "mean_matrix_fp16_Before": "a_ub_fp16",
                   "mean_matrix_mul_bp_A_Before": "a_ub_mul",
                   "c_ub_exact_hw": "c_ub_exact_hw"}

        for op in color_op.body_ops:
            if op["op"] in tag_map.keys():
                tensor_map[tag_map[op["op"]]] = op["dst_buffer"]
            if "mean_matrix_" in op["op"]:
                continue

            if "conv3d_backprop_input_" not in op["op"] and op["next_op"]:
                sch[op["dst_buffer"]].set_scope(tbe_platform_info.scope_ubuf)

        for op in color_op.input_ops:
            if op["op"] in tag_map.keys():
                tensor_map[tag_map[op["op"]]] = op["dst_buffer"]
            if "mean_matrix_" in op["op"]:
                continue

            tmp_read_map = []
            for nop in op["next_op"]:
                if (nop["op"] in _FUSION_NODE_WHITELIST):
                    continue
                else:
                    tmp_read_map.append(nop["dst_buffer"])
            if tmp_read_map:
                tmp_cache_buffer = sch.cache_read(op["dst_buffer"],
                                                  tbe_platform_info.scope_ubuf,
                                                  list(set(tmp_read_map)))
                op["cache_buffer"] = tmp_cache_buffer

        return tensor_map, color_op.body_ops, color_op.input_ops

    def _emit_insn_fusion_op():
        for tensor in aub_body_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "vector_auto")

        for tensor in cub_body_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "vector_auto")

        for tensor in aub_input_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")

        for tensor in cub_input_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")

    def _fusion_op_compute_at():
        for tensor in aub_body_tensors:
            sch[tensor].compute_at(sch[compute_at_buffer[0]],
                                   compute_at_axis[0])

        for tensor in aub_input_tensors:
            sch[tensor].compute_at(sch[compute_at_buffer[0]],
                                   compute_at_axis[0])

        for tensor in cub_body_tensors:
            sch[tensor].compute_at(sch[compute_at_buffer[1]],
                                   compute_at_axis[1])

        for tensor in cub_input_tensors:
            sch[tensor].compute_at(sch[compute_at_buffer[1]],
                                   compute_at_axis[1])

    def _get_ubfusion_tensors():
        aub_input_tensors = []
        cub_input_tensors = []
        aub_body_tensors = []
        cub_body_tensors = []
        for lop in body_ops:
            if "conv3d_backprop_input_" in lop["op"] or "mean_matrix_" in lop["op"]:
                continue

            if "_Before" in lop["op"]:
                aub_body_tensors.append(lop["dst_buffer"])
                continue

            if lop["next_op"]:
                cub_body_tensors.append(lop["dst_buffer"])

        for lop in input_ops:
            if "conv3d_backprop_input_" in lop["op"] or \
               "cache_buffer" not in lop.keys() or \
               "mean_matrix_" in lop["op"]:
                continue

            if "_Before" in lop["op"]:
                aub_input_tensors.append(lop["cache_buffer"])
                continue

            if lop["next_op"][0]["op"] in _FUSION_NODE_WHITELIST:
                continue

            cub_input_tensors.append(lop["cache_buffer"])

        return aub_input_tensors, cub_input_tensors, aub_body_tensors, cub_body_tensors

    c_ddr = tensor
    sch = sch_list[0]
    color_op = AutoScheduleOp(c_ddr)
    var_map = _get_var_map(var_range)
    tensor_map, body_ops, input_ops = _get_op_infor(color_op)
    aub_fusion_flag = True if "elewise_mul_before" in tensor_map.keys() else False
    cub_fusion_flag = True if "c_ub_exact_hw" in tensor_map.keys() else False

    if cub_fusion_flag:
        c_ub_exact_hw = tensor_map["c_ub_exact_hw"]
        res_ub = sch.cache_write(c_ddr, tbe_platform_info.scope_ubuf)
        body_ops[0]["next_op"] = color_op.output_ops
        body_ops[0]["dst_buffer"] = res_ub
        cub_fused_num = _get_cub_fuse_num(tensor_map, color_op)
    else:
        c_ub_exact_hw = c_ddr
        cub_fused_num = 0

    aub_input_tensors, cub_input_tensors, \
        aub_body_tensors, cub_body_tensors = _get_ubfusion_tensors()
    mean_flag = False
    if "a_ub_init" in tensor_map.keys():
        aub_fusion_flag = True
        mean_flag = True
        mean_matrix_init = tensor_map.get("a_ub_init")
        mean_matrix_mul = tensor_map.get("a_ub_mul")
        mean_matrix_fp16 = tensor_map.get("a_ub_fp16")

    tensor_attr, group_dict = _fetch_tensor_info(var_map)
    c_ub_vn = tensor_map.get("c_ub_vn")
    c_ub = tensor_map.get("c_ub")
    c_col = tensor_map.get("c_col")
    a_col = tensor_map.get("a_col")
    b_col = tensor_map.get("b_col")
    b_ddr = tensor_map.get("b_ddr")
    a_col_before = tensor_map.get("a_col_before")
    a_l1 = tensor_map.get("a_l1")
    a_vn = tensor_map.get("a_vn")
    a_filling = tensor_map.get("a_filling")
    a_zero = tensor_map.get("a_zero")
    a_ddr = tensor_map.get("a_ddr")
    b_l1 = tensor_map.get("b_l1")
    a_ub = tensor_map.get("a_ub")
    output_shape = tensor_attr.get("output_shape")
    padding = tensor_attr.get("padding")
    padding_var = tensor_attr.get("padding_var")
    stride_h = tensor_attr.get("stride_h")
    stride_w = tensor_attr.get("stride_w")
    stride_d = tensor_attr.get("stride_d")
    b_ddr_kd = tensor_attr.get("kernel_d")
    bias_add_vector = tensor_map.get("bias_add_vector")
    c_fill_zero = tensor_map.get("c_fill_zero")
    _, dilation_h, dilation_w = tensor_map.get("dilation")

    # =========================tiling_query======================#
    real_g = group_dict["real_g"].value
    cout_g = group_dict["cout_g"].value
    cin1_g = group_dict["cin1_g"].value
    cout1_g = cout_g // tbe_platform.CUBE_MKN[b_ddr.dtype]["mac"][2]

    padu, padd, padl, padr = padding
    padu_var, padd_var, padl_var, padr_var = padding_var
    pad_head, pad_tail = cube_util.shape_to_list(c_ub_exact_hw.op.attrs["depth_pad"])
    tensor_attr['pad_head'] = pad_head
    tensor_attr['pad_tail'] = pad_tail
    if not var_map:
        _, _, _, _, _, kernel_h, kernel_w, _ = list(i.value for i in a_col_before.shape)
    else:
        kernel_h = tensor_attr.get("kernel_h")
        kernel_w = tensor_attr.get("kernel_w")

    _, _, _, ho_l1, wo_l1, _ = cube_util.shape_to_list(a_l1.shape)
    img_shape = cube_util.shape_to_list(a_ddr.shape)
    _, dy_depth, dy_cout1, _, dy_w, _ = img_shape

    b_ddr_n1, b_ddr_k1, b_ddr_k0, b_ddr_n0 = list(i.value for i in b_ddr.shape)

    b_group = real_g

    filter_shape = [b_ddr_k1 * b_ddr_k0,
                    b_ddr_kd, b_ddr_n1 // (kernel_h * kernel_w * real_g * b_ddr_kd),
                    kernel_h, kernel_w, b_ddr_n0]

    cddr_batch, cddr_depth, cddr_c1, cddr_h, cddr_w, cdder_c0 = output_shape
    tiling_output = [cddr_batch,
                     cddr_depth, cddr_h, cddr_w, cddr_c1 * cdder_c0]
    kd_reduce_flag = bool(len(c_col.op.reduce_axis) == _NUM_3)
    aub_fused_num = 1 if aub_fusion_flag else 0

    tiling_img_shape = img_shape
    tiling_img_shape[2] = cout1_g
    tiling_filter_shape = [cout_g, b_ddr_kd, cin1_g, kernel_h, kernel_w, b_ddr_n0]
    tiling_output_shape = tiling_output

    if not var_map:
        info_dict = {
            "a_shape": tiling_img_shape,
            "b_shape": tiling_filter_shape,
            "c_shape": tiling_output_shape,
            "a_dtype": 'float16',
            "b_dtype": 'float16',
            "c_dtype": 'float16',
            "mad_dtype": 'float32',
            "pad": [pad_head, pad_tail, padu, padd, padl, padr],
            "stride": [stride_d, 1, 1],
            "strideh_expand": stride_h,
            "stridew_expand": stride_w,
            "dilation": [1, dilation_h, dilation_w],
            "group": real_g,
            "fused_coefficient": [aub_fused_num, 0, cub_fused_num],
            "bias_flag": False,
            "op_type": "conv3d_backprop_input",
            "kernel_name": c_ub_exact_hw.op.attrs["kernel_name"].value
        }
        tiling = get_tiling(info_dict)
    else:
        tiling = tiling_case

    if tiling["AL0_matrix"][2] == _DEFAULT_TILING_FLAG:
        tiling = _default_tiling()

    if stride_w == 1 and stride_h == 1 and aub_fusion_flag == False:
        tiling['AUB_shape'] = None
    _, n_dim, m_dim, _ = tiling.get("block_dim")
    aub_pbuffer = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
    al1_pbuffer = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
    bl1_pbuffer = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
    al0_pbuffer = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
    bl0_pbuffer = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
    l0c_pbuffer = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
    cub_pbuffer = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
    al1_pbuffer, bl1_pbuffer, bl0_pbuffer = _redefine_doublebuffer()

    _, _, al1_co1, _, _, al1_co0 = cube_util.shape_to_list(a_l1.shape)
    _, _, _, _, c_l0c_hw, _ = cube_util.shape_to_list(c_col.shape)
    _, _, bl1_k1, bl1_co1, bl1_co0, _ = list(i.value for i in b_l1.shape)
    _, a_col_batch, _, a_col_ma, a_col_ka, _, _ = cube_util.shape_to_list(a_col.shape)
    cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, _, _, _ = tiling.get("CUB_matrix")
    cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, _, _, _ = tiling.get("CL0_matrix")
    al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, _, _, al0_tiling_dfactor = tiling.get("AL0_matrix")
    bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_kd = _tiling_l0_process()
    al1_tiling_k, al1_tiling_m, bl1_tiling_k, bl1_tiling_n, bl1_tiling_kdparts = _tiling_l1_process()

    # tiling_check
    _tiling_check()

    if cub_fusion_flag:
        batch_after_multicore, d_after_multicore = \
            sch[c_ddr].split(c_ddr.op.axis[0], factor=cddr_depth)
        n_after_multicore = c_ddr.op.axis[1]
        m_after_multicore = c_ddr.op.axis[2]
    else:
        batch_after_multicore = c_ddr.op.axis[0]
        d_after_multicore = c_ddr.op.axis[1]
        n_after_multicore = c_ddr.op.axis[2]
        m_after_multicore = c_ddr.op.axis[3]

    # axis management
    g_axis, n_after_multicore = sch[c_ddr].split(n_after_multicore, factor=cin1_g)
    sch[c_ddr].reorder(g_axis,
                       batch_after_multicore,
                       d_after_multicore,
                       n_after_multicore,
                       m_after_multicore)
    # cub
    cddr_n_outer, cddr_m_outer, cddr_n_for_cub = _cub_process()
    # l0c
    cddr_deep_factor = al0_tiling_dfactor
    kd_factor = bl0_tiling_kd

    kd_tiling_l1_factor = bl1_tiling_kdparts
    (batch_outer, al1_at_ddr_m_outer, batch_inner, bl1_at_ddr_n_outer, bl1_at_ddr_n_inner,
     col_at_ddr_axis, cddr_m_outer_inner, _, c_ddr_deep_outer, _) = _l0c_procees()

    c_ddr_deep_outer_value = compute_util.int_ceil_div(cddr_depth, cddr_deep_factor)
    # l0a_l0b
    al0_m_factor = al0_tiling_ma * al0_tiling_m0
    if kd_reduce_flag is False:
        c_col_deep_outer, c_col_deep_inner, c_col_k_outer, c_col_m_outer, c_col_n_outer = _l0a_and_l0b_process()
    else:
        (c_col_deep_outer, c_col_deep_inner, c_col_k_outer, c_col_m_outer,
         c_col_n_outer, reduce_axis_kd, reduce_axis_kd_outer,
         reduce_axis_kd_inner) = _l0a_and_l0b_process()

    # l1a_l1b
    k_al1_factor = al1_tiling_k // al0_tiling_ka // tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0]
    k_bl1_factor = bl1_tiling_k // bl0_tiling_kb // tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0]
    reduce_axis_serial, bl1_at_l0c_axis, al1_at_l0c_axis = _al1_and_bl1_process()

    a_l1_h_outer = None
    if stride_h > 1 or stride_w > 1 or aub_fusion_flag:
        a_l1_h_outer = _aub_process()

    _reorder_management()

    # buffer_align
    if kd_reduce_flag:
        sch[c_col].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0]),
            (1, tbe_platform.CUBE_MKN[c_col.dtype]["mac"][2]),
            (1, 1),
            (1, 1),
            (1, tbe_platform.CUBE_MKN[c_col.dtype]["mac"][1]))
    else:
        sch[c_col].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0]),
            (1, tbe_platform.CUBE_MKN[c_col.dtype]["mac"][2]),
            (1, 1),
            (1, tbe_platform.CUBE_MKN[c_col.dtype]["mac"][1]))

    sch[a_col_before].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (cddr_w, cddr_w),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, tbe_platform.CUBE_MKN[a_col_before.dtype]["mac"][1]))

    sch[c_ub].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][0]),
        (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][2]))
    sch[c_fill_zero].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][0]),
        (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][2]))
    sch[c_ub_vn].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][0]),
        (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][2]))
    if bias_add_vector is not None:
        sch[c_ub_vn].reused_by(bias_add_vector)
        sch[bias_add_vector].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][0]),
            (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][2]))
    (batch_outer, c_ddr_deep_outer, bl1_at_ddr_n_outer, al1_at_ddr_m_outer,
     g_axis, blockidx, blocks) = _multi_core()

    compute_at_buffer = []
    compute_at_axis = []
    compute_at_buffer.append(a_l1)
    compute_at_axis.append(a_l1_h_outer)
    compute_at_buffer.append(c_ddr)
    compute_at_axis.append(cddr_m_outer_inner)

    _do_compute_at()
    _fusion_op_compute_at()

    _double_buffer()

    # emit insn
    _emit_insn_process()
    _emit_insn_fusion_op()

    def _handle_dynamic_workspace(stride_w):
        def _get_al1_m_extent(al0_m):
            al1_h = tvm.select((tvm.floormod(al0_m, output_shape[4]) == 0).asnode(),
                               kernel_h + (al0_m // output_shape[4]) - 1,
                               tvm.select(tvm.any(tvm.floormod(2*al0_m, output_shape[4]) == 0,
                                                  tvm.floormod(output_shape[4], al0_m) == 0),
                                          kernel_h + (al0_m // output_shape[4]),
                                          kernel_h + (al0_m // output_shape[4]) + 1))
            al1_w = a_l1.shape[-2]
            return al1_h, al1_w

        def _get_al0_bound():
            tiling_ma, tiling_ka, tiling_m0, tiling_k0 = tiling["AL0_matrix"][:4]
            estimate_d = (bl0_tiling_kd - 2 + al0_tiling_dfactor + stride_d - 1) // stride_d + 1
            if "dedy_d" in var_map:
                d_factor = tvm.min(estimate_d, dy_depth)
            else:
                d_factor = min(estimate_d, dy_depth)
            return tiling_ma * tiling_ka * tiling_m0 * tiling_k0 * d_factor

        def _get_al1_bound():
            d_factor = _dfactor_dynamic(var_map)
            if tiling["AL1_shape"]:
                al0_m = cl0_tiling_mc * cl0_tiling_m0
                al1_h, al1_w = _get_al1_m_extent(al0_m)
                k_al1 = tiling["AL1_shape"][0]
                al1_c = k_al1 // kernel_h // kernel_w
                al1_bound = al1_c * al1_h * al1_w * d_factor
            else:
                al1_m = compute_util.int_ceil_div(a_l1.shape[3] * a_l1.shape[4], cl0_tiling_m0) * cl0_tiling_m0
                al1_c = cout1_g * al1_co0
                al1_h, al1_w = a_l1.shape[3], a_l1.shape[4]
                al1_bound = al1_c * al1_m * d_factor
            return al1_bound, al1_h

        def _get_bl1_bound():
            if tiling["BL1_shape"]:
                n_bound = tiling["BL1_shape"][1] * tiling["CL0_matrix"][0] * tiling["CL0_matrix"][2]
                k_bound = tiling["BL1_shape"][0]
                if kd_reduce_flag:
                    d_bound = tiling["BL1_shape"][-1] * bl0_tiling_kd
                else:
                    d_bound = b_ddr_kd
                bl1_bound = n_bound * k_bound * d_bound
            else:
                bl1_full_load = cout_g * b_ddr_kd * cin1_g * kernel_h * kernel_w * b_ddr_n0
                bl1_bound = compute_util.int_ceil_div(bl1_full_load, tiling["block_dim"][1])
            return bl1_bound

        def _set_aub_bound(al1_h):
            if stride_h > 1 or stride_w > 1 or aub_fusion_flag:
                d_factor = _dfactor_dynamic(var_map)
                aub_co0 = tbe_platform.CUBE_MKN[c_col.dtype]["mac"][1]
                aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
                aub_co1 = aub_tiling_k // (kernel_h * kernel_w * aub_co0)
                aub_filling_w = dy_w * stride_w
                aub_h = (aub_tiling_m + stride_h - 1) // stride_h + 1
                al1_m = compute_util.int_ceil_div(a_l1.shape[3] * a_l1.shape[4], cl0_tiling_m0) * cl0_tiling_m0
                if tiling["AL1_shape"]:
                    multi_m_al1 = tiling["AL1_shape"][1]
                    al1_m = multi_m_al1 * cl0_tiling_mc * cl0_tiling_m0
                # for aub_tiling_m is 1, then aub_h must be 1
                if aub_tiling_m != 1:
                    aub_h = tvm.select(
                            tvm.all((tvm.floormod(al1_h, stride_h) == 0).asnode(),
                                    (tvm.floormod(al1_m, output_shape[-2]) == 0).asnode()),
                            aub_h,
                            aub_h + 1)
                a_filling_bound = aub_co1 * aub_tiling_m * aub_filling_w * aub_co0 * d_factor
                dedy_bound = aub_co1 * aub_h * dy_w * aub_co0 * d_factor
                if stride_h > 1 or stride_w > 1:
                    sch[a_zero].set_storage_bound(a_filling_bound)
                    sch[a_vn].set_storage_bound(a_filling_bound)
                if mean_flag:
                    sch[mean_matrix_init].set_storage_bound(dedy_bound)
                    sch[mean_matrix_fp16].set_storage_bound(dedy_bound)
                    if stride_h > 1 or stride_w > 1:
                        sch[mean_matrix_mul].set_storage_bound(dedy_bound)
                sch[a_filling].set_storage_bound(a_filling_bound)

        al1_bound, al1_h = _get_al1_bound()
        extent_h = tvm.select(tvm.floordiv(al1_h, ho_l1) < 1,
                              al1_h,
                              al1_h + padu)
        extent_h_var = tvm.var("extent_h")
        sch[a_l1].buffer_tile((None, None), (None, None), (None, None),
                              (None, extent_h_var), (None, None), (None, None))
        sch.set_var_value(extent_h_var, extent_h)
        sch.set_var_range(extent_h_var, 1, None)

        sch[a_l1].set_storage_bound(al1_bound)
        sch[b_l1].set_storage_bound(_get_bl1_bound())
        sch[a_col].set_storage_bound(_get_al0_bound())
        _set_aub_bound(al1_h)

        sch.disable_allocate(tbe_platform_info.scope_cbuf)
        sch.disable_allocate(tbe_platform_info.scope_ca)
        sch.disable_allocate(tbe_platform_info.scope_cb)
        sch.disable_allocate(tbe_platform_info.scope_cc)
        sch.disable_allocate(tbe_platform_info.scope_ubuf)

        sch[a_l1].mem_unique()
        sch[a_col].mem_unique()
        sch[b_l1].mem_unique()
        sch[b_col].mem_unique()
        sch[c_col].mem_unique()

    if var_map:
        _handle_dynamic_workspace(stride_w)
    return sch


class AutoScheduleDict(dict):
    """
    AutoScheduleDict
    """

    def __init__(self, **kwargs):
        super(AutoScheduleDict, self).__init__(**kwargs)
        self.read_only = False


class AutoScheduleOp:
    """
    AutoScheduleOp
    """

    def __init__(self, *init_args):
        if len(init_args) == 1 and isinstance(init_args[0], tvm.tensor.Tensor):
            res_tensor = init_args[0]
            self._color_count = 0
            self._op = []
            self.body_ops = []
            self.input_ops = []
            self.output_ops = []
            self._res_tensor = res_tensor
            self._before_conv_flag = False
            self.__scrapy_tensor_graph(self._res_tensor)
            self.__connect_op()
            self._end_op = self._get_op_by_tensor(self._res_tensor)
            self._end_op["color"] = self._color_count
            self.__init_color(self._end_op)
            self.__analyse_input_output()

    def __split_tensor(self, tensor):
        tmp_op = AutoScheduleDict()
        operator = tensor.op
        if hasattr(operator, "tag"):
            if operator.tag == "":
                tmp_op["op"] = operator.name
            else:
                tmp_op["op"] = operator.tag
        if tmp_op["op"].find("|") != -1:
            str_list = operator.tag.split("|")
            tmp_op["op"] = str_list[0]
        if hasattr(tensor, "tag"):
            tmp_op["op"] = tmp_op["op"] + "_" + tensor.tag
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(operator.input_tensors)

        if "bp_A" in tmp_op["op"] and "conv3d_backprop_input_" not in tmp_op["op"]:
            self._before_conv_flag = True
        if self._before_conv_flag:
            tmp_op["op"] = tmp_op["op"] + "_Before"

        return tmp_op

    def __scrapy_tensor_graph(self, res_tensor):
        operation_list = [res_tensor]
        while operation_list:
            tmp_operation_list = []
            for operation in operation_list:
                tmp_op = self.__split_tensor(operation)
                self._op.append(tmp_op)
                for i in tmp_op["src_buffer"]:
                    i.next = operation
                    operation.prev = i
                    tmp_operation_list.append(i)
                    if tmp_op["op"] == "conv3d_backprop_input_dy_l1_s1" or \
                       tmp_op["op"] == "conv3d_backprop_input_dy_filling":
                        i.tag = "bp_A"

            operation_list = list(set(tmp_operation_list))

    def __connect_op(self):
        for lop in self._op:
            lop["prev_op"] = []
            lop["next_op"] = []

        for lop in self._op:
            for src_tensor in lop["src_buffer"]:
                tmp_op = self._get_op_by_tensor(src_tensor)
                lop["prev_op"].append(tmp_op)
                tmp_op["next_op"].append(lop)

    def __init_color(self, start_op):
        for p_op in start_op["prev_op"]:
            p_op["color"] = start_op["color"]
            self.__init_color(p_op)

    def _get_op_by_tensor(self, src_tensor):
        for i in self._op:
            if i["dst_buffer"].same_as(src_tensor):
                return i
        return None

    def __analyse_input_output(self):
        input_ops = []
        output_ops = []
        body_ops = []
        input_tensor_name = []
        body_tensor_name = []
        for lop in self._op:
            if not lop["prev_op"]:
                lop["color"] = -1
                if lop["dst_buffer"].name not in input_tensor_name:
                    input_ops.append(lop)
                    input_tensor_name.append(lop["dst_buffer"].name)
                else:
                    continue
            else:
                if lop["dst_buffer"].name not in body_tensor_name:
                    body_ops.append(lop)
                    body_tensor_name.append(lop["dst_buffer"].name)
                else:
                    continue
                if not lop["next_op"]:
                    output_ops.append(lop)

        for i in input_ops:
            i["color"] = -1
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.body_ops = body_ops
