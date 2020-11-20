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
import te.platform as tbe_platform
from te.lang.cce.te_compute import util as te_util
from te.domain.tiling.tiling_query import tiling_query
from te.utils.error_manager import error_manager_util
from te import tvm


NUM_3 = 3
DEFAULT_TILING_FLAG = 32


def general_schedule(tensor, sch_list):  # pylint:disable=R0914, R0915
    """
    auto_schedule for cce AI-CORE.
    For now, only one convolution operation is supported.

    Parameters
    ----------
    sch_list: use sch_list[0] to return conv schedule

    Returns
    -------
    True for sucess, False for no schedule
    """
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
            reduce_axis_kd_outer_outer, _ = sch[c_col].split(reduce_axis_kd_outer, kd_tiling_l1_factor)
        if k_al1_factor > k_bl1_factor:
            factor_outer, factor_inner = k_al1_factor//k_bl1_factor, k_bl1_factor
            c_col_k_outer_outer, c_col_k_outer_inner = sch[c_col].split(c_col_k_outer, factor=factor_inner)
            c_col_k_outer_outer_outer, c_col_k_outer_outer_inner = sch[c_col].split(c_col_k_outer_outer,
                                                                                    factor=factor_outer)
            bl1_at_l0c_axis, al1_at_l0c_axis = c_col_k_outer_outer_inner, c_col_k_outer_outer_outer
            if kd_reduce_flag:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer,
                    reduce_axis_kd_outer_outer,
                    c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)
            else:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)

        else:
            factor_outer, factor_inner = k_bl1_factor//k_al1_factor, k_al1_factor
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
        aub_tiling_k_factor, aub_tiling_m_factor = aub_tiling_k // (kernel_h * kernel_w * 16), aub_tiling_m
        _, _, _, _, aub_w, _ = list(i.value for i in a_filling.shape)
        a_l1_k_outer, a_l1_k_inner = sch[a_l1].split(sch[a_l1].op.axis[2], factor=aub_tiling_k_factor)
        a_l1_h_outer, a_l1_h_inner = sch[a_l1].split(sch[a_l1].op.axis[3], factor=aub_tiling_m_factor)
        a_l1_w_outer, a_l1_w_inner = sch[a_l1].split(sch[a_l1].op.axis[4], factor=aub_w)
        sch[a_l1].reorder(a_l1_k_outer, a_l1_h_outer, a_l1_w_outer,
                          sch[a_l1].op.axis[0], sch[a_l1].op.axis[1],
                          a_l1_k_inner, a_l1_h_inner, a_l1_w_inner)
        return a_l1_h_outer

    def _multi_core():  # pylint:disable=R0914
        nonlocal g_axis
        block_dim = tiling['block_dim']
        group_dim = tiling['g_dim']
        batch_dim, n_dim, m_dim, d_dim = block_dim
        blocks = group_dim * batch_dim * n_dim * m_dim * d_dim

        if blocks != 1:
            multicore_batch, batch_outer_inner = sch[c_ddr].split(batch_outer, nparts=batch_dim)
            multicore_d, c_ddr_deep_outer_inner = sch[c_ddr].split(c_ddr_deep_outer, nparts=d_dim)
            # split n axis
            multicore_n, bl1_at_ddr_n_outer_inner = sch[c_ddr].split(bl1_at_ddr_n_outer, nparts=n_dim)
            # split m axis
            multicore_m, al1_at_ddr_m_outer_inner = sch[c_ddr].split(al1_at_ddr_m_outer, nparts=m_dim)
            # split g axis
            multicore_group, g_axis = sch[c_ddr].split(g_axis, nparts=group_dim)
            # reorder
            sch[c_ddr].reorder(
                multicore_group,
                multicore_batch, multicore_d,
                multicore_n, multicore_m, g_axis,
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
            bind_out, bind_in = sch[c_ddr].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_ddr].bind(bind_out, blockidx)
        else:
            batch_outer_outer, batch_outer_inner = sch[c_ddr].split(batch_outer, nparts=1)
            bind_out, bind_in = sch[c_ddr].split(batch_outer_outer, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_ddr].bind(bind_out, blockidx)

            c_ddr_deep_outer_inner = c_ddr_deep_outer
            bl1_at_ddr_n_outer_inner = bl1_at_ddr_n_outer
            al1_at_ddr_m_outer_inner = al1_at_ddr_m_outer
        return (batch_outer_inner, c_ddr_deep_outer_inner, bl1_at_ddr_n_outer_inner, al1_at_ddr_m_outer_inner,
            g_axis, bind_in, blockidx, blocks)

    def _tiling_check():
        _tiling_check_none()
        _tiling_check_value()
        _tiling_check_factor()
        _tiling_check_pbuffer()
        if stride_h == 1 and stride_w == 1:
            if tiling.get("AUB_shape") is not None:
                dict_args = {
                    'errCode': 'E62306',
                    'desc': 'stride = 1 but AUB_shape is not None.',
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

        if tiling.get("BL0_matrix") == [] and tiling.get("BL1_shape") != []:
            dict_args = {
                'errCode': 'E62306',
                'desc': 'BL0 full load but BL1 not!',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _tiling_check_value():
        if tiling.get("BL0_matrix"):
            if al0_tiling_ka != bl0_tiling_kb:
                dict_args = {
                    'errCode': 'E62306',
                    'desc': 'in BL0_matrix, ka != kb',
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if bl0_tiling_nb != cl0_tiling_nc:
                dict_args = {
                    'errCode': 'E62306',
                    'desc': 'in BL0_matrix, nb != nc.',
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(
                    dict_args))

        if al0_tiling_ma != cl0_tiling_mc:
            dict_args = {
                'errCode': 'E62306',
                'desc': 'ma != mc.',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _tiling_check_none():
        if ((tiling.get("AL1_shape") is None) or
            (tiling.get("BL1_shape") is None) or
            (tiling.get("CUB_matrix") is None)):
            dict_args = {
                'errCode': 'E62305',
                'param_name': 'AL1_shape/BL1_shape/CUB_matrix',
                'expect_value': 'not None',
                'value': 'None'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if ((tiling.get("AL0_matrix") is None) or
            (tiling.get("BL0_matrix") is None) or
            (tiling.get("CL0_matrix") is None)):
            dict_args = {
                'errCode': 'E62305',
                'param_name': 'AL0_matrix/BL0_matrix/CL0_matrix',
                'expect_value': 'not None',
                'value': 'None'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if (tiling["BUB_shape"] is None
            or tiling["BUB_shape"][0] is None
            or tiling["BUB_shape"][0] == 0):
            tiling["g_dim"] = 1
        else:
            tiling["g_dim"] = tiling["BUB_shape"][0]

    def _tiling_check_factor():
        if (kernel_w * kernel_h * cout1_g) % al0_tiling_ka != 0:
            dict_args = {
                'errCode': 'E62305',
                'param_name': 'Co1*Hk*Wk % ka',
                'expect_value': '0',
                'value': str((kernel_w * kernel_h * dy_cout1) % al0_tiling_ka)
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if al1_tiling_k % al0_tiling_ka != 0:
            dict_args = {
                'errCode': 'E62305',
                'param_name': 'k_AL1 % ka',
                'expect_value': '0',
                'value': str(al1_tiling_k % al0_tiling_ka)
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
            if bl1_tiling_k % bl0_tiling_kb != 0:
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'k_BL1 % kb',
                    'expect_value': '0',
                    'value': str(bl1_tiling_k % bl0_tiling_kb)
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

        if cl0_tiling_nc % cub_tiling_nc_factor != 0:
            dict_args = {
                'errCode': 'E62305',
                'param_name': 'nc % nc_factor',
                'expect_value': '0',
                'value': str(cl0_tiling_nc % cub_tiling_nc_factor)
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if tiling.get("BL1_shape"):
            if al1_tiling_k > bl1_tiling_k and al1_tiling_k % bl1_tiling_k != 0:
                dict_args = {
                    'errCode': 'E62306',
                    'desc': 'k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.'
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if bl1_tiling_k > al1_tiling_k and bl1_tiling_k % al1_tiling_k != 0:
                dict_args = {
                    'errCode': 'E62306',
                    'desc': 'k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.'
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

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
                    dict_args = {
                        'errCode': 'E62305',
                        'param_name': name,
                        'expect_value': '1 or 2',
                        'value': str(flag)
                    }
                    raise RuntimeError(dict_args,
                                    error_manager_util.get_error_message(dict_args))

    def _fetch_tensor_info():  # pylint:disable=R0914, R0915
        tensor_map = {}
        tensor_attr = {}
        stride_d = c_ddr.op.attrs["stride_d"].value
        group_dict = c_ddr.op.attrs["group_dict"]
        kernel_d, _, _ = list(i.value for i in c_ddr.op.attrs["kernels"])
        c_fill_zero = c_ddr.op.input_tensors[0]
        c_ub = c_fill_zero.op.input_tensors[0]
        tensor_map['c_fill_zero'] = c_fill_zero
        sch[c_fill_zero].set_scope(tbe_platform.scope_ubuf)
        c_col = c_ub.op.input_tensors[0]
        a_col = c_col.op.input_tensors[0]  # im2col_fractal in L0A
        b_col = c_col.op.input_tensors[1]  # weight_transform in L0B
        b_l1 = b_col.op.input_tensors[0]
        sch[b_l1].set_scope(tbe_platform.scope_cbuf)
        b_ddr = b_l1.op.input_tensors[0]  # weight in ddr
        a_col_before = a_col.op.input_tensors[0]  # im2col_row_major in L1

        tensor_map['c_ub'] = c_ub
        tensor_map['c_col'] = c_col
        tensor_map['a_col'] = a_col
        tensor_map['b_col'] = b_col
        tensor_map['b_l1'] = b_l1
        tensor_map['b_ddr'] = b_ddr
        tensor_map['a_col_before'] = a_col_before

        # stride > 1
        if (a_col_before.op.input_tensors[0].op.tag == "dy_l1" or
                a_col_before.op.input_tensors[0].op.tag == "dy_l1_cut"):

            a_l1 = a_col_before.op.input_tensors[0]
            a_filling = a_l1.op.input_tensors[0]

            stride_h, stride_w = list(i.value for i in
                                      a_filling.op.attrs["stride_expand"])
            a_ddr = a_filling.op.input_tensors[0]  # dEdY in ddr
            a_zero = a_filling.op.input_tensors[1]  # dEdY_zero in ub
            tensor_map['a_l1'] = a_l1
            tensor_map['a_filling'] = a_filling
            tensor_map['a_ddr'] = a_ddr
            tensor_map['a_zero'] = a_zero
        else:
            a_ddr = a_col_before.op.input_tensors[0]  # dEdY in ddr
            stride_h = 1
            stride_w = 1
            tensor_map['a_ddr'] = a_ddr
        tensor_attr['stride_w'] = stride_w
        tensor_attr['stride_h'] = stride_h
        # dataflow management
        sch[b_col].set_scope(tbe_platform.scope_cb)
        if stride_h == 1 and stride_w == 1:
            a_l1 = sch.cache_read(a_ddr, tbe_platform.scope_cbuf, [a_col_before])
            tensor_map['a_l1'] = a_l1
        else:
            a_ub = sch.cache_read(a_ddr, tbe_platform.scope_ubuf, [a_filling])
            tensor_map['a_ub'] = a_ub
            # generate a_zero in ub
            sch[a_zero].set_scope(tbe_platform.scope_ubuf)
            sch[a_filling].set_scope(tbe_platform.scope_ubuf)
            # dma : a_filling ub------>L1
            sch[a_l1].set_scope(tbe_platform.scope_cbuf)

        sch[a_col_before].set_scope(tbe_platform.scope_cbuf)
        sch[a_col].set_scope(tbe_platform.scope_ca)

        sch[c_col].set_scope(tbe_platform.scope_cc)
        sch[c_ub].set_scope(tbe_platform.scope_ubuf)
        padding = list(i.value for i in a_col_before.op.attrs["padding"])
        output_shape = list(i.value for i in c_ddr.op.attrs["output_shape"])
        tensor_attr['padding'] = padding
        tensor_attr['output_shape'] = output_shape
        tensor_attr['stride_d'] = stride_d
        tensor_attr['kernel_d'] = kernel_d
        return tensor_map, tensor_attr, group_dict

    def _tiling_l0_process():
        if al0_tiling_ma == a_col_ma and al0_tiling_ka == a_col_ka and a_col_batch == 1:
            tiling["AL0_matrix"] = []
        if tiling.get("BL0_matrix"):
            bl0_tiling_kb, bl0_tiling_nb, _, _, _, bl0_tiling_kd = tiling.get("BL0_matrix")
        else:
            bl0_tiling_group, bl0_tiling_kd, bl0_tiling_kb, bl0_tiling_nb, _, _ = list(i.value for i in b_col.shape)
            bl0_tiling_nb = bl0_tiling_nb // n_dim
        return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_kd

    def _tiling_l1_process():
        if tiling.get("AL1_shape"):
            al1_tiling_k, al1_tiling_m, _, _ = tiling.get("AL1_shape")
            if (al1_tiling_k == kernel_h * kernel_w * cout1_g * al1_co0 and al1_tiling_m == te_util.int_ceil_div(
                c_l0c_hw, (tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0]*cl0_tiling_mc))):
                tiling["AL1_shape"] = []
        else:
            # batch = 1 other axes full load
            al1_tiling_k = kernel_h * kernel_w * cout1_g * al1_co0
            al1_tiling_m = c_l0c_hw // (tbe_platform.CUBE_MKN[c_col.dtype]["mac"][0] *
                                        cl0_tiling_mc) // m_dim
        if tiling.get("BL1_shape"):
            bl1_tiling_k, bl1_tiling_n, _, bl1_tiling_kdparts = tiling.get("BL1_shape")
        else:
            bl1_tiling_k = kernel_h * kernel_w * bl1_co0 * bl1_co1
            bl1_tiling_n = bl1_k1 // (kernel_h *
                                       kernel_w * cl0_tiling_nc) // n_dim
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

    def _get_h_l1(howo_size):
        left = 0
        right = 0
        max_dis = 0

        for x in range(1, te_util.int_ceil_div(cddr_h * cddr_w, howo_size)):
            m_length = x * howo_size
            right = m_length // wo_l1
            distance = right - left + 1 if (m_length % wo_l1 != 0) else right - left
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

        d_factor = min((b_ddr_kd - 2 + al0_tiling_dfactor + stride_d - 1) // stride_d + 1, dy_depth)
        if b_ddr_kd == stride_d:
            d_factor = max(d_factor - 1, 1)
        h_l1 = _get_h_l1(howo_size)
        dy_l1_size = d_factor * dy_cout1 * h_l1 * (wo_l1 + padl + padr) * c0_size * 2

        if (dy_l1_size + b_l1_size) > tbe_platform.get_soc_spec("L1_SIZE"):
            return True

        return False

    def _check_exceed_ub_buffer():
        c0_size = 16
        aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
        aub_tiling_k_factor, aub_tiling_m_factor = aub_tiling_k // (kernel_h * kernel_w * 16), aub_tiling_m
        d_factor = min((b_ddr_kd - 2 + al0_tiling_dfactor + stride_d - 1) // stride_d + 1, dy_depth)
        if b_ddr_kd == stride_d:
            d_factor = max(d_factor - 1, 1)

        dedy_ub_size = (d_factor * aub_tiling_k_factor * dy_w * c0_size * 2 *
                        te_util.int_ceil_div(aub_tiling_m_factor, stride_h))
        dy_filing_size = d_factor * aub_tiling_k_factor * aub_tiling_m_factor * (dy_w * stride_w) * c0_size * 2
        c_ub_size = cub_tiling_nc_factor * cub_tiling_mc_factor * c0_size**2 * cub_pbuffer * 2

        ub_size = tbe_platform.get_soc_spec("UB_SIZE")
        if (dedy_ub_size + dy_filing_size + c_ub_size) > ub_size:
            return True

        return False

    def _check_exceed_buffer(howo_size):
        if _check_exceed_l1_buffer(howo_size):
            return True

        if stride_h > 1 or stride_w > 1:
            if _check_exceed_ub_buffer():
                return True

        return False

    def _do_compute_at():
        m_dim = tiling['block_dim'][2]
        howo_out = cddr_h * cddr_w
        howo_deep_outer = te_util.int_ceil_div(howo_out, m_dim)
        howo_m_outer = al1_tiling_m * al0_tiling_ma * al0_tiling_m0

        if (not tiling['AL1_shape'] and not
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
            sch[b_l1].compute_at(sch[c_ddr], bind_in)
        elif (bl1_tiling_k == kernel_h * kernel_w * bl1_co0 * bl1_co1 and
              kd_factor * kd_tiling_l1_factor == b_ddr_kd):
            sch[b_l1].compute_at(sch[c_ddr], bl1_at_ddr_n_outer)
        else:
            sch[b_l1].compute_at(sch[c_col], bl1_at_l0c_axis)
        sch[c_ub].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_fill_zero].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_col].compute_at(sch[c_ddr], col_at_ddr_axis)
        sch[a_col].compute_at(sch[c_col], c_col_m_outer)
        if not tiling['BL0_matrix'] and not tiling['BL1_shape']:
            sch[b_col].compute_at(sch[c_ddr], c_ddr_deep_outer)
        else:
            sch[b_col].compute_at(sch[c_col], c_col_n_outer)
        if stride_h > 1 or stride_w > 1:
            sch[a_filling].compute_at(sch[a_l1], a_l1_h_outer)
            sch[a_zero].compute_at(sch[a_l1], a_l1_h_outer)
            sch[a_ub].compute_at(sch[a_l1], a_l1_h_outer)

    def _double_buffer():
        if stride_h > 1 or stride_w > 1:
            if aub_pbuffer == 2:
                sch[a_ub].double_buffer()
                sch[a_filling].double_buffer()
                sch[a_zero].double_buffer()

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

    def _default_tiling():
        tiling = {}
        # defaut value 16
        k0_size = tbe_platform.CUBE_MKN[a_ddr.dtype]["mac"][1]
        k_al1 = kernel_h * kernel_w * k0_size

        if stride_h > 1 or stride_w > 1:
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

        if stride_h == 1 and stride_w == 1:
            sch[a_l1].emit_insn(sch[a_l1].op.axis[0], "dma_copy")
        else:
            sch[a_ub].emit_insn(sch[a_ub].op.axis[0], "dma_copy")
            afill_n, afill_d, afill_c, afill_h, afill_w, _ = sch[a_filling].op.axis

            afill_w_out, afill_w_inner = sch[a_filling].split(
                afill_w, factor=stride_w)
            sch[a_filling].reorder(
                afill_w_inner,
                afill_n,
                afill_d,
                afill_c,
                afill_h,
                afill_w_out)
            sch[a_filling].unroll(afill_w_inner)
            sch[a_filling].reused_by(a_zero)
            sch[a_zero].emit_insn(sch[a_zero].op.axis[0], "vector_dup")
            sch[a_filling].emit_insn(afill_n, "vector_muls")
            sch[a_l1].emit_insn(sch[a_l1].op.axis[0], "dma_copy")

        setfmatrix_dict = {"conv_kernel_h": kernel_h,
                           "conv_kernel_w": kernel_w,
                           "conv_padding_top": padu,
                           "conv_padding_bottom": padd,
                           "conv_padding_left": padl,
                           "conv_padding_right": padr,
                           "conv_stride_h": 1,
                           "conv_stride_w": 1,
                           "conv_fm_c": a_l1.shape[2] * a_l1.shape[5],
                           "conv_fm_h": a_l1.shape[3],
                           "conv_fm_w": a_l1.shape[4]}

        sch[a_col_before].emit_insn(a_col_before.op.axis[2],
                                    'set_fmatrix', setfmatrix_dict)
        _, a_col_deep_inner = sch[a_col].split(sch[a_col].op.axis[2], factor=1)
        sch[a_col].emit_insn(a_col_deep_inner, 'im2col')
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

        sch[c_fill_zero].reused_by(c_ub)
        sch[c_fill_zero].emit_insn(sch[c_fill_zero].op.axis[0], "vector_dup")
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

    c_ddr = tensor
    sch = sch_list[0]
    tensor_map, tensor_attr, group_dict = _fetch_tensor_info()
    c_ub = tensor_map.get("c_ub")
    c_col = tensor_map.get("c_col")
    a_col = tensor_map.get("a_col")
    b_col = tensor_map.get("b_col")
    b_ddr = tensor_map.get("b_ddr")
    a_col_before = tensor_map.get("a_col_before")
    a_l1 = tensor_map.get("a_l1")
    a_filling = tensor_map.get("a_filling")
    a_zero = tensor_map.get("a_zero")
    a_ddr = tensor_map.get("a_ddr")
    b_l1 = tensor_map.get("b_l1")
    a_ub = tensor_map.get("a_ub")
    output_shape = tensor_attr.get("output_shape")
    padding = tensor_attr.get("padding")
    stride_h = tensor_attr.get("stride_h")
    stride_w = tensor_attr.get("stride_w")
    stride_d = tensor_attr.get("stride_d")
    b_ddr_kd = tensor_attr.get("kernel_d")
    c_fill_zero = tensor_map.get("c_fill_zero")

    # =========================tiling_query======================#
    real_g = group_dict["real_g"].value
    cout_g = group_dict["cout_g"].value
    cin1_g = group_dict["cin1_g"].value
    cout1_g = cout_g // tbe_platform.CUBE_MKN[b_ddr.dtype]["mac"][2]

    padu, padd, padl, padr = padding
    pad_head, pad_tail = list(i.value for i in c_ddr.op.attrs["depth_pad"])
    tensor_attr['pad_head'] = pad_head
    tensor_attr['pad_tail'] = pad_tail
    _, _, _, _, _, kernel_h, kernel_w, _ = list(i.value for i in a_col_before.shape)
    _, _, _, ho_l1, wo_l1, _ = list(i.value for i in a_l1.shape)
    img_shape = list(i.value for i in a_ddr.shape)
    _, dy_depth, dy_cout1, _, dy_w, _ = img_shape
    b_ddr_n1, b_ddr_k1, b_ddr_k0, b_ddr_n0 = list(i.value for i in b_ddr.shape)

    b_group = real_g

    filter_shape = [b_ddr_k1 * b_ddr_k0,
                    b_ddr_kd, b_ddr_n1 // (kernel_h * kernel_w * real_g * b_ddr_kd),
                    kernel_h, kernel_w, b_ddr_n0]

    cddr_batch, cddr_depth, cddr_c1, cddr_h, cddr_w, cdder_c0 = output_shape
    tiling_output = [cddr_batch,
                     cddr_depth, cddr_h, cddr_w, cddr_c1 * cdder_c0]
    kd_reduce_flag = bool(len(c_col.op.reduce_axis) == NUM_3)

    # tiling_query
    tiling = tiling_query(a_shape=img_shape,
                          b_shape=filter_shape,
                          c_shape=tiling_output,
                          a_dtype='float16',
                          b_dtype='float16',
                          c_dtype='float16',
                          mad_dtype='float32',
                          padl=padl,
                          padr=padr,
                          padu=padu,
                          padd=padd,
                          padf=pad_head,
                          padb=pad_tail,
                          strideh=1, stridew=1,
                          strideh_expand=stride_h,
                          stridew_expand=stride_w,
                          strided=stride_d,
                          bias_flag=False,
                          op_tag="conv3d_backprop_input",
                          kernel_name=c_ddr.op.attrs["kernel_name"].value)

    if tiling["AL0_matrix"][2] == DEFAULT_TILING_FLAG:
        tiling = _default_tiling()

    if stride_w == 1 and stride_h == 1:
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

    _, _, al1_co1, _, _, al1_co0 = list(i.value for i in a_l1.shape)
    _, _, _, _, c_l0c_hw, _ = list(i.value for i in c_col.shape)
    _, _, bl1_k1, bl1_co1, bl1_co0, _ = list(i.value for i in b_l1.shape)
    a_col_shape = list(i.value for i in a_col.shape)
    _, a_col_batch, _, a_col_ma, a_col_ka, _, _ = a_col_shape
    cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, _, _, _ = tiling.get("CUB_matrix")
    cl0_tiling_nc, cl0_tiling_mc, _, _, _, _ = tiling.get("CL0_matrix")
    al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, _, _, al0_tiling_dfactor = tiling.get("AL0_matrix")
    bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_kd = _tiling_l0_process()
    al1_tiling_k, al1_tiling_m, bl1_tiling_k, bl1_tiling_n, bl1_tiling_kdparts = _tiling_l1_process()

    # tiling_check
    _tiling_check()

    # axis management
    g_axis, n_after_multicore = sch[c_ddr].split(c_ddr.op.axis[2], nparts=real_g)
    batch_after_multicore, d_after_multicore, m_after_multicore = (
        c_ddr.op.axis[0], c_ddr.op.axis[1], c_ddr.op.axis[3])
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

    c_ddr_deep_outer_value = c_ddr.op.axis[1].dom.extent.value // cddr_deep_factor
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

    if stride_h > 1 or stride_w > 1:
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
    _, _, _, _, dx_w, _ = output_shape
    sch[a_col_before].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (dx_w, dx_w),
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
    (batch_outer, c_ddr_deep_outer, bl1_at_ddr_n_outer, al1_at_ddr_m_outer,
     g_axis, bind_in, blockidx, blocks) = _multi_core()

    _do_compute_at()

    # to correct n_axis inaccurate inference
    def _n_buffer_tile():
        batch_dim, n_dim, m_dim, d_dim = tiling["block_dim"]
        group_dim = tiling["g_dim"]
        bl1_outer_size = te_util.int_ceil_div(cddr_c1, real_g * n_dim) // bl1_tiling_n // cl0_tiling_nc
        bl1_inner_size = bl1_tiling_n
        g_size = te_util.int_ceil_div(real_g, group_dim)

        if blocks != 1:
            block_g = blockidx // (batch_dim * n_dim * m_dim * d_dim)
            block_n = blockidx // m_dim % n_dim
            axis1 = block_g * (n_dim * bl1_outer_size * bl1_inner_size)
            axis2 = block_n * (g_size * bl1_outer_size * bl1_inner_size)
            n_axis = (axis1 + axis2 + bl1_at_ddr_n_outer * bl1_inner_size + bl1_at_ddr_n_inner) * cl0_tiling_nc
        else:
            n_axis = (bl1_at_ddr_n_outer * bl1_inner_size + bl1_at_ddr_n_inner) * cl0_tiling_nc
        extent = cl0_tiling_nc

        if kd_reduce_flag:
            sch[c_col].buffer_tile(
                (None, None), (None, None),
                (None, None), (n_axis, extent),
                (None, None), (None, None),
                (None, None), (None, None),
                (None, None)
            )
        else:
            sch[c_col].buffer_tile(
                (None, None), (None, None),
                (None, None), (n_axis, extent),
                (None, None), (None, None),
                (None, None), (None, None)
            )

    _n_buffer_tile()
    _double_buffer()

    # emit insn
    _emit_insn_process()
    return sch
