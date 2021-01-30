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
conv3d backprop filter schudule.
"""
import te.platform as tbe_platform
from te.lang.cce.te_compute import util as te_util
from te.domain.tiling import get_tiling
from tbe.common.utils.errormgr import error_manager_util
from te import tvm


_CUBE_DIM = 16
_FLOAT16_SIZE = 2
_CUBE_MUL_SHAPE = 256
_OPEN_DOUBLE_BUFFER = 2
_DEFAULT_TILING_CASE = 32


def _print_ir_conv(process, sch):
    """
    print ir for input sch

    Parameter:
    --------------------------------------------------------------
    :param process: tag
    :param sch: schedule
    :return: IR process
    ---------------------------------------------------------------
    """
    from te.platform.cce_build import build_config
    with build_config:
        sch1 = sch.normalize()
        start = process + " IR start"
        end = process + " IR end\n"
        print(start)
        bounds = tvm.schedule.InferBound(sch1)
        stmt = tvm.schedule.ScheduleOps(sch1, bounds, True)
        print(stmt)
        print(end)


class CceConv3dBackpropFilterOp(object):  # pylint: disable=too-few-public-methods
    """
    CceConv3dBackpropFilterOp: schedule definition of conv3d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    schedule : schedule definition of conv3d_backprop_filter

    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        """
        initialization

        Parameters:
        ----------
        scope : scope definition

        need_tensorize : whether needs tensorize

        need_pragma : whether needs pragma

        Returns
        -------
        None
        """
        self.scope = scope
        self.need_tensorize = need_tensorize
        self.need_pragma = need_pragma
        self.spec_node_list = []

    def schedule(
            self,  # pylint: disable=R0914,R0915
            res,
            spec_node_list,
            sch_list):
        """
        schedule definition of conv3d_backprop_filter

        Parameters:
        ----------
        res :

        spec_node_list :

        sch_list:

        Returns
        -------
        None
        """
        self.spec_node_list = spec_node_list

        def _tiling_shape_check():
            """
            do tiling shape paramters general check

            """

            al1_shape = tiling.get("AL1_shape")
            bl1_shape = tiling.get("BL1_shape")
            al0_matrix = tiling.get("AL0_matrix")
            bl0_matrix = tiling.get("BL0_matrix")
            cl0_matrix = tiling.get("CL0_matrix")
            if al1_shape:
                if al1_shape[0] % al0_matrix[1] != 0:
                    cube_err.raise_err_specific("conv3d",
                        "k of AL1_shape should be integral multiple of AL0_matrix")

                if al1_shape[1] < 1:
                    cube_err.raise_err_specific("conv3d",
                        "m of AL1_shape should be integral multiple of AL0_matrix")

            if bl1_shape:
                if (bl1_shape[0] // _CUBE_DIM) % bl0_matrix[0] != 0:
                    cube_err.raise_err_specific("conv3d",
                        "k of BL1_shape should be integral multiple of BL0_matrix")

                if bl1_shape[1] < 1:
                    cube_err.raise_err_specific("conv3d",
                        "n of BL1_shape should be integral multiple of BL0_matrix")

            if al0_matrix:
                if al0_matrix[0] != cl0_matrix[1]:
                    cube_err.raise_err_specific("conv3d",
                        "mc of AL0_matrix and CL0_matrix should be same")

            if bl0_matrix:
                if bl0_matrix[1] != cl0_matrix[0]:
                    cube_err.raise_err_specific("conv3d",
                        "nc of BL0_matrix and CL0_matrix should be same")

            if al0_matrix and bl0_matrix:
                if al0_matrix[1] != bl0_matrix[0]:
                    cube_err.raise_err_specific("conv3d",
                        "k of AL0_matrix and BL0_matrix should be same")

        def _tiling_buffer_check():
            """
            Do buffer paramters general check

            """
            block_cout = tiling.get("block_dim")[2]

            al1_pbuff = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
            bl1_pbuff = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
            al0_pbuff = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
            bl0_pbuff = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
            l0c_pbuff = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
            cub_pbuff = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
            cl0_matrix = tiling.get("CL0_matrix")
            cub_matrix = tiling.get("CUB_matrix")
            if cl0_matrix[0] % cub_matrix[0] != 0 or cl0_matrix[1] != cub_matrix[1]:
                cube_err.raise_err_specific("conv3d", "invalid CUB_matrix value")

            # blockIdx must be positive int
            if block_cout < 1:
                cube_err.raise_err_specific("conv3d", "blockIdx must be positive int")

            # only support no dbuffer/ dbuffer
            if al1_pbuff not in (1, 2):
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'AL1_pbuffer',
                    'expect_value': '1 or 2',
                    'value': str(al1_pbuff)
                }
                raise RuntimeError(dict_args,
                    error_manager_util.get_error_message(dict_args))

            if bl1_pbuff not in (1, 2):
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'BL1_pbuffer',
                    'expect_value': '1 or 2',
                    'value': str(bl1_pbuff)
                }
                raise RuntimeError(dict_args,
                    error_manager_util.get_error_message(dict_args))

            if al0_pbuff not in (1, 2):
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'AL0_pbuffer',
                    'expect_value': '1 or 2',
                    'value': str(al0_pbuff)
                }
                raise RuntimeError(dict_args,
                    error_manager_util.get_error_message(dict_args))

            if bl0_pbuff not in (1, 2):
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'BL0_pbuffer',
                    'expect_value': '1 or 2',
                    'value': str(bl0_pbuff)
                }
                raise RuntimeError(dict_args,
                    error_manager_util.get_error_message(dict_args))

            if l0c_pbuff not in (1, 2):
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'L0C_pbuffer',
                    'expect_value': '1 or 2',
                    'value': str(l0c_pbuff)
                }
                raise RuntimeError(dict_args,
                    error_manager_util.get_error_message(dict_args))

            if cub_pbuff not in (1, 2):
                dict_args = {
                    'errCode': 'E62305',
                    'param_name': 'CUB_pbuffer',
                    'expect_value': '1 or 2',
                    'value': str(cub_pbuff)
                }
                raise RuntimeError(dict_args,
                    error_manager_util.get_error_message(dict_args))

        def _l1_limit_check():
            """
            do L1 size limit check

            """
            al1_min_byte = _CUBE_DIM * _CUBE_DIM * _FLOAT16_SIZE
            if width_grads >= _CUBE_DIM:
                if width_grads % _CUBE_DIM == 0:
                    bl1_min_byte = kernel_height * width_fmap * _CUBE_DIM * _FLOAT16_SIZE
                else:
                    bl1_min_byte = (kernel_height + stride_height) * width_fmap * _CUBE_DIM * _FLOAT16_SIZE
            else:
                bl1_align_factor = te_util.int_ceil_div(_CUBE_DIM, width_grads)
                if _CUBE_DIM % width_grads == 0:
                    bl1_min_byte = (kernel_height + (bl1_align_factor-1)
                                    * stride_height) * width_fmap * _CUBE_DIM * _FLOAT16_SIZE
                else:
                    bl1_min_byte = (kernel_height +
                                    bl1_align_factor * stride_height) * width_fmap * _CUBE_DIM * _FLOAT16_SIZE
            l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
            if (al1_min_byte + bl1_min_byte) > l1_size:
                cube_err.raise_err_attr_range_invalid("conv3d",
                    "(,{}]".format(l1_size),
                    "al1_and_bl1_byte",
                    str(al1_min_byte + bl1_min_byte))

        def _atomic_add(sch, res_cc, res_ub, res_ddr):
            """
            achieve atomic add according to refactor dw_cc

            """

            # redefine dw_ddr, dw_ub, dw_cc to achieve atomic write
            ub_reduce = res_ub
            ddr_reduce = res_ddr

            batch, real_k = sch[res_cc].op.reduce_axis
            batch_core, batch_in = sch[res_cc].split(batch, nparts=block_dim_batch)

            real_k, k_in = sch[res_cc].split(real_k, _CUBE_DIM)
            k_1_multicore, real_k = sch[res_cc].split(real_k, nparts=block_dim_hw)

            sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, real_k, k_in)
            fused_atomic_write = sch[res_cc].fuse(k_1_multicore, batch_core)

            # after rfactor op, dw_cc becomes dw_ddr, original dw_ub and dw_ddr
            # will be dropped
            res_ddr = res_cc
            res_cc = sch.rfactor(res_ddr, fused_atomic_write)
            sch[res_cc].set_scope(tbe_platform.scope_cc)
            res_ub = sch.cache_read(res_cc, tbe_platform.scope_ubuf, [res_ddr])
            return res_cc, res_ub, res_ddr, ub_reduce, ddr_reduce

        def _full_k_check():
            """
            set flag whether axis K is fully loaded in L0A and L0B
            return:
            -------
            full_k_l0a: 1 or 0,
                        1 means K is fully loaded in L0A
            full_k_l0b: 1 or 0,
                        1 means K is fully loaded in L0B
            """

            # if k is fully load in BL1 and
            # there is multi load in N1 and N1 in BL1
            # isn't aligned to kernel_height*kernel_width, then align to it
            if (tiling.get("BL1_shape") and tiling.get("BL1_shape")[1] > 1 and
                    tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1]
                    % (kernel_height * kernel_width) != 0):
                tiling["BL1_shape"][1] = te_util.align(
                    tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1],
                    kernel_height * kernel_width) // tiling.get("BL0_matrix")[1]

            # whether axis K is fully loaded in L0A and L0B
            # excluding axis batch
            if not tiling["AL0_matrix"]:
                full_k_l0a = 1
            else:
                full_k_l0a = tiling["AL0_matrix"][1] // te_util.int_ceil_div(hw_pad_1, block_dim_hw)

            if not tiling["BL0_matrix"]:
                full_k_l0b = 1
            else:
                full_k_l0b = tiling["BL0_matrix"][0] // te_util.int_ceil_div(hw_pad_1, block_dim_hw)

            return full_k_l0a, full_k_l0b

        def _compute_tiling_parts():
            """
            compute the parts or the factors of tensors

            """

            if not tiling["AL0_matrix"]:  # if grads no tiling in L0A
                tiling["AL1_shape"] = []  # then no tiling in L1

            # dw_cc is (fmap_channel_1*kernel_height*kernel_width,
            #          grads_channel_1, C0_grads, C0_fmap)
            dw_tiling_factor = [
                tiling["CL0_matrix"][0], tiling["CL0_matrix"][1]
            ]
            # nparts N, nparts M
            # dw_tiling_nparts only describe the nparts from single core to L0
            dw_tiling_nparts = [
                te_util.int_ceil_div(fkk // block_dim_cin, dw_tiling_factor[0]),
                te_util.int_ceil_div(te_util.int_ceil_div(c1_grads, dw_tiling_factor[1]),
                                     block_dim_cout)
                ]

            # tiling parameters of dw_ub
            dw_ub_tiling_factor = [
                tiling["CUB_matrix"][0], tiling["CUB_matrix"][1]
            ]
            dw_ub_tiling_nparts = [
                te_util.int_ceil_div(dw_tiling_factor[0], dw_ub_tiling_factor[0]),
                te_util.int_ceil_div(dw_tiling_factor[1], dw_ub_tiling_factor[1])
            ]

            # only support loading one batch to L1 at a time for now
            # cout:out->single core(sc)->L1
            if tiling["AL1_shape"]:  # if grads needs tiling in L1
                if len(tiling["AL1_shape"]) == 1:  # but no C_1 tiling info
                    tiling["AL1_shape"] = tiling["AL1_shape"] + [1]
                # nparts K1 in L1, nparts M1 in L1
                grads_l1_tiling_nparts = [
                    hw_pad_1 // block_dim_hw //
                    (tiling["AL1_shape"][0] // _CUBE_DIM),
                    dw_tiling_nparts[1] // tiling["AL1_shape"][1]
                ]
            else:
                grads_l1_tiling_nparts = [1, 1]

            if tiling["BL1_shape"]:  # if fmap needs tiling in L1
                if len(tiling["BL1_shape"]) == 1:  # but no fkk tiling info
                    tiling["BL1_shape"] = tiling["BL1_shape"] + [1]  # tiling fkk=1
                # DDR to L1 [nparts K1, nparts N1]
                fmap_l1_tiling_nparts = [
                    hw_pad_1 // block_dim_hw //
                    (tiling["BL1_shape"][0] // _CUBE_DIM),
                    dw_tiling_nparts[0] // tiling["BL1_shape"][1]
                ]
            else:
                fmap_l1_tiling_nparts = [1, 1]

            # during L1 to L0 [nparts N1, nparts M1]
            l1_2_l0_tiling_nparts = [
                dw_tiling_nparts[0] // fmap_l1_tiling_nparts[1],
                dw_tiling_nparts[1] // grads_l1_tiling_nparts[1]
            ]
            # ka and kb may be different,
            # the min value corresponds to one MMAD,
            # the larger one is []
            if tiling["AL0_matrix"]:  # dw_k equals to ka if L0A needs tiling
                dw_k = tiling["AL0_matrix"][1]
            elif tiling["BL0_matrix"]:
                dw_k = tiling["BL0_matrix"][0]
            else:  # both fully loaded
                dw_k = hw_pad_1 // block_dim_hw

            tiling_patrs_dict = dict()
            tiling_patrs_dict["dw_tiling_factor"] = dw_tiling_factor
            tiling_patrs_dict["dw_tiling_nparts"] = dw_tiling_nparts
            tiling_patrs_dict["dw_ub_tiling_factor"] = dw_ub_tiling_factor
            tiling_patrs_dict["dw_ub_tiling_nparts"] = dw_ub_tiling_nparts
            tiling_patrs_dict["grads_l1_tiling_nparts"] = grads_l1_tiling_nparts
            tiling_patrs_dict["fmap_l1_tiling_nparts"] = fmap_l1_tiling_nparts
            tiling_patrs_dict["l1_2_l0_tiling_nparts"] = l1_2_l0_tiling_nparts
            tiling_patrs_dict["dw_k"] = dw_k
            return tiling_patrs_dict

        def _l0_attach():
            """
            achieve Al0 and Bl0 compute at loc or ddr

            """

            if tiling["AL0_matrix"]:
                if (batch_num_sc == 1) and (full_k_in_l0a == 1):
                    # L0A data is more than that L0C needed, attach to dw_ddr
                    sch[grads_fractal].compute_at(sch[dw_ddr], c_grads_mad_at)
                else:
                    sch[grads_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
            else:  # else: fully load, attach to thread_axis
                sch[grads_fractal].compute_at(sch[dw_ddr], fused_multi_core)

            if tiling["BL0_matrix"]:
                if (batch_num_sc == 1) and (full_k_in_l0b == 1):
                    sch[fmap_fractal].compute_at(sch[dw_ddr], c_fmap_mad_at)
                else:
                    sch[fmap_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
            else:  # else: fully load, attach to thread_axis
                sch[fmap_fractal].compute_at(sch[dw_ddr], fused_multi_core)

        def _al1_attach():
            """
            achieve Al1 compute at l0c or ddr

            """
            if tiling["AL1_shape"]:
                # if axis K needs split, then attach to dw_cc
                if grads_l1_tiling_nparts[0] != 1 or batch_num_sc != 1:
                    sch[grads_matrix].compute_at(sch[dw_cc], al1_at_axis)
                else:  # if axis K fully load in L1, attach to dw_ddr
                    sch[grads_matrix].compute_at(sch[dw_ddr], c_grads_l1_at)
            else:  # else: fully load, attach to thread_axis
                sch[grads_matrix].compute_at(sch[dw_ddr], fused_multi_core)

        def _bl1_attach():
            """
            achieve Bl1 compute at l0c or ddr

            """
            if tiling["BL1_shape"]:
                # if axis K needs split, then attach to dw_cc
                if fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1:
                    sch[fmap_matrix].compute_at(sch[dw_cc], bl1_at_axis)
                    if not load2d_flag:
                        sch[fmap_l1].compute_at(sch[dw_cc], bl1_at_axis)
                else:  # if axis K fully load in L1, attach to dw_ddr
                    sch[fmap_matrix].compute_at(sch[dw_ddr], c_fmap_l1_at)
                    if not load2d_flag:
                        sch[fmap_l1].compute_at(sch[dw_ddr], c_fmap_l1_at)

            else:  # else: fully load, attach to thread_axis
                sch[fmap_matrix].compute_at(sch[dw_ddr], fused_multi_core)
                if not load2d_flag:
                    sch[fmap_l1].compute_at(sch[dw_ddr], fused_multi_core)

        def _double_buffer():
            """
            achieve double_buffer

            """
            if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == _OPEN_DOUBLE_BUFFER:
                sch[grads_matrix].double_buffer()

            if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == _OPEN_DOUBLE_BUFFER:
                if not load2d_flag:
                    sch[fmap_l1].double_buffer()
                else:
                    sch[fmap_matrix].double_buffer()

            if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == _OPEN_DOUBLE_BUFFER:
                sch[grads_fractal].double_buffer()

            if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == _OPEN_DOUBLE_BUFFER:
                sch[fmap_fractal].double_buffer()

            if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == _OPEN_DOUBLE_BUFFER:
                sch[dw_cc].double_buffer()

            if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") == _OPEN_DOUBLE_BUFFER:
                sch[dw_ub].double_buffer()

        def _emit_insn():
            """
            achieve emit_insn

            """
            setfmatrix_dict = dict()
            setfmatrix_dict["conv_kernel_h"] = kernel_height
            setfmatrix_dict["conv_kernel_w"] = kernel_width
            setfmatrix_dict["conv_padding_top"] = pad_up
            setfmatrix_dict["conv_padding_bottom"] = pad_down
            setfmatrix_dict["conv_padding_left"] = pad_left
            setfmatrix_dict["conv_padding_right"] = pad_right
            setfmatrix_dict["conv_stride_h"] = stride_height
            setfmatrix_dict["conv_stride_w"] = stride_width
            setfmatrix_dict["conv_fm_c"] = featuremap_channel
            setfmatrix_dict["conv_fm_h"] = featuremap_height
            setfmatrix_dict["conv_fm_w"] = featuremap_width
            setfmatrix_dict["conv_dilation_h"] = dilation_height
            setfmatrix_dict["conv_dilation_w"] = dilation_width

            mad_dict = {
                "mad_pattern":
                2,
                "k_outer": [
                    batch_insn_o, hw_mad_1_l1_out_at, hw_mad_1_l1_in_at,
                    hw_mad_1_mad_at
                ]
            }

            if pad_front != 0 or pad_back != 0:
                batch_do_axis = (
                    (block.var) // block_dim_cout // block_dim_cin // block_dim_g %
                    block_dim_batch) * (batch_fmap * depth_grads //
                                        block_dim_batch) + batch_insn_o

                dk_c1_axis = (
                    (block % (block_dim_cin)) *
                    (fkk // block_dim_cin) +
                    (((c_fmap_l1_c1 * factor_kh + c_fmap_l1_kh) * factor_kw +
                      c_fmap_l1_at) *
                     (dw_tiling_nparts[0] // fmap_l1_tiling_nparts[1]) +
                     c_fmap_mad_at) * dw_tiling_factor[0]) // (kernel_height *
                                                               kernel_width)

                c1_fmap_info = group_dict['cin1_g']
                mad_dict = {
                    "mad_pattern":
                    2,
                    "k_outer": [
                        batch_insn_o, hw_mad_1_l1_out_at, hw_mad_1_l1_in_at,
                        hw_mad_1_mad_at
                    ],
                    "k_coeff":
                    tvm.all((batch_do_axis % depth_grads) * stride_depth +
                            dk_c1_axis // c1_fmap_info >= pad_front,
                            (batch_do_axis % depth_grads) * stride_depth +
                            dk_c1_axis // c1_fmap_info < pad_front + depth_fmap),
                    "k_cond":
                    tvm.any(
                        tvm.all((batch_do_axis % depth_grads - 1) * stride_depth +
                                dk_c1_axis // c1_fmap_info < pad_front,
                                axis_k_reduce_for_mad <= 0, batch_do_axis %
                                (batch_fmap * depth_grads // block_dim_batch) <
                                depth_grads),
                        batch_insn_o + axis_k_reduce_for_mad <= 0)
                }

            sch[grads_matrix].emit_insn(grads_matrix.op.axis[0], 'dma_copy')
            # move grads from L1 to L0A
            sch[grads_fractal].emit_insn(grads_fractal.op.axis[0], 'dma_copy')

            # move fmap from ddr to L1
            if not load2d_flag:
                sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')
                sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0],
                                           'set_fmatrix', setfmatrix_dict)
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[1], 'im2col')
            else:
                sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0], 'dma_copy')
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0],
                                            'dma_copy')

            # move dw from L0C to UB
            sch[dw_ub].emit_insn(dw_ub.op.axis[0], 'dma_copy')
            sch[dw_cc].emit_insn(batch_insn, 'mad', mad_dict)

            # add condition for pad of D dimension
            if pad_front != 0 or pad_back != 0:
                ddr_condition_left = (
                    (block.var) // block_dim_cout // block_dim_cin // block_dim_g %
                    block_dim_batch) * (batch_fmap * depth_grads //
                                        block_dim_batch) + batch_insn_o_size - 1
                ddr_condition_right = (
                    (block.var) // block_dim_cout // block_dim_cin // block_dim_g %
                    block_dim_batch) * (batch_fmap * depth_grads //
                                        block_dim_batch)
                sch[dw_ddr].set_store_predicate(
                    tvm.all((ddr_condition_left % depth_grads) * stride_depth +
                            dk_c1_axis // c1_fmap_info >= pad_front,
                            (ddr_condition_right % depth_grads) * stride_depth +
                            dk_c1_axis // c1_fmap_info < pad_front + depth_fmap))

            # move dw form UB to ddr
            sch[dw_ddr].emit_insn(c_fmap_2_ub_insn, 'dma_copy')

            sch[dw_ddr_reduce].emit_insn(dw_ddr_reduce.op.axis[0],
                                         'phony_insn')
            sch[dw_ub_reduce].emit_insn(dw_ub_reduce.op.axis[0], 'phony_insn')

            sch_list.append(dw_ddr)

        # ####################### get computing graph #######################
        dw_ddr = res  # pylint: disable=too-many-statements
        dw_ub = dw_ddr.op.input_tensors[0]
        dw_cc = dw_ub.op.input_tensors[0]
        grads_fractal = dw_cc.op.input_tensors[0]
        fmap_fractal = dw_cc.op.input_tensors[1]
        grads_matrix = grads_fractal.op.input_tensors[0]
        fmap_matrix = fmap_fractal.op.input_tensors[0]
        grads = grads_matrix.op.input_tensors[0]
        load2d_flag = fmap_matrix.op.attrs['load2d_flag'].value
        group_dict = fmap_matrix.op.attrs['group_dict']
        kernel_name = dw_cc.op.attrs['kernel_name'].value
        if load2d_flag:
            fmap = fmap_matrix.op.input_tensors[0]
        else:
            fmap_l1 = fmap_matrix.op.input_tensors[0]
            fmap = fmap_l1.op.input_tensors[0]

        # ########################extract parameters##########################
        default_tiling = {
            'AUB_shape': None,
            'BUB_shape': None,
            'AL1_shape': [_CUBE_DIM, 1, 1],
            'BL1_shape': [_CUBE_DIM, 1, 1],
            'AL0_matrix': [1, 1, _CUBE_DIM, _CUBE_DIM, 1],
            'BL0_matrix': [1, 1, _CUBE_DIM, _CUBE_DIM, 1],
            'CL0_matrix': [1, 1, _CUBE_DIM, _CUBE_DIM, 1],
            'CUB_matrix': [1, 1, _CUBE_DIM, _CUBE_DIM, 1],
            'block_dim': [1, 1, 1],
            'cout_bef_batch_flag': 0,
            'A_overhead_opt_flag': 0,
            'B_overhead_opt_flag': 0,
            'manual_pingpong_buffer': {
                'AUB_pbuffer': 1,
                'BUB_pbuffer': 1,
                'AL1_pbuffer': 1,
                'BL1_pbuffer': 1,
                'AL0_pbuffer': 1,
                'BL0_pbuffer': 1,
                'CL0_pbuffer': 1,
                'CUB_pbuffer': 1,
                'UBG_pbuffer': 1
            }
        }
        cin1_g = group_dict['cin1_g'].value
        cout_g = group_dict['cout_g'].value
        real_g = group_dict['real_g'].value

        batch_grads, depth_grads, c1_grads, height_grads, \
            width_grads, c0_grads = list(x.value for x in grads.shape)
        grads_shape = [
            batch_grads, depth_grads,
            cout_g // c0_grads, height_grads, width_grads,
            c0_grads
        ]

        batch_fmap, depth_fmap, c1_fmap, height_fmap, width_fmap, c0_fmap \
            = list(x.value for x in fmap.shape)
        _, fkk, _, _ = list(x.value for x in dw_cc.shape)
        _, _, hw_pad_1, _, _, _ = list(x.value for x in fmap_fractal.shape)

        fmap_shape = [
            batch_fmap, depth_fmap, cin1_g, height_fmap, width_fmap, c0_fmap
        ]

        # load_3d parameters
        stride_depth = fmap_matrix.op.attrs['stride'][0].value
        stride_height = fmap_matrix.op.attrs['stride'][1].value
        stride_width = fmap_matrix.op.attrs['stride'][2].value
        pad_front = fmap_matrix.op.attrs['pad'][0].value
        pad_back = fmap_matrix.op.attrs['pad'][1].value
        pad_up = fmap_matrix.op.attrs['pad'][2].value
        pad_down = fmap_matrix.op.attrs['pad'][3].value
        pad_left = fmap_matrix.op.attrs['pad'][4].value
        pad_right = fmap_matrix.op.attrs['pad'][5].value
        kernel_depth = fmap_matrix.op.attrs['kernel_size'][1].value
        kernel_height = fmap_matrix.op.attrs['kernel_size'][3].value
        kernel_width = fmap_matrix.op.attrs['kernel_size'][4].value
        dilation_height = fmap_matrix.op.attrs['dilation'][3].value
        dilation_width = fmap_matrix.op.attrs['dilation'][4].value
        featuremap_channel = kernel_depth * c1_fmap * c0_fmap
        featuremap_height = height_fmap
        featuremap_width = width_fmap

        weight_shape = [
            cout_g, kernel_depth, kernel_height, kernel_width,
            cin1_g * c0_fmap
        ]

        sch = sch_list[0]

        info_dict = {
            "a_shape": grads_shape,
            "b_shape": fmap_shape,
            "c_shape": weight_shape,
            "a_dtype": grads.dtype,
            "b_dtype": fmap.dtype,
            "c_dtype": dw_cc.dtype,
            "mad_dtype": dw_cc.dtype,
            "pad": [pad_front, pad_back, pad_up, pad_down, pad_left,
                    pad_right],
            "stride": [stride_depth, stride_height, stride_width],
            "strideh_expand": 1,
            "stridew_expand": 1,
            "dilation": [1, dilation_height, dilation_width],
            "group": real_g,
            "fused_coefficient": [0, 0, 0],
            "bias_flag": False,
            "op_type": "conv3d_backprop_filter",
            "kernel_name": kernel_name
        }
        tiling = get_tiling.get_tiling(info_dict)

        _tiling_shape_check()
        _tiling_buffer_check()
        # if no valid tiling found, the flag is as follows
        if tiling["AL0_matrix"][2] == _DEFAULT_TILING_CASE:
            tiling = default_tiling
        _l1_limit_check()

        batch_num = batch_grads * depth_grads
        if tiling.get("AUB_shape"):
            block_dim_hw = tiling.get("AUB_shape")[0]
        else:
            block_dim_hw = 1
        block_dim_batch = tiling.get("block_dim")[0]
        block_dim_cout = tiling.get("block_dim")[2]
        block_dim_cin = tiling.get("block_dim")[1]
        if tiling.get("BUB_shape") and tiling.get("BUB_shape")[0]:
            block_dim_g = tiling.get("BUB_shape")[0]
        else:
            block_dim_g = 1

        sch[grads_matrix].set_scope(tbe_platform.scope_cbuf)
        sch[grads_matrix].storage_align(sch[grads_matrix].op.axis[1],
                                        _CUBE_MUL_SHAPE, 0)

        sch[grads_fractal].set_scope(tbe_platform.scope_ca)
        sch[grads_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                        (1, _CUBE_DIM), (1, _CUBE_DIM))

        # fmap_shape_original_matrix is (batch_size*grads_depth,
        #                               grads_height*grads_width,
        #                               kernel_depth*fmap_channel_1,
        #                               kernel_height,
        #                               kernel_width,
        #                               C0_fmap)
        if not load2d_flag:
            sch[fmap_l1].set_scope(tbe_platform.scope_cbuf)
            sch[fmap_matrix].buffer_align(
                (1, 1), (width_grads, width_grads), (1, 1),
                (kernel_height, kernel_height), (kernel_width, kernel_width),
                (1, _CUBE_DIM))
        else:
            sch[fmap_matrix].storage_align(sch[fmap_matrix].op.axis[1],
                                           _CUBE_MUL_SHAPE, 0)

        sch[fmap_matrix].set_scope(tbe_platform.scope_cbuf)

        sch[fmap_fractal].set_scope(tbe_platform.scope_cb)
        sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                       (1, _CUBE_DIM), (1, _CUBE_DIM))

        dw_cc, dw_ub, dw_ddr, dw_ub_reduce, dw_ddr_reduce \
            = _atomic_add(sch, dw_cc, dw_ub, dw_ddr)

        # #######################tiling parameters analyze####################
        batch_num_sc = batch_num // block_dim_batch

        full_k_in_l0a, full_k_in_l0b = _full_k_check()

        tiling_patrs_dict = _compute_tiling_parts()
        dw_tiling_factor = tiling_patrs_dict["dw_tiling_factor"]
        dw_tiling_nparts = tiling_patrs_dict["dw_tiling_nparts"]
        dw_ub_tiling_factor = tiling_patrs_dict["dw_ub_tiling_factor"]
        grads_l1_tiling_nparts = tiling_patrs_dict["grads_l1_tiling_nparts"]
        fmap_l1_tiling_nparts = tiling_patrs_dict["fmap_l1_tiling_nparts"]
        l1_2_l0_tiling_nparts = tiling_patrs_dict["l1_2_l0_tiling_nparts"]
        dw_k = tiling_patrs_dict["dw_k"]

        # #############################split axis N##########################
        g_multicore, g_axis = sch[dw_ddr].split(
            sch[dw_ddr].op.axis[0], nparts=block_dim_g)
        c_fmap_multicore, c_fmap_mad_at \
            = sch[dw_ddr].split(sch[dw_ddr].op.axis[1], nparts=block_dim_cin)

        c_fmap_mad_at, c_fmap_mad_insn = sch[dw_ddr].split(
            c_fmap_mad_at, nparts=dw_tiling_nparts[0])

        c_fmap_l1_ori, c_fmap_mad_at = sch[dw_ddr].split(
            c_fmap_mad_at,
            nparts=fmap_l1_tiling_nparts[1])

        def _ddr_n_split():
            # for N axis, if Hk and Wk needs split, do explict split
            if not load2d_flag:
                if tiling.get("BL1_shape"):
                    nc_cc = tiling.get("CL0_matrix")[0] * tiling.get("BL1_shape")[1]
                else:
                    nc_cc = kernel_depth * cin1_g * kernel_width * kernel_height // block_dim_cin

                factor_kw = te_util.int_ceil_div(kernel_width, nc_cc)
                factor_kh = te_util.int_ceil_div(kernel_width * kernel_height, nc_cc) // factor_kw

                c_fmap_l1_out, c_fmap_l1_at = sch[dw_ddr].split(c_fmap_l1_ori, factor_kw)

                c_fmap_l1_c1, c_fmap_l1_kh = sch[dw_ddr].split(c_fmap_l1_out, factor_kh)
            else:
                factor_kw = 1
                factor_kh = 1
                c_fmap_l1_out, c_fmap_l1_at = sch[dw_ddr].split(c_fmap_l1_ori, 1)

                c_fmap_l1_c1, c_fmap_l1_kh = sch[dw_ddr].split(c_fmap_l1_out, 1)
            return c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at, factor_kw, factor_kh

        c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at, factor_kw, factor_kh = _ddr_n_split()

        # split axis M
        c_grads_mad_at, c_grads_mad_insn = sch[dw_ddr].split(
            sch[dw_ddr].op.axis[2], dw_tiling_factor[1]*_CUBE_DIM)

        c_grads_multicore, c_grads_mad_at = sch[dw_ddr].split(
            c_grads_mad_at, nparts=block_dim_cout)

        c_grads_l1_at, c_grads_mad_at = sch[dw_ddr].split(
            c_grads_mad_at, nparts=grads_l1_tiling_nparts[1])

        # reorder according to requirments of mmad EmitInsn
        sch[dw_ddr].reorder(sch[dw_ddr].op.reduce_axis[0], g_multicore,
                            c_grads_multicore, c_fmap_multicore,
                            g_axis, c_fmap_l1_c1, c_fmap_l1_kh,
                            c_fmap_l1_at, c_grads_l1_at, c_fmap_mad_at,
                            c_grads_mad_at, c_fmap_mad_insn, c_grads_mad_insn)

        def _ub_and_cc_attach():
            # optimization by move small loops to outer
            reorder_flag = False
            # during L1 to L0, if M loop is smaller, then move to outer
            if l1_2_l0_tiling_nparts[0] > l1_2_l0_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_mad_at, c_fmap_mad_at)
                reorder_flag = True
            # during sc to L1, if M loop is smaller, then move to outer
            if fmap_l1_tiling_nparts[1] > grads_l1_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_l1_at, c_fmap_l1_c1, c_fmap_l1_kh,
                                    c_fmap_l1_at)

            # dw_ub attach
            # dw_ub split
            c_fmap_2_ub_at, c_fmap_2_ub_insn = sch[dw_ddr].split(c_fmap_mad_insn, dw_ub_tiling_factor[0])
            # dw_ub attach
            sch[dw_ub].compute_at(sch[dw_ddr], c_fmap_2_ub_at)

            # dw attach
            if reorder_flag:
                sch[dw_cc].compute_at(sch[dw_ddr], c_fmap_mad_at)
            else:
                sch[dw_cc].compute_at(sch[dw_ddr], c_grads_mad_at)
            return c_fmap_2_ub_insn

        c_fmap_2_ub_insn = _ub_and_cc_attach()

        # dw_cc split
        # get the 3 reduce axis of dw_cc
        batch_axis_sc, k_1_axis_sc, k_0 = sch[dw_cc].op.reduce_axis

        # dw_k is the part for one MMAD
        hw_mad_1_mad_at, hw_mad_1_mad_insn = sch[dw_cc].split(k_1_axis_sc, dw_k)

        # mad_pattern :2 , the 1st axis should be 1, so do a fake split
        batch_insn_o, batch_insn = sch[dw_cc].split(batch_axis_sc, 1)

        # K of AL1 and BL1 can be different, there are 2 split methods
        # on which one is larger
        k_1_axis_sc_out_size = k_1_axis_sc.dom.extent // dw_k
        batch_insn_o_size = batch_axis_sc.dom.extent

        if grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]:
            hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                hw_mad_1_mad_at, nparts=grads_l1_tiling_nparts[0])
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                hw_mad_1_l1_at, nparts=fmap_l1_tiling_nparts[0])
            al1_at_axis = hw_mad_1_l1_in_at
            bl1_at_axis = hw_mad_1_l1_out_at
            axis_k_reduce_for_mad = (
                hw_mad_1_l1_out_at *
                (grads_l1_tiling_nparts[0] // fmap_l1_tiling_nparts[0]) +
                hw_mad_1_l1_in_at) * (
                    k_1_axis_sc_out_size //
                    grads_l1_tiling_nparts[0]) + hw_mad_1_mad_at
        else:
            hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                hw_mad_1_mad_at, nparts=fmap_l1_tiling_nparts[0])
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
            al1_at_axis = hw_mad_1_l1_out_at
            bl1_at_axis = hw_mad_1_l1_in_at
            axis_k_reduce_for_mad = (
                hw_mad_1_l1_out_at *
                (fmap_l1_tiling_nparts[0] // grads_l1_tiling_nparts[0]) +
                hw_mad_1_l1_in_at) * (
                    k_1_axis_sc_out_size //
                    fmap_l1_tiling_nparts[0]) + hw_mad_1_mad_at

        # split dw_cc.op.axis[0](N1), factor is one MMAD
        fkk_mad_at, fkk_mad_insn \
            = sch[dw_cc].split(sch[dw_cc].op.axis[2], dw_tiling_factor[0])

        # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
        lc_mad_at, lc_mad_insn = sch[dw_cc].split(sch[dw_cc].op.axis[3],
                                                  dw_tiling_factor[1] * _CUBE_DIM)

        sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, sch[dw_cc].op.axis[0],
                           batch_insn_o, hw_mad_1_l1_out_at, hw_mad_1_l1_in_at,
                           hw_mad_1_mad_at, batch_insn, fkk_mad_insn,
                           lc_mad_insn, sch[dw_cc].op.axis[4],
                           hw_mad_1_mad_insn, k_0)

        # #############################multi core#############################
        def _bind_core():
            fused_multi_core = sch[dw_ddr].fuse(sch[dw_ddr].op.reduce_axis[0],
                                                g_multicore,
                                                c_grads_multicore,
                                                c_fmap_multicore)
            fused_multi_core, pragma_at = sch[dw_ddr].split(fused_multi_core, 1)
            block = tvm.thread_axis("blockIdx.x")

            sch[dw_ddr].bind(fused_multi_core, block)
            blocks = block_dim_batch * block_dim_cin * block_dim_cout * block_dim_hw
            if blocks == block_dim_batch:
                sch[dw_ddr].pragma(pragma_at, 'json_info_batchBindOnly')

            return fused_multi_core, block

        fused_multi_core, block = _bind_core()

        _l0_attach()
        _al1_attach()
        _bl1_attach()
        _double_buffer()
        _emit_insn()

        return True
