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
Schedule of conv2d in v220/v300.
"""
from functools import reduce
import tbe
from tbe import tvm
from tbe.common.utils import log
from tbe.common.platform import CUBE_MKN
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.tiling import tiling_api
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.dsl.compute.max_pool2d_3_2_fusion_compute import MaxPoolParam
from tbe.dsl.static_schedule import util
from tbe.dsl.static_schedule.conv_fixpipefusion_schedule import FixpipeFusion
from tbe.dsl.static_schedule.conv_fixpipefusion_schedule import FixpipeFusionNew
from tbe.dsl.static_schedule.conv_schedule_util import ceil, ceil_div, is_support_fixpipe_op, get_src_tensor
from tbe.dsl.static_schedule.conv_ubfusion_schedule import EltwiseUBFusion
from tbe.dsl.static_schedule.conv_ubfusion_schedule import QuantFusion
from tbe.dsl.static_schedule.conv_maxpoolfusion_schedule import MaxpoolFusion
from tbe.dsl.static_schedule.conv_bn1_fusion_schedule import Conv2dBN1Fusion
from te.platform import cce_params


NON_L1_FUSION = -1
DEPTH_L1_FUSION = 0
BREADTH_L1_FUSION = 1

DDR_SCOPE = 0
L1_SCOPE = 1
L2_SCOPE = 2


class InputNd2Nz:
    """
    Class of input nd2nz.
    Transform the fmap of NHWC format in ddr to the fmap_l1 of 5hd format in L1.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.input_nd_flag

    def inline_input_nd_dynamic(self, sch, tensor_map, dynamic_flag):
        """
        inline al1 in nd2nz dynamic situation (group = 1).
        """
        if self.flag and dynamic_flag:
            input_nd_dynamic = tensor_map["fmap"]
            sch[input_nd_dynamic].compute_inline()

    @staticmethod
    def al1_nd2nz_emit_insn(sch, al1):
        """
        Dma for the al1 tensor in nd2nz situation.
        """
        sch[al1].emit_insn(al1.op.axis[1],
                           "dma_copy",
                           {"layout_transform": "nd2nz"})


class WeightNd2Nz:
    """
    Class of weight nd2nz.
    Transform the weight of NHWC format in ddr to the weight_l1 of fracZ format in L1.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.weight_nd_flag

    def check_bl1_nd2nz_tiling(self, tiling):
        """
        Check whether the bl1 tiling is None in weight nd2nz situation.
        """
        if self.flag and tiling["BL1_shape"] is None:
            err_man.raise_err_specific(
                "conv2d", "BL1 tiling cannot be None when weight nd2nz.")

    @staticmethod
    def bl1_nd2nz_emit_insn(sch, bl1):
        """
        Dma for the bl1 tensor in weight nd2nz situation.
        """
        sch[bl1].emit_insn(bl1.op.axis[0],
                           "dma_copy",
                           {"layout_transform": "nd2nz"})


class OutputNz2Nd:
    """
    Class of output nz2nd.
    Transform the output of 5hd format in L0C to the output of NHWC format in DDR.
    """
    def __init__(self, res):
        self.flag = res.op.tag == "5HD_trans_NHWC"

    @staticmethod
    def res_nz2nd_emit_insn(sch, res, res_pragma_axis):
        """
        Emit insn for res in output nz2nd situation.
        """
        sch[res].emit_insn(res_pragma_axis,
                           "dma_copy",
                           attrs={"layout_transform": "nz2nd"})


class LxFusion:
    """
    Class of L1 fusion and L2 fusion.
    """
    def __init__(self, conv_param):
        # lxfusion params
        self.fusion_para = conv_param.fusion_para
        self.l1_fusion_type = self.fusion_para.get("l1_fusion_type")
        self.fmap_l1_addr_flag = self.fusion_para.get("fmap_l1_addr_flag")
        self.fmap_l1_valid_size = self.fusion_para.get("fmap_l1_valid_size")
        self.input_memory_type = self.fusion_para.get("input_memory_type")
        self.output_memory_type = self.fusion_para.get("output_memory_type")

        # combined parameters
        self.l1_fusion_flag = self.l1_fusion_type in (DEPTH_L1_FUSION, BREADTH_L1_FUSION) # l1fusion enable

    def config_default_tiling(self, default_tiling):
        """
        Config default tiling in l1fusion.
        """
        if self.l1_fusion_flag:
            default_tiling["block_dim"] = [1, 1, 1, 1]
        if self.l1_fusion_type == BREADTH_L1_FUSION:
            default_tiling["AL1_shape"] = []
        return default_tiling

    def check_l1fusion_tiling(self, tiling):
        """
        Check the tiling of l1fusion.
        """
        if self.l1_fusion_type == 1 and tiling["AL1_shape"] != []:
            err_man.raise_err_value_or_format_invalid(
                "conv2d", "tiling[\"AL1_shape\"]", "[]",
                "when l1_fusion_type is breadth fusion.")

        if self.l1_fusion_flag:
            if tiling["A_overhead_opt_flag"]:
                err_man.raise_err_value_or_format_invalid(
                    "conv2d", 'tiling["A_overhead_opt_flag"]', "False", "when l1_fusion.")

            if tiling["B_overhead_opt_flag"]:
                err_man.raise_err_value_or_format_invalid(
                    "conv2d", 'tiling["B_overhead_opt_flag"]', "False", "when l1_fusion.")

            if tiling["block_dim"] != [1, 1, 1, 1]:
                err_man.raise_err_specific(
                    "conv2d", "only support one core tiling in L1 Fusion.")

    def config_al1_scope(self):
        """
        Config L1 scope to scope_cbuf_fusion when l1fusion is enabled.
        """
        if self.l1_fusion_flag:
            return cce_params.scope_cbuf_fusion
        return cce_params.scope_cbuf

    @staticmethod
    def align_al1_lxfusion(sch, al1):
        """
        AL1 buffer_align in l1fusion breadth fusion.
        """
        sch[al1].buffer_align(
            (1, 1),
            (1, 1),
            (al1.shape[2], al1.shape[2]),
            (al1.shape[3], al1.shape[3]),
            (1, 1))

    def config_l1_tensormap(self, sch, fmap, al1, op_graph):
        """
        Append L1 tensor in cce kernel function.
        """
        l1_tensor_map = {}
        if self.fmap_l1_addr_flag == "nothing":
            l1_tensor_map = None
        else:
            if self.l1_fusion_flag and self.input_memory_type[0] in (DDR_SCOPE, L2_SCOPE):
                for input_op in op_graph.input_ops:
                    l1_tensor_map[input_op["dst_buffer"]] = tvm.var("dummy")
                l1_tensor_map[fmap] = al1
                if self.fmap_l1_valid_size > 0:
                    sch[al1].set_buffer_size(self.fmap_l1_valid_size)
            else:
                l1_tensor_map = None

        util.L1CommonParam.l1_fusion_tensors_map = l1_tensor_map

    def al1_l1fusion_pragma(self, sch, al1):
        """
        Pragma on AL1 when l1fusion and fmap ddr in.
        """
        if self.input_memory_type[0] != L1_SCOPE and self.l1_fusion_flag:
            sch[al1].pragma(al1.op.axis[0], 'jump_data', 1)


class AippFusion:
    """
    Class of Aipp + conv2d fusion.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.aipp_fuse_flag

    @staticmethod
    def al1_aipp_emit_insn(sch, al1):
        """
        Emit insn for al1 in aipp fusion.
        """
        aipp_map = al1.op.attrs
        aipp_map['spr_0'] = al1.op.axis[0]
        aipp_map_res = {"spr_0": al1.op.axis[0],
                        "spr_1": aipp_map["spr_1"],
                        "spr_2": aipp_map["spr_2"],
                        "spr_3": aipp_map["spr_3"],
                        "spr_4": aipp_map["spr_4"],
                        "spr_8": aipp_map["spr_8"],
                        "spr_9": aipp_map["spr_9"],
                        "src_image_h": aipp_map["src_image_h"],
                        "src_image_w": aipp_map["src_image_w"],
                        "input_format": aipp_map["input_format"],
                        "load_start_pos_h": aipp_map["load_start_pos_h"],
                        "load_start_pos_w": aipp_map["load_start_pos_w"],
                        "crop_size_h": aipp_map["crop_size_h"],
                        "crop_size_w": aipp_map["crop_size_w"]}
        # v300 spr5-7 is deleted
        if not is_support_fixpipe_op():
            aipp_map_res["spr_5"] = aipp_map["spr_5"]
            aipp_map_res["spr_6"] = aipp_map["spr_6"]
            aipp_map_res["spr_7"] = aipp_map["spr_7"]

        sch[al1].emit_insn(al1.op.axis[1], "load_image_to_cbuf", aipp_map_res)


class DynamicShape:
    """
    Class of dynamic shape.
    """
    def __init__(self, conv_param, tiling_dict_flag, tiling_case, var_range):
        self.flag = conv_param.dynamic_flag
        self.var_map = conv_param.dyn_var_map
        self.tiling_dict_flag = tiling_dict_flag
        self.tiling_case = tiling_case
        self.var_range = var_range

        #==========combined parameters==================
        self.h_dynamic = "fmap_h" in self.var_map
        self.w_dynamic = "fmap_w" in self.var_map
        self.hw_dynamic = self.h_dynamic or self.w_dynamic
        self.n_dynamic = "batch_n" in self.var_map

    def fetch_tiling_case(self):
        """
        Fetch tiling case in dynamic shape.
        """
        return self.tiling_case

    def handle_var_range(self, sch):
        """
        Set var range for hi, ho, wi, wo, batch.
        """
        var_range = self.var_range
        var_map = self.var_map

        if self.h_dynamic:
            fmap_h_range = var_range['fmap_h']
            ho_range = var_range['ho']
            sch.set_var_range(var_map['fmap_h'], fmap_h_range[0], fmap_h_range[1])
            sch.set_var_range(var_map['ho'], ho_range[0], ho_range[1])

        if self.w_dynamic:
            fmap_w_range = var_range['fmap_w']
            wo_range = var_range['wo']
            sch.set_var_range(var_map['fmap_w'], fmap_w_range[0], fmap_w_range[1])
            sch.set_var_range(var_map['wo'], wo_range[0], wo_range[1])

        if self.n_dynamic:
            batch_range = var_range['batch_n']
            sch.set_var_range(var_map['batch_n'], batch_range[0], batch_range[1])

    def check_dynamic_overhead_opt_flag(self, tiling):
        """
        Fmap overhead opti is not supported when hi or wi is dynamic.
        """
        if self.hw_dynamic and tiling["A_overhead_opt_flag"]:
            err_man.raise_err_value_or_format_invalid(
                "conv2d", 'tiling["A_overhead_opt_flag"]', "False", "when dynamic shape.")

    def disable_memory_reuse(self, sch, tensor_param):
        """
        Disable memory reuse in dynamic situation.
        """
        if self.flag:
            al1 = tensor_param["al1"]
            bl1 = tensor_param["bl1"]
            al0 = tensor_param["al0"]
            bl0 = tensor_param["bl0"]
            cl0 = tensor_param["cl0"]

            # sequential_malloc
            sch.sequential_malloc(cce_params.scope_cbuf)
            sch.sequential_malloc(cce_params.scope_ca)
            sch.sequential_malloc(cce_params.scope_cb)
            sch.sequential_malloc(cce_params.scope_cc)

            # mem_unique
            sch[al1].mem_unique()
            sch[al0].mem_unique()
            if bl1 is not None:
                sch[bl1].mem_unique()
            sch[bl0].mem_unique()
            sch[cl0].mem_unique()

    def set_al1_bound(self, sch, al1, conv_param, tiling_param, l0a_load2d_flag, strideh_opti_flag):
        """
        Set al1 bound for dynamic shape.
        """
        def modify_m_for_load3d():
            ho_len = tvm.floordiv(m_al1, out_width) + additional_rows
            hi_max = kernel_h + (ho_len - 1)*stride_h_update
            return hi_max*in_width

        if self.flag:
            _, in_c1, in_height, in_width, in_c0 = tiling_param["fmap_5hd_shape"]
            al1_tiling = tiling_param["al1_tiling"]
            stride_h_update = tiling_param["stride_h_update"]
            m_cl0 = tiling_param["m_cl0"]

            stride_h = conv_param.stride_h
            kernel_h = conv_param.filter_h
            out_width = conv_param.w_out

            if al1_tiling:
                multi_m_al1 = tiling_param["multi_m_al1"]
                k1_al1 = tiling_param["k1_al1"]
                m_al1 = multi_m_al1 * m_cl0
                if l0a_load2d_flag:
                    pass
                else:
                    if self.hw_dynamic:
                        additional_rows = tvm.select(
                            tvm.floormod(m_al1, out_width) == 0,
                            0,
                            tvm.select(tvm.floormod(m_al1 * 2, out_width) == 0, 1, 2))
                    elif self.n_dynamic:
                        if m_al1 % out_width == 0:
                            additional_rows = 0
                        elif m_al1 * 2 % out_width == 0:
                            additional_rows = 1
                        else:
                            additional_rows = 2
                    m_al1 = modify_m_for_load3d()
                al1_bound = m_al1 * k1_al1 * in_c0
            else:
                if strideh_opti_flag:
                    in_height = (in_height - 1) // stride_h + 1
                m_al1 = ceil(in_height * in_width, in_c0)
                al1_bound = m_al1 * in_c1 * in_c0

            sch[al1].set_buffer_size(al1_bound)

    def set_cl0_bound(self, sch, cl0, cl0_tiling):
        """
        Set storage bound for CL0 in dynamic shape
        to solve the memory allocate problem of using storage_align for CL0.
        """
        if self.flag:
            sch[cl0].set_buffer_size(reduce((lambda x, y: x*y), cl0_tiling))

    def res_hw_dynamic_pragma(self, sch, res, res_pragma_axis):
        """
        Pragma for res when hw dynamic.
        """
        if self.hw_dynamic:
            sch[res].pragma(res_pragma_axis, "gm_no_sync", 1)

    @staticmethod
    def dynamic_mode_im2col_v2(sch, conv_param, tensor_param, tiling_param,
                               emit_insn_dict, input_nd_flag, l0a_load2d_flag):
        """
        Use im2col_v2 in dynamic shape situation.
        """
        stride_h_update = tiling_param["stride_h_update"]
        fmap = tensor_param["fmap"]
        al1 = tensor_param["al1"]
        al0 = tensor_param["al0"]
        dynamic_al0_pragma_axis = emit_insn_dict["dynamic_al0_pragma_axis"]

        im2col_attr = {
            'set_fmatrix': 1,
            'conv_kernel_h': conv_param.filter_h,
            'conv_kernel_w': conv_param.filter_w,
            'conv_padding_top': conv_param.padding[0],
            'conv_padding_bottom': conv_param.padding[1],
            'conv_padding_left': conv_param.padding[2],
            'conv_padding_right': conv_param.padding[3],
            'conv_stride_h': stride_h_update,
            'conv_stride_w': conv_param.stride_w,
            'conv_fm_c': fmap.shape[4]*fmap.shape[1],
            'conv_fm_c1': fmap.shape[1],
            'conv_fm_h': fmap.shape[2],
            'conv_fm_w': fmap.shape[3],
            'conv_fm_c0': fmap.shape[4],
        }
        im2col_attr_0 = {
            'set_fmatrix': 0,
            'conv_kernel_h': conv_param.filter_h,
            'conv_kernel_w': conv_param.filter_w,
            'conv_padding_top': conv_param.padding[0],
            'conv_padding_bottom': conv_param.padding[1],
            'conv_padding_left': conv_param.padding[2],
            'conv_padding_right': conv_param.padding[3],
            'conv_stride_h': stride_h_update,
            'conv_stride_w': conv_param.stride_w,
            'conv_fm_c': fmap.shape[4]*fmap.shape[1],
            'conv_fm_c1': fmap.shape[1],
            'conv_fm_h': fmap.shape[2],
            'conv_fm_w': fmap.shape[3],
            'conv_fm_c0': fmap.shape[4],
            'group_flag': 1,
            'l1_group_flag': 1
        }

        if l0a_load2d_flag:
            sch[al1].emit_insn(al1.op.axis[0], "dma_copy")
        elif input_nd_flag:
            im2col_attr.update({"layout_transform": "nd2nz"})
            sch[al1].emit_insn(al1.op.axis[2], "dma_copy", im2col_attr)
        else:
            sch[al1].emit_insn(al1.op.axis[0], "dma_copy", im2col_attr)

        if l0a_load2d_flag:
            sch[al0].emit_insn(dynamic_al0_pragma_axis, "dma_copy")
        else:
            sch[al0].emit_insn(dynamic_al0_pragma_axis, 'im2col_v2', im2col_attr_0)


class StridedRead:
    """
    Class of StridedRead + Conv2d fusion.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.strided_read_flag

    def process_strided_read(self, sch, al1, strideh_opti_flag, l0a_load2d_flag):
        """
        Inline the output tensor of strided read when strideh_opti or l0a_load2d is enabled.
        """
        def inline_fmap_strided_read(fusion_flag, al1):
            if fusion_flag and self.flag:
                fmap_strided_read = get_src_tensor(al1)
                sch[fmap_strided_read].compute_inline()

        inline_fmap_strided_read(strideh_opti_flag, al1)
        inline_fmap_strided_read(l0a_load2d_flag, al1)


class StridedWrite:
    """
    Class of Conv2d + StridedWrite fusion.
    """
    def __init__(self, res):
        self.flag = res.op.tag == "strided_write"

    def process_strided_write(self, sch, res):
        """
        Inline the output tensor of strided write.
        """
        if self.flag:
            strided_write_src = get_src_tensor(res)
            res_hw, res_c0 = res.shape[-2:]
            sch[strided_write_src].compute_inline()
            sch[res].bind_buffer(res.op.axis[0], res_hw * res_c0 * res.op.attrs['stride'], 0)


class Im2colDma:
    """
    Class of Im2col dma when load3d exceeds L1 size.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.l0a_dma_flag

    def config_al1_im2col(self, sch, tensor_map):
        """
        Get the im2col_fractal tensor in L1.
        """
        if self.flag:
            al1_im2col = tensor_map["fmap_im2col"]
            sch[al1_im2col].set_scope(cce_params.scope_cbuf)
        else:
            al1_im2col = None
        return al1_im2col

    @staticmethod
    def config_al0_im2coldma(sch, al1_im2col, cl0):
        """
        Cache read al1_im2col into L0.
        """
        al0 = sch.cache_read(al1_im2col, cce_params.scope_ca, [cl0])
        return al0

    def align_al1_im2col(self, sch, al1_im2col, block_k0):
        """
        Buffer align al1_im2col.
        """
        if self.flag:
            sch[al1_im2col].buffer_align(
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, block_k0))

    def inline_al1_im2coldma(self, sch, al1, fmap_row_major):
        """
        Inline al1 and row major tensor.
        """
        if self.flag:
            sch[al1].compute_inline()
            sch[fmap_row_major].compute_inline()

    @staticmethod
    def im2col_dma_emit_insn(sch, al1_im2col, al0, al0_axis_list):
        """
        Emit insn for al1_im2col and al0.
        """
        sch[al1_im2col].emit_insn(al1_im2col.op.axis[5], "dma_copy")
        sch[al0].emit_insn(al0_axis_list[0], "dma_copy")


class InnerBatch:
    """
    Class of l0b innerbatch.
    """
    def __init__(self):
        self.flag = False # inner_batch of sharing L0B weight

    def set_l0b_innerbatch_flag(self, tiling, batch_cl0):
        """
        Set l0b innerbatch flag by tiling.
        """
        if batch_cl0 > 1 and tiling["BL1_shape"] is None:
            self.flag = True
            log.debug("enable innerbatch")

    def config_innerbatch_axis(self, batch_al1, batch_cl0):
        """
        Config innerbatch case batch inner split axis.
        """
        if self.flag:
            return batch_cl0
        return batch_al1

    def check_innerbatch_tiling(self, tiling, batch, out_hw):
        """
        Check the tiling of l0b innerbatch.
        """
        if self.flag:
            _, mc_cl0, m0_cl0, _, batch_cl0, _ = tiling["CL0_matrix"]
            batch_dim, _, _, _ = tiling["block_dim"]
            if mc_cl0 * m0_cl0 != ceil(out_hw, m0_cl0):
                err_man.raise_err_specific("conv2d", "innerbatch case must full load M.")
            if batch_cl0 * batch_dim > batch:
                err_man.raise_err_specific("conv2d", "innerbatch batch_cl0*batch_dim cannot be greater than batch.")


class L0aLoad2d:
    """
    class of fmap load2d optimization.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.l0a_load2d_flag

    def align_al1_load2d(self, sch, al1):
        """
        al1 M align.
        """
        if self.flag:
            sch[al1].storage_align(al1.op.axis[1], 256, 0)

    @staticmethod
    def load2d_emit_insn(sch, al1, al0):
        """
        Emit insn for al1 and al0 when al1_load2d is enabled.
        """
        sch[al1].emit_insn(al1.op.axis[0], "dma_copy")
        sch[al0].emit_insn(al0.op.axis[0], "dma_copy")


class StridehOpti:
    """
    class of stride_h optimization when kernel_h = 1 and stride_h > 1.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.strideh_opti_flag
        self.stride_h_update = 1 if self.flag else conv_param.stride_h


class C04Opti:
    """
    Class of C04 optimization.
    """
    def __init__(self, conv_param):
        self.mode = conv_param.v220_c04_mode
        self.flag = conv_param.v220_c04_mode != "disabled"


class Conv1dSplitw:
    """
    Class of Conv1d.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.conv1d_split_w_flag

    @staticmethod
    def align_row_major_conv1d(sch, fmap_row_major, block_k0):
        """
        Buffer align row major in Conv1d.
        """
        sch[fmap_row_major].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, block_k0))


class Conv2dSchedule:
    """
    Class of Conv2d Schedule.
    """
    def __init__(self, sch, res, spec_node_list, conv_param, op_graph, tiling_dict_flag, tiling_case, var_range):
        self._sch = sch
        self._res = res
        self._conv_param = conv_param
        self._op_graph = op_graph

        self._tensor_map = conv_param.tensor_map
        self._dim_map = conv_param.dim_map
        self._para_dict = conv_param.para_dict

        # dtype
        self._fmap_dtype = self._tensor_map["fmap"].dtype
        self._weight_dtype = self._tensor_map["filter"].dtype
        self._res_dtype = res.dtype

        # frac unit size
        self._block_m0, self._block_k0, self._block_n0 = CUBE_MKN[self._weight_dtype]["mac"]

        # conv2d params
        self._kernel_h = conv_param.filter_h
        self._kernel_w = conv_param.filter_w
        self._stride_h = conv_param.stride_h
        self._stride_w = conv_param.stride_w
        self._dilate_h = conv_param.dilate_h
        self._dilate_w = conv_param.dilate_w

        self._out_height = conv_param.h_out
        self._out_width = conv_param.w_out
        self._out_hw = self._out_height*self._out_width

        self._group_opt = self._para_dict["group_opt"]
        self._ci1_opt = self._para_dict["c1_opt"]
        self._co1_opt = self._para_dict["cout1_opt"]
        self._batch, self._in_c1, self._in_height, self._in_width, self._in_c0 = self._para_dict["a_shape"]
        self._quant_fusion_muti_groups_in_cl0 = self._co1_opt % 2 == 1 and self._group_opt > 1 and res.dtype == "int8"
        self._bias_tensor = self._para_dict["bias_tensor"]
        self._bias_flag = self._bias_tensor is not None

        # self._tiling params
        self._tiling_query_param = conv_param.tiling_query_param
        self._tiling = {}
        self._fusion_type = op_graph.fusion_type

        # device params
        self._core_num = get_soc_spec("CORE_NUM")

        #===========================parse ub fusion=======================================
        self._eltwise_ub_fusion = EltwiseUBFusion(res, op_graph, conv_param)
        self._fixpipe_res = self._eltwise_ub_fusion.cub if self._eltwise_ub_fusion.flag else res

        self._quant_fusion = QuantFusion(res, op_graph)
        #====================create feature instance=============================
        self._fixpipe_fusion = FixpipeFusionNew(self._fixpipe_res) if is_support_fixpipe_op() \
            else FixpipeFusion(self._fixpipe_res)
        self._input_nd2nz = InputNd2Nz(conv_param)
        self._weight_nd2nz = WeightNd2Nz(conv_param)
        self._output_nz2nd = OutputNz2Nd(res)
        self._lx_fusion = LxFusion(conv_param)
        self._aipp_fusion = AippFusion(conv_param)
        self._dynamic_shape = DynamicShape(conv_param, tiling_dict_flag, tiling_case, var_range)
        self._strided_read = StridedRead(conv_param)
        self._strided_write = StridedWrite(res)
        self._im2col_dma = Im2colDma(conv_param)
        self._inner_batch = InnerBatch()
        self._l0a_load2d = L0aLoad2d(conv_param)
        self._strideh_opti = StridehOpti(conv_param)
        self._c04 = C04Opti(conv_param)
        self._conv1d = Conv1dSplitw(conv_param)
        self._pooling_fusion = MaxpoolFusion(res, MaxPoolParam)
        self._convbn1 = Conv2dBN1Fusion(conv_param, self._fmap_dtype, op_graph, self._eltwise_ub_fusion.cub)

        #===================parse fusion pattern========================
        self._fixpipe_fusion.parse_fusion_pattern()

        # multi_out specify
        if len(spec_node_list) > 1:
            self._multi_out = spec_node_list[:-1]
        else:
            self._multi_out = None

        if self._aipp_fusion.flag:
            self._fmap_dtype = "float16"

    def fetch_info_dict(self, tiling_case):
        """
        Fetch the info_dict to get tiling.
        """
        def cal_cub_coeff():
            """
            get cub space coefficient for get_tiling.
            """
            eltwise_coeff, channelwise_coeff, scalar_num = self._eltwise_ub_fusion.coeff_eltwise_cal(self._res)
            eltwise_coeff += self._quant_fusion.cal_quant_coeff()
            eltwise_coeff += self._pooling_fusion.cal_pooling_coeff()
            return eltwise_coeff, channelwise_coeff, scalar_num

        def get_info_dict_cdtype(c_dtype):
            """
            get info_dict cdtype
            """
            if self._convbn1.flag:
                return self._convbn1.get_info_dict_cdtype()
            return c_dtype

        def get_info_dict_fusion_type(fusion_type):
            """
            get info_dict fusion_type
            """
            if self._convbn1.flag:
                return self._convbn1.fusion_type
            return fusion_type

        if self._dynamic_shape.flag and tiling_case: # pass when tiling_case
            return None

        tiling_query_param = self._tiling_query_param
        conv_param = self._conv_param

        fmap_shape_nc1hwc0 = list(tiling_query_param["fmap_shape_nc1hwc0"])
        shape_w_nc1hwc0 = list(tiling_query_param["shape_w_nc1hwc0"])
        c_shape = tiling_query_param["c_shape"]
        mad_dtype = tiling_query_param["mad_dtype"]
        bias_flag = tiling_query_param["bias_flag"]
        quant_pre_flag, relu_pre_flag, quant_post_flag, relu_post_flag, anti_quant_flag = \
            self._fixpipe_fusion.fetch_quant_relu_flag()
        eltwise_flag = False
        eltwise_dtype = "float16"
        if is_support_fixpipe_op():
            eltwise_flag, eltwise_dtype = self._fixpipe_fusion.fetch_eltwise_info()

        eltwise_coeff, channelwise_coeff, scalar_num = cal_cub_coeff()

        pooling_shape, pooling_stride = self._pooling_fusion.config_window_stride()

        c_dtype = get_info_dict_cdtype(self._res_dtype)
        fusion_type = get_info_dict_fusion_type(self._fusion_type)
        # group conv, send one group_opt a, b, c shape to tiling
        info_dict = {"op_type": 'conv2d',
                     "a_shape": fmap_shape_nc1hwc0,
                     "b_shape": shape_w_nc1hwc0,
                     "c_shape": c_shape,
                     "a_dtype": self._fmap_dtype,
                     "b_dtype": self._weight_dtype,
                     "c_dtype": c_dtype,
                     "mad_dtype": mad_dtype,
                     "pad": [conv_param.pad_w[0], conv_param.pad_w[1],
                             conv_param.pad_h[0], conv_param.pad_h[1]],
                     "stride": [self._stride_h, self._stride_w],
                     "dilation": [self._dilate_h, self._dilate_w],
                     "group": self._group_opt,
                     "bias_flag": bias_flag,
                     "fixpipe_fusion_flag_dict": {"quant_pre_flag": quant_pre_flag,
                                                  "relu_pre_flag": relu_pre_flag,
                                                  "quant_post_flag": quant_post_flag,
                                                  "relu_post_flag": relu_post_flag,
                                                  "anti_quant_flag": anti_quant_flag},
                     "eltwise_dict": {
                         "eltwise_flag": eltwise_flag,
                         "eltwise_dtype": eltwise_dtype
                     },
                     "in_fm_memory_type": self._lx_fusion.input_memory_type,
                     "out_fm_memory_type": self._lx_fusion.output_memory_type,
                     "l1_fusion_type": self._lx_fusion.l1_fusion_type,
                     "fm_l1_valid_size": self._lx_fusion.fmap_l1_valid_size,
                     "fusion_type": fusion_type,
                     "kernel_name": conv_param.kernel_name,
                     "special_mode": {"use_c04_mode": 2 if self._c04.flag else 0, # 3 for v220 c04
                                      # disable strideh opti when input nd2nz.
                                      "input_nd_flag": self._input_nd2nz.flag,
                                      "scalar_num": scalar_num,
                                     },
                     "placeholder_fmap_5hd_shape": list(self._dim_map["fmap_5hd_shape"]),
                     "fused_coefficient": [0, 0, eltwise_coeff],
                     "fused_channel_wise": [0, 0, channelwise_coeff],
                     "pooling_shape": pooling_shape,
                     "pooling_stride": pooling_stride,
                    }
        return info_dict

    def fetch_tiling(self, info_dict, tiling_case):
        """
        Fetch tiling info.
        """
        def get_default_tiling():
            """
            Set default tiling when fetch tiling failed.
            """
            default_tiling = {}
            tiling_m = 1
            tiling_k = 1
            if self._quant_fusion_muti_groups_in_cl0:
                tiling_n = self._co1_opt
                group_cl0 = 2
            else:
                tiling_n = 2
                group_cl0 = 1

            default_tiling["AL1_shape"] = [1, 1, 1, 1]
            default_tiling["BL1_shape"] = None

            default_tiling["AL0_matrix"] = [tiling_m, tiling_k, 16, self._in_c0, 1, 1]
            default_tiling["BL0_matrix"] = [tiling_k, tiling_n, 16, self._in_c0, 1, 1]
            default_tiling["CL0_matrix"] = [tiling_n, tiling_m, 16, 16, 1, group_cl0]
            default_tiling["CUB_matrix"] = [tiling_n, tiling_m, 16, 16, 1, 1]
            default_tiling["AUB_shape"] = [1, 1, 1, 1]
            default_tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                                        'BL1_pbuffer': 1,
                                                        'AL0_pbuffer': 1,
                                                        'BL0_pbuffer': 1,
                                                        'CL0_pbuffer': 1,
                                                        'AUB_pbuffer': 1,
                                                        'BUB_pbuffer': 1,
                                                        'CUB_pbuffer': 1,
                                                        'UBG_pbuffer': 1}
            default_tiling["n_bef_batch_flag"] = False
            default_tiling["A_overhead_opt_flag"] = False
            default_tiling["B_overhead_opt_flag"] = False
            default_tiling["CUB_channel_wise_flag"] = True

            default_tiling["block_dim"] = [1, 1, 1, 1]
            if self._batch > 1 and self._core_num > 1:
                if self._batch <= self._core_num:
                    default_tiling["block_dim"][0] = self._batch
                else:
                    for i in range(self._core_num, 0, -1):
                        if self._batch % i == 0:
                            break
                    default_tiling["block_dim"][0] = i
            else:
                default_tiling["block_dim"][0] = 1

            default_tiling = self._lx_fusion.config_default_tiling(default_tiling)

            return default_tiling

        def get_v220_tiling(info_dict):
            """
            Get tiling in v220 situation.
            """
            tiling = tiling_api.get_tiling(info_dict)
            if tiling is None or tiling["AL0_matrix"][2] == 32:
                log.warn("get invalid tiling, default tiling will be used")
                tiling = get_default_tiling()

            return tiling

        if self._dynamic_shape.flag and tiling_case:
            self._tiling = self._dynamic_shape.fetch_tiling_case()
        else:
            self._tiling = get_v220_tiling(info_dict)


    def verify_tiling(self):
        """
        Verify whether the tiling returned is legal.
        """
        def check_l0_tiling():
            """
            Check al0 tiling and bl0 tiling.
            """
            if ma_al0 != mc_cl0:
                err_man.raise_err_equal_invalid("conv2d", "ma", "mc")

            if tiling["BL0_matrix"]:
                kb_bl0, nb_bl0, _, _, _, _ = tiling["BL0_matrix"]
                if ka_al0 != kb_bl0:
                    err_man.raise_err_equal_invalid("conv2d", "ka", "kb")
                if nb_bl0 != nc_cl0:
                    err_man.raise_err_equal_invalid("conv2d", "nb", "nc")

        def check_overhead_opt_flag():
            """
            Check the overhead_opt_flag.
            """
            if group_cl0 > 1 and tiling["A_overhead_opt_flag"]:
                err_man.raise_err_value_or_format_invalid(
                    "conv2d", 'tiling["A_overhead_opt_flag"]', "False", "when multi_cl0_group.")

        def modify_bl0_tiling():
            """
            Modify bl0 tiling in certain circumstances.
            """
            filter_matrix = list(self._dim_map["filter_matrix_dim"]) # [k1, n1, n0, k0]
            filter_matrix[1] = filter_matrix[1] // n_dim

            if self._tiling["BL0_matrix"]:
                if self._tiling["BL0_matrix"][0: 4] == filter_matrix and group_bl0 == 1:
                    self._tiling["BL0_matrix"] = []

        def modify_bl1_tiling():
            """
            Modify bl1 tiling in certain circumstances.
            """
            if self._tiling["BL0_matrix"] == []:
                if ceil_div(self._group_opt, group_dim) == 1:
                    self._tiling["BL1_shape"] = None
                elif ceil_div(self._group_opt, group_dim) > 1 and self._tiling["BL1_shape"] is not None:
                    self._tiling["BL1_shape"] = []

        def check_cub_tiling():
            """
            check cub tiling in no ub fusion situation.
            """
            if not self._eltwise_ub_fusion.flag and tiling["CUB_matrix"] != tiling["CL0_matrix"]:
                err_man.raise_err_specific("conv2d", "CUB_matrix must be equal to CL0_matrix in no ub fusion cases!")

        tiling = self._tiling
        bl0_tiling = tiling["BL0_matrix"]
        ma_al0, ka_al0, _, _, _, _ = tiling["AL0_matrix"]
        nc_cl0, mc_cl0, _, _, batch_cl0, group_cl0 = tiling["CL0_matrix"]
        if bl0_tiling:
            _, _, _, _, _, group_bl0 = bl0_tiling

        _, n_dim, _, group_dim = tiling["block_dim"]

        self._lx_fusion.check_l1fusion_tiling(tiling)

        self._weight_nd2nz.check_bl1_nd2nz_tiling(tiling)

        self._dynamic_shape.check_dynamic_overhead_opt_flag(tiling)

        self._inner_batch.set_l0b_innerbatch_flag(tiling, batch_cl0)
        self._inner_batch.check_innerbatch_tiling(tiling, self._batch, self._out_hw)

        check_l0_tiling()
        check_overhead_opt_flag()
        check_cub_tiling()

        #==================modify tiling to be deleted=========================
        modify_bl0_tiling()
        modify_bl1_tiling()

        tiling = self._pooling_fusion.modify_tiling(
            tiling, self._dim_map["filter_matrix_dim"], self._out_width, self._conv_param, self._block_k0)

    def config_scope(self):
        """
        Config tensor scope.
        """
        def config_cl0():
            """
            Config cl0 scope.
            """
            cl0 = tensor_map["cl0"]
            sch[cl0].set_scope(cce_params.scope_cc)
            return cl0

        def config_fmap_row_major():
            """
            Config row major scope.
            """
            if self._dynamic_shape.flag or self._l0a_load2d.flag:
                return None
            fmap_row_major = tensor_map["fmap_row_major"]
            sch[fmap_row_major].set_scope(cce_params.scope_cbuf)
            return fmap_row_major

        def config_al1():
            """
            Config al1 scope.
            """
            scope_al1 = self._lx_fusion.config_al1_scope()
            al1_already_exist_flags = (self._dynamic_shape.flag,
                                       self._l0a_load2d.flag,
                                       self._strideh_opti.flag,
                                       self._input_nd2nz.flag,
                                       self._strided_read.flag,
                                       self._aipp_fusion.flag,
                                       self._c04.mode == "not_first_layer_c04")
            for flag in al1_already_exist_flags:
                if flag:
                    al1 = tensor_map["fmap_l1"]
                    sch[al1].set_scope(scope_al1)
                    return al1

            al1 = sch.cache_read(fmap, scope_al1, [fmap_row_major])
            return al1

        def config_al0():
            """
            Config al0 scope.
            """
            if self._im2col_dma.flag:
                return self._im2col_dma.config_al0_im2coldma(sch, al1_im2col, cl0)

            al0 = tensor_map["fmap_im2col"]
            sch[al0].set_scope(cce_params.scope_ca)
            return al0

        def config_bl1():
            """
            Config bl1 scope.
            """
            if self._tiling["BL1_shape"] is None:
                bl1 = None
            elif self._weight_nd2nz.flag:
                bl1 = weight
                sch[bl1].set_scope(cce_params.scope_cbuf)
            else:
                bl1 = sch.cache_read(weight, cce_params.scope_cbuf, [cl0])
            return bl1

        def config_bl0():
            """
            Config bl0 scope.
            """
            if self._tiling["BL1_shape"] is None:
                bl0 = sch.cache_read(weight, cce_params.scope_cb, [cl0])
            else:
                bl0 = sch.cache_read(bl1, cce_params.scope_cb, [cl0])
            return bl0

        def config_bias():
            """
            Config bias scope.
            """
            if self._bias_flag:
                bias_l1 = tensor_map["bias_l1"]
                sch[bias_l1].set_scope(cce_params.scope_cbuf)
                bias_bt = tensor_map["bias_bt"]
                sch[bias_bt].set_scope(cce_params.scope_bt)
                return bias_l1, bias_bt

            return None, None

        #========set scope && cache_read && cache_write==========
        tensor_map = self._tensor_map
        sch = self._sch

        fmap = tensor_map["fmap"]
        weight = tensor_map["filter"]
        fmap_row_major_reshape = tensor_map.get("fmap_row_major_reshape", None)
        al1_im2col = self._im2col_dma.config_al1_im2col(sch, tensor_map)

        cl0 = config_cl0()
        fmap_row_major = config_fmap_row_major()
        al1 = config_al1()
        al0 = config_al0()
        bl1 = config_bl1()
        bl0 = config_bl0()
        bias_l1, bias_bt = config_bias()

        self._fixpipe_fusion.fixpipe_inputs_set_scope(sch, self._op_graph)

        self._eltwise_ub_fusion.cub_set_scope(sch)
        self._eltwise_ub_fusion.inputs_cache_read(sch, self._op_graph)
        self._eltwise_ub_fusion.res_cache_write(sch, self._res)

        self._quant_fusion.quant_tensors_set_scope(sch)

        self._pooling_fusion.set_maxpool_ub_scope(sch, self._op_graph.body_ops)

        tensor_param = {"al1": al1, "bl1": bl1,
                        "fmap": fmap, "weight": weight,
                        "fmap_row_major": fmap_row_major, "fmap_row_major_reshape": fmap_row_major_reshape,
                        "al1_im2col": al1_im2col,
                        "al0": al0, "bl0": bl0, "cl0": cl0,
                        "bias_l1": bias_l1, "bias_bt": bias_bt}
        return tensor_param

    def special_process_pre(self, res, tensor_param):
        """
        Special process before tiling is parsed.
        """
        def align_al1():
            """
            Align al1 in various situation.
            """
            if self._lx_fusion.l1_fusion_type == BREADTH_L1_FUSION:
                return self._lx_fusion.align_al1_lxfusion(sch, al1)
            if self._l0a_load2d.flag:
                return self._l0a_load2d.align_al1_load2d(sch, al1)
            if self._pooling_fusion.flag:
                return self._pooling_fusion.align_al1_pooling(sch, al1)
            return None

        def align_row_major():
            """
            Align row major in various situation.
            """
            if self._dynamic_shape.flag or self._l0a_load2d.flag:
                return None
            if self._conv1d.flag:
                return self._conv1d.align_row_major_conv1d(sch, fmap_row_major, self._block_k0)

            sch[fmap_row_major].buffer_align(
                (1, 1),
                (1, 1),
                (self._out_width, self._out_width),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 4 if self._c04.flag else self._block_k0))
            return None

        sch = self._sch
        al1 = tensor_param["al1"]
        fmap_row_major = tensor_param["fmap_row_major"]
        fmap_row_major_reshape = tensor_param["fmap_row_major_reshape"]
        al1_im2col = tensor_param["al1_im2col"]

        align_al1()
        align_row_major()
        if not self._convbn1.flag:
            self._fixpipe_fusion.inline_fixpipe_tensor(sch)
        if res.op.name == "res_fp32_conv2d" and is_support_fixpipe_op():
            # for single conv2d inline res_conv2d
            sch[res.op.input_tensors[0]].compute_inline()
        # bn1 fusion
        self._convbn1.bn1fusion_special_process_pre(sch)

        # inline row_major_reshape
        if fmap_row_major_reshape is not None:
            sch[fmap_row_major_reshape].compute_inline()

        self._im2col_dma.inline_al1_im2coldma(sch, al1, fmap_row_major)
        self._im2col_dma.align_al1_im2col(sch, al1_im2col, self._block_k0)


        self._strided_read.process_strided_read(sch, al1, self._strideh_opti.flag, self._l0a_load2d.flag)
        self._strided_write.process_strided_write(sch, self._res)

        # inline input_nd
        self._input_nd2nz.inline_input_nd_dynamic(sch, self._tensor_map, self._dynamic_shape.flag)

        # quant fusion
        self._quant_fusion.inline_input_ub(sch)

        # ub fusion
        self._eltwise_ub_fusion.ub_tensors_inline(sch)

        # dynamic shape
        self._dynamic_shape.handle_var_range(sch)
        self._dynamic_shape.disable_memory_reuse(sch, tensor_param)

    def tile_attach_tensor(self, res, sch_list, tensor_param):
        """
        Split tensor axis and attach tensors.
        """
        def tile_tensor_al0():
            """
            tile al0 for load3d emit insn
            """
            al0_mo, al0_mi = sch[al0].split(al0.op.axis[2], ma_al0)
            al0_ko, al0_ki = sch[al0].split(al0.op.axis[3], ka_al0)
            al0_no, al0_ni = sch[al0].split(al0.op.axis[1], 1)

            sch[al0].reorder(al0.op.axis[0],  # group
                             al0_no,  # batch.outer
                             al0_mo,  # m_1.outer
                             al0_ko,  # k_1.outer
                             al0_ni,  # batch.inner = 1
                             al0_mi,  # m_1.inner
                             al0_ki,  # k_1.inner
                             al0.op.axis[4],  # m_0
                             al0.op.axis[5])  # k_0
            al0_axis_list = [al0_no, al0_mo, al0_ko,
                             al0_ni, al0_mi, al0_ki,
                             al0.op.axis[4], al0.op.axis[5]]  # axis for im2col
            dynamic_al0_pragma_axis = al0_ni
            return al0_axis_list, dynamic_al0_pragma_axis

        def get_reorder_mn_flag():
            """
            get_reorder_mn_flag
            """
            if not bl1_tiling:
                return True
            if pingpong_buffer["AL1_pbuffer"] == pingpong_buffer["BL1_pbuffer"]:
                if not self._dynamic_shape.flag and bl1_nparts[1] >= al1_nparts[1]:
                    return True
                return False
            if pingpong_buffer["BL1_pbuffer"] == 2:
                return True
            return False

        def tile_tensor_res():
            """
            tile tensor res
            """
            def set_reorder_mn_flag():
                """
                Reorder axis m and n to achieve better performance.
                """
                def reorder_res_mn_axis():
                    """
                    Reorder axis m and n according to various flags.
                    """
                    if self._inner_batch.flag:
                        if reorder_mn_flag:
                            sch[res].reorder(
                                out2al1_loopbatch_axis,
                                singlecore_out2al1_loopm_axis,
                                res_batch_1_axis,
                                singlecore_out2bl1_loopn_axis
                                )
                        else:
                            sch[res].reorder(
                                out2al1_loopbatch_axis,
                                singlecore_out2bl1_loopn_axis,
                                singlecore_out2al1_loopm_axis,
                                res_batch_1_axis
                                )
                    else:
                        # singlecore_out2bl1_loopn_axis means nparts of co1 axis loading into L1 in single core. (N axis)
                        # singlecore_out2al1_loopm_axis means nparts of howo axis loading into L1 in single core. (M axis)
                        if reorder_mn_flag:
                            sch[res].reorder(
                                out2al1_loopbatch_axis,
                                singlecore_out2al1_loopm_axis,
                                res_batch_1_axis,
                                singlecore_out2bl1_loopn_axis,
                                bl12bl0_loopn_axis,
                                res_al1_batch_axis
                                )
                        else:
                            sch[res].reorder(
                                out2al1_loopbatch_axis,
                                singlecore_out2bl1_loopn_axis,
                                singlecore_out2al1_loopm_axis,
                                bl12bl0_loopn_axis,
                                res_batch_1_axis,
                                res_al1_batch_axis
                                )

                reorder_mn_flag = get_reorder_mn_flag()

                reorder_res_mn_axis()

                return reorder_mn_flag

            # Only fixpipe fusion. The special axis split operation works on res.
            fixpipe_nz2nd_flag = self._output_nz2nd.flag or \
                (self._fixpipe_fusion.nz2nd_flag and not self._eltwise_ub_fusion.flag)
            fixpipe_channelsplit_flag = res.dtype == "float32" and not self._eltwise_ub_fusion.flag
            fixpipe_channelmerge_flag = res.dtype in ("int4", "int8") and not self._eltwise_ub_fusion.flag
            fixpipe_antiquant_flag = anti_quant_spilt_flag(res)

            if fixpipe_nz2nd_flag:
                fixpipe_channelsplit_flag = False
                fixpipe_channelmerge_flag = False

            special_axis_dict = {}
            dtype_coeff = {
                "int4": 4,
                "int8": 2,
            }

            def fetch_base_axis():
                """
                Fetch axes of the res tensor.
                """
                if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
                    res_n_axis, res_hw_axis, res_c_axis = res.op.axis  # [n, howo, co]
                    # split c axis into c1 and c0 to avoid nonlinear ir
                    res_c1_axis, res_c0_axis = sch[res].split(res_c_axis, 16)  # [n, howo, co1, co0]
                else:
                    res_n_axis, res_c1_axis, res_hw_axis, res_c0_axis = res.op.axis  # [n, co1, howo, co0]

                return res_n_axis, res_c1_axis, res_hw_axis, res_c0_axis

            def cal_co1_opt_factor():
                """
                Calculate the co1_opt factor to split out group axis.
                """
                if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
                    co1_opt_factor = co1_opt
                elif res.dtype in ("int4", "int8"):
                    if multi_cl0_group:  # group_cl0 is even while co1_opt is odd.
                        co1_opt_factor = ceil_div(co1_opt * multi_cl0_group, dtype_coeff[res.dtype])
                    else:
                        co1_opt_factor = ceil_div(co1_opt, dtype_coeff[res.dtype])
                elif res.dtype == "float32":
                    co1_opt_factor = co1_opt * 2
                elif res.dtype in ("float16", "bfloat16", "int32"):
                    co1_opt_factor = co1_opt
                else:
                    err_man.raise_err_specific("conv2d", "res dtype is not supported!")

                return co1_opt_factor

            def cal_nc_cl0_factor():
                """
                Fetch nc_cl0 for various tiling situation.
                """
                nc_cl0_factor = nc_cl0
                if res.dtype in ("int4", "int8") and not (self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag):
                    if multi_cl0_group:
                        nc_cl0_factor = co1_opt * group_cl0 // dtype_coeff[res.dtype]
                    else:
                        nc_cl0_factor = nc_cl0 // dtype_coeff[res.dtype]
                return nc_cl0_factor

            def split_group_opt_axis(co1_opt_factor):
                """
                Split out group_opt axis and co1_opt axis.
                """
                res_group_opt_axis, res_co1_opt_axis = sch[res].split(res_c1_axis, factor=co1_opt_factor)

                if fixpipe_channelsplit_flag:
                    res_co1_opt_axis_ori = res_co1_opt_axis
                    res_co1_opt_axis, res_c0_npart_axis = sch[res].split(res_co1_opt_axis_ori, 2)
                    special_axis_dict["fixpipe_channelsplit_res_c0_npart_axis"] = res_c0_npart_axis

                return res_group_opt_axis, res_co1_opt_axis

            def split_nc_cl0_axis(nc_cl0_factor):
                """
                Split out nc_cl0_axis.
                """
                out2cl0_loopn_axis, res_nc_cl0_axis = sch[res].split(res_co1_opt_axis, factor=nc_cl0_factor)

                if fixpipe_antiquant_flag:
                    # fp16 and not nd2nz and anti_quant, split 2 for pass claculate c0 stride
                    res_nc_cl0_axis_ori = res_nc_cl0_axis
                    res_nc_cl0_axis, res_c0_npart_axis = sch[res].split(res_nc_cl0_axis_ori, 2)
                    special_axis_dict["fixpipe_antiquant_res_c0_npart_axis"] = res_c0_npart_axis

                return out2cl0_loopn_axis, res_nc_cl0_axis

            def special_axis_process(res_c0_axis):
                """
                Process special axes in fixpipe fusion situation.
                """
                if fixpipe_channelmerge_flag:
                    # split c0=16 in channel merging to avoid nonlinear ir
                    _, _ = sch[res].split(res_c0_axis, factor=16)

            res_n_axis, res_c1_axis, res_hw_axis, res_c0_axis = fetch_base_axis()

            co1_opt_factor = cal_co1_opt_factor()
            res_group_opt_axis, res_co1_opt_axis = split_group_opt_axis(co1_opt_factor)

            nc_cl0_factor = cal_nc_cl0_factor()
            out2cl0_loopn_axis, res_nc_cl0_axis = split_nc_cl0_axis(nc_cl0_factor)

            special_axis_process(res_c0_axis)

            # split cl0 tiling m axis
            res_m_cl0_factor = mc_cl0 * m0_cl0
            res_m_cl0_factor = self._pooling_fusion.modify_res_m_cl0_factor(res_m_cl0_factor)
            out2cl0_loopm_axis, res_m_cl0_axis = sch[res].split(res_hw_axis, res_m_cl0_factor)

            # split cub tiling axis
            cl02cub_loopn_axis, res_nc_factor_axis = sch[res].split(res_nc_cl0_axis,
                                                                    nparts=ceil_div(nc_cl0, nc_factor_cub))
            cl02cub_loopm_axis, res_m_factor_axis = sch[res].split(res_m_cl0_axis, nparts=1)

            if fixpipe_nz2nd_flag:
                sch[res].reorder(res_group_opt_axis,
                                 out2cl0_loopn_axis,
                                 out2cl0_loopm_axis,
                                 #========cl0 tiling=================
                                 cl02cub_loopn_axis, # nc_cl0 // nc_factor_cub
                                 cl02cub_loopm_axis, # 1
                                 #========cub tiling=================
                                 res_m_factor_axis, # mc_factor_cub*m0_cl0
                                 res_nc_factor_axis) # nc_factor_cub
                # [n, group_opt, co1_opt // nc, howo // (mc*m0),
                # ||| nc // nc_factor, 1,
                # ||| mc_factor*m0, nc_factor, co0]
            elif fixpipe_channelsplit_flag:
                sch[res].reorder(res_group_opt_axis,
                                 out2cl0_loopn_axis,
                                 out2cl0_loopm_axis,
                                 #========cl0 tiling=================
                                 cl02cub_loopn_axis, # nc_cl0 // nc_factor_cub
                                 cl02cub_loopm_axis, # 1
                                 #========cub tiling=================
                                 res_nc_factor_axis, # nc_factor_cub
                                 special_axis_dict["fixpipe_channelsplit_res_c0_npart_axis"],
                                 res_m_factor_axis) # mc_factor_cub*m0_cl0
                # [n, group_opt, co1_opt // 2*nc, howo // (mc*m0),
                # ||| nc // nc_factor, 1,
                # ||| nc_factor, 2, mc_factor*m0, co0]
            elif fixpipe_antiquant_flag:
                sch[res].reorder(out2cl0_loopn_axis, # co1_opt // nc
                                 out2cl0_loopm_axis, # howo // (mc*m0)
                                 #========cl0 tiling=================
                                 cl02cub_loopn_axis, # nc_cl0 // nc_factor_cub
                                 cl02cub_loopm_axis, # 1
                                 #========cub tiling=================
                                 res_nc_factor_axis, # nc_factor_cub // 2
                                 special_axis_dict["fixpipe_antiquant_res_c0_npart_axis"],
                                 res_m_factor_axis) # mc_factor_cub*m0_cl0
            else:
                sch[res].reorder(out2cl0_loopn_axis, # co1_opt // nc
                                 out2cl0_loopm_axis, # howo // (mc*m0)
                                 #========cl0 tiling=================
                                 cl02cub_loopn_axis, # nc_cl0 // nc_factor_cub
                                 cl02cub_loopm_axis, # 1
                                 #========cub tiling=================
                                 res_nc_factor_axis, # nc_factor_cub
                                 res_m_factor_axis) # mc_factor_cub*m0_cl0
                # [n, group_opt, co1_opt // nc, howo // (mc*m0),
                # ||| nc // nc_factor, 1,
                # ||| nc_factor, mc_factor*m0, co0]
            out2al1_loopm_axis, al12al0_loopm_axis = sch[res].split(out2cl0_loopm_axis, nparts=al1_nparts[1])
            # when multi_cl0_group, cl0_factor[0] is 1 and bl1_nparts[1] is 1
            out2bl1_loopn_axis, bl12bl0_loopn_axis = sch[res].split(out2cl0_loopn_axis, nparts=bl1_nparts[1])

            # split batch of res
            if self._dynamic_shape.n_dynamic:
                batch_dim_factor = tvm.max(1, ceil_div(batch, batch_dim))
                res_batch_dim_axis, res_singlecore_batch_axis = sch[res].split(res_n_axis, batch_dim_factor)
            else:
                res_batch_dim_axis, res_singlecore_batch_axis = sch[res].split(res_n_axis, nparts=batch_dim)

            res_group_dim_axis, res_singlecore_group_opt_axis = sch[res].split(res_group_opt_axis, nparts=group_dim)

            batch_factor = self._inner_batch.config_innerbatch_axis(batch_al1, batch_cl0)
            out2al1_loopbatch_axis, res_al1_batch_axis = sch[res].split(res_singlecore_batch_axis, batch_factor)

            if group_cl0 == 1 and multi_bl0_group:
                singlecore_out2bl0_loopg_axis, res_bl0_group_opt_axis = sch[res].split(res_singlecore_group_opt_axis,
                                                                                    factor=group_bl0)
            else:
                singlecore_out2bl0_loopg_axis, res_bl0_group_opt_axis = sch[res].split(res_singlecore_group_opt_axis, 1)

            # split cout of res
            res_n_dim_axis, singlecore_out2bl1_loopn_axis = sch[res].split(out2bl1_loopn_axis, nparts=n_dim)
            res_m_dim_axis, singlecore_out2al1_loopm_axis = sch[res].split(out2al1_loopm_axis, nparts=m_dim)

            if self._inner_batch.flag:
                if fixpipe_nz2nd_flag:
                    pass
                elif fixpipe_channelsplit_flag:
                    sch[res].reorder(res_batch_dim_axis,
                                     res_group_dim_axis,
                                     res_n_dim_axis,
                                     res_m_dim_axis,
                                     singlecore_out2bl0_loopg_axis,
                                     res_bl0_group_opt_axis,
                                     out2al1_loopbatch_axis,
                                     singlecore_out2bl1_loopn_axis,
                                     singlecore_out2al1_loopm_axis,
                                     bl12bl0_loopn_axis,
                                     al12al0_loopm_axis,
                                     #===============cl0 tiling========================
                                     cl02cub_loopn_axis,
                                     cl02cub_loopm_axis,
                                     #===============cub tiling========================
                                     res_nc_factor_axis,
                                     res_al1_batch_axis,
                                     # third axis from bottom must be 2 for fp32
                                     special_axis_dict["fixpipe_channelsplit_res_c0_npart_axis"],
                                     res_m_factor_axis)
                else:
                    if fixpipe_antiquant_flag:
                        sch[res].reorder(res_batch_dim_axis,
                                         res_group_dim_axis,
                                         res_n_dim_axis,
                                         res_m_dim_axis,
                                         singlecore_out2bl0_loopg_axis,
                                         res_bl0_group_opt_axis,
                                         out2al1_loopbatch_axis,
                                         singlecore_out2bl1_loopn_axis,
                                         singlecore_out2al1_loopm_axis,
                                         bl12bl0_loopn_axis,
                                         al12al0_loopm_axis,
                                         #===============cl0 tiling========================
                                         cl02cub_loopn_axis,
                                         cl02cub_loopm_axis,
                                         #===============cub tiling========================
                                         res_nc_factor_axis,
                                         special_axis_dict["fixpipe_antiquant_res_c0_npart_axis"],
                                         res_al1_batch_axis,
                                         res_m_factor_axis)
                    else:
                        sch[res].reorder(res_batch_dim_axis,
                                         res_group_dim_axis,
                                         res_n_dim_axis,
                                         res_m_dim_axis,
                                         singlecore_out2bl0_loopg_axis,
                                         res_bl0_group_opt_axis,
                                         out2al1_loopbatch_axis,
                                         singlecore_out2bl1_loopn_axis,
                                         singlecore_out2al1_loopm_axis,
                                         bl12bl0_loopn_axis,
                                         al12al0_loopm_axis,
                                         #===============cl0 tiling========================
                                         cl02cub_loopn_axis,
                                         cl02cub_loopm_axis,
                                         #===============cub tiling========================
                                         res_nc_factor_axis,
                                         res_al1_batch_axis,
                                         res_m_factor_axis)
            else:
                sch[res].reorder(res_batch_dim_axis,
                                 res_group_dim_axis,
                                 res_n_dim_axis,
                                 res_m_dim_axis,
                                 singlecore_out2bl0_loopg_axis,
                                 res_bl0_group_opt_axis,
                                 out2al1_loopbatch_axis,
                                 singlecore_out2bl1_loopn_axis,
                                 singlecore_out2al1_loopm_axis,
                                 res_al1_batch_axis,
                                 bl12bl0_loopn_axis)

            if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
                res_pragma_axis = res_m_factor_axis
            else:
                res_pragma_axis = res_nc_factor_axis

            blocks = batch_dim*n_dim*m_dim*group_dim

            if blocks != 1:
                multicore_axis = sch[res].fuse(
                    res_batch_dim_axis,
                    res_group_dim_axis,
                    res_n_dim_axis,
                    res_m_dim_axis)
                if self._dynamic_shape.flag:
                    multicore_axis_o, _ = sch[res].split(multicore_axis, factor=1)
                else:
                    multicore_axis_o, _ = sch[res].split(multicore_axis, nparts=blocks)

                bindcore_axis, batchbindonly_pragma_axis = sch[res].split(multicore_axis_o, 1)
                sch[res].bind(bindcore_axis, tvm.thread_axis("blockIdx.x"))

                if blocks == batch_dim:
                    sch[res].pragma(batchbindonly_pragma_axis, 'json_info_batchBindOnly', 1)
            else:
                bindcore_axis = res_batch_dim_axis

            out2al1_loopbatch_axis_ori = out2al1_loopbatch_axis
            out2al1_loopbatch_axis, res_batch_1_axis = sch[res].split(out2al1_loopbatch_axis_ori, factor=1)

            reorder_mn_flag = set_reorder_mn_flag()

            if self._tiling["n_bef_batch_flag"] and not reorder_mn_flag and not self._inner_batch.flag:
                sch[res].reorder(singlecore_out2bl1_loopn_axis, out2al1_loopbatch_axis)

            def get_res_attach_axis():
                """
                prepare res attach axis for compute_at
                """
                # get cub_at_res_axis
                cub_at_res_axis = cl02cub_loopm_axis

                # get cl0_at_res_axis
                cl0_at_res_axis = al12al0_loopm_axis

                # get bl0_at_res_axis
                bl0_at_res_axis = singlecore_out2bl0_loopg_axis

                # get al1_at_res_axis
                al1_at_res_axis = out2al1_loopbatch_axis
                if self._lx_fusion.l1_fusion_type == DEPTH_L1_FUSION:
                    al1_at_res_axis = singlecore_out2bl0_loopg_axis
                if al1_tiling or self._conv1d.flag:
                    al1_at_res_axis = singlecore_out2al1_loopm_axis

                # get bl1_at_res_axis
                bl1_at_res_axis = singlecore_out2bl0_loopg_axis
                if bl1_tiling:
                    bl1_at_res_axis = singlecore_out2bl1_loopn_axis
                    if group_cl0 == 1 and multi_bl0_group:
                        bl1_at_res_axis = singlecore_out2bl0_loopg_axis
                if self._inner_batch.flag:
                    bl1_at_res_axis = singlecore_out2bl1_loopn_axis

                return [cub_at_res_axis, cl0_at_res_axis, bl0_at_res_axis,
                        al1_at_res_axis, bl1_at_res_axis]

            res_axis_list = get_res_attach_axis()

            attach_axis_dict.update(
                {
                "cub_at_res_axis": res_axis_list[0],
                "singlecore_out2al1_loopm_axis": singlecore_out2al1_loopm_axis,
                "al12al0_loopm_axis": al12al0_loopm_axis,
                "batchbindonly_pragma_axis": batchbindonly_pragma_axis if blocks != 1 else None,
                "res_m_dim_axis": res_m_dim_axis
                }
                )

            return res_axis_list, bindcore_axis, res_pragma_axis

        def bn1fusion_tile_tensor_res():
            """
            bn1fusion special tile_tensor_res
            """
            # res.op.axis [Co1, Co0]
            # res.op.reduce_axis [n, howo]
            res_n_axis, res_hw_axis = res.op.reduce_axis
            out2cl0_loopm_axis, res_m_cl0_axis = sch[res].split(res_hw_axis, factor=mc_cl0 * m0_cl0)
            res_batch_dim_axis, res_singlecore_batch_axis = sch[res].split(res_n_axis, nparts=batch_dim)

            out2al1_loopm_axis, al12al0_loopm_axis = sch[res].split(out2cl0_loopm_axis, nparts=al1_nparts[1])
            res_m_dim_axis, singlecore_out2al1_loopm_axis = sch[res].split(out2al1_loopm_axis, nparts=m_dim)

            sch[res].reorder(
                res_batch_dim_axis,
                res_m_dim_axis,
                res_singlecore_batch_axis,
                singlecore_out2al1_loopm_axis)
            res_batch_m_dim_fused = sch[res].fuse(res_batch_dim_axis, res_m_dim_axis)

            # res_batch_dim_axis and res_m_dim_axis are usually bindcore axis
            # in this scenario, these two are reduction axes
            # TBE don't support bindcore to a reduction axis
            # use rfactor to factor a reduction axis in tensor's schedule to be an explicit axis
            # which create a new stage that generated the new tensor with axis as the first dimension
            # The tensor's body will be rewritten as a reduction over the factored tensor
            res_ub_rf, _ = sch.rfactor(res, res_batch_m_dim_fused)
            sch[res_ub_rf].set_scope(cce_params.scope_ubuf)

            res_batch_m_dim_fused, res_c1_axis, res_c0_axis = res_ub_rf.op.axis
            res_singlecore_batch_axis, singlecore_out2al1_loopm_axis, al12al0_loopm_axis, res_m_cl0_axis = res_ub_rf.op.reduce_axis

            sum_x_global, square_sum_x_global = sch.cache_write([res, res], "global")

            sch[res].emit_insn(sch[res].op.axis[0], "phony_insn")

            bn_c1_axis, bn_c0_axis = sum_x_global.op.axis
            bn_batch_m_dim_fused = sum_x_global.op.reduce_axis[0]

            sch[sum_x_global].reorder(
                bn_batch_m_dim_fused,
                bn_c1_axis,
                bn_c0_axis)

            # use sch_list[0] to return conv schedule
            # use sch_list[1:] to indicate real_outs
            sch_list.append(self._multi_out[0])
            sch_list.append(sum_x_global)
            sch_list.append(square_sum_x_global)

            # split sum_x_global
            if self._convbn1.fp32_bn1_flag:
                bn_group_opt_axis, bn_co1_opt_axis_ori = sch[sum_x_global].split(bn_c1_axis, factor=co1_opt*2)
                bn_co1_opt_axis, bn_c0_npart_axis = sch[sum_x_global].split(bn_co1_opt_axis_ori, factor=2)
            else:
                bn_group_opt_axis, bn_co1_opt_axis = sch[sum_x_global].split(bn_c1_axis, factor=co1_opt)

            bn_group_dim_axis, bn_singlecore_group_opt_axis = sch[sum_x_global].split(bn_group_opt_axis,
                                                                                      nparts=group_dim)
            bn_out2cl0_loopn_axis, bn_nc_cl0_axis = sch[sum_x_global].split(bn_co1_opt_axis, factor=nc_cl0)

            bn_out2bl1_loopn_axis, bn_bl12bl0_loopn_axis = sch[sum_x_global].split(bn_out2cl0_loopn_axis,
                                                                                   nparts=bl1_nparts[1])
            bn_n_dim_axis, bn_singlecore_out2bl1_loopn_axis = sch[sum_x_global].split(bn_out2bl1_loopn_axis,
                                                                                      nparts=n_dim)
            if self._convbn1.fp32_bn1_flag:
                sch[sum_x_global].reorder(
                    bn_batch_m_dim_fused,
                    bn_group_dim_axis,
                    bn_n_dim_axis,
                    bn_singlecore_group_opt_axis,
                    bn_bl12bl0_loopn_axis,
                    bn_singlecore_out2bl1_loopn_axis,
                    bn_nc_cl0_axis,
                    bn_c0_npart_axis,  # 2
                    bn_c0_axis)
            else:
                sch[sum_x_global].reorder(
                    bn_batch_m_dim_fused,
                    bn_group_dim_axis,
                    bn_n_dim_axis,
                    bn_singlecore_group_opt_axis,
                    bn_bl12bl0_loopn_axis,
                    bn_singlecore_out2bl1_loopn_axis,
                    bn_nc_cl0_axis,
                    bn_c0_axis)

            multicore_axis = sch[sum_x_global].fuse(bn_batch_m_dim_fused, bn_group_dim_axis, bn_n_dim_axis)

            # split res_ub_rf
            if self._convbn1.fp32_bn1_flag:
                res_group_opt_axis, res_co1_opt_axis_ori = sch[res_ub_rf].split(res_c1_axis, factor=co1_opt*2)
                res_co1_opt_axis, res_c0_npart_axis = sch[res_ub_rf].split(res_co1_opt_axis_ori, factor=2)
            else:
                res_group_opt_axis, res_co1_opt_axis = sch[res_ub_rf].split(res_c1_axis, factor=co1_opt)

            out2cl0_loopn_axis, res_nc_cl0_axis = sch[res_ub_rf].split(res_co1_opt_axis, factor=nc_cl0)

            cl02cub_loopm_axis, res_m_factor_axis = sch[res_ub_rf].split(res_m_cl0_axis, nparts=1)

            # when multi_cl0_group, cl0_factor[0] is 1 and bl1_nparts[1] is 1
            out2bl1_loopn_axis, bl12bl0_loopn_axis = sch[res_ub_rf].split(out2cl0_loopn_axis, nparts=bl1_nparts[1])

            res_group_dim_axis, res_singlecore_group_opt_axis = sch[res_ub_rf].split(res_group_opt_axis,
                                                                                     nparts=group_dim)

            batch_factor = self._inner_batch.config_innerbatch_axis(batch_al1, batch_cl0)

            out2al1_loopbatch_axis, res_al1_batch_axis = sch[res_ub_rf].split(res_singlecore_batch_axis,
                                                                              factor=batch_factor)

            if group_cl0 == 1 and multi_bl0_group:
                singlecore_out2bl0_loopg_axis, res_bl0_group_opt_axis = sch[res_ub_rf].split(res_singlecore_group_opt_axis,
                                                                                             factor=group_bl0)
            else:
                singlecore_out2bl0_loopg_axis, res_bl0_group_opt_axis = sch[res_ub_rf].split(res_singlecore_group_opt_axis,
                                                                                             factor=1)

            res_n_dim_axis, singlecore_out2bl1_loopn_axis = sch[res_ub_rf].split(out2bl1_loopn_axis, nparts=n_dim)

            cl02cub_loopn_axis, res_nc_factor_axis = sch[res_ub_rf].split(res_nc_cl0_axis, factor=nc_factor_cub)

            if self._convbn1.fp32_bn1_flag:
                sch[res_ub_rf].reorder(
                    res_batch_m_dim_fused,
                    res_group_dim_axis,
                    res_n_dim_axis,
                    singlecore_out2bl0_loopg_axis,
                    out2al1_loopbatch_axis,
                    singlecore_out2bl1_loopn_axis,
                    singlecore_out2al1_loopm_axis,
                    res_bl0_group_opt_axis,
                    res_al1_batch_axis,
                    bl12bl0_loopn_axis,
                    al12al0_loopm_axis,
                    cl02cub_loopn_axis,
                    cl02cub_loopm_axis,
                    res_nc_factor_axis,
                    res_c0_npart_axis,  # 2
                    res_m_factor_axis,
                    res_c0_axis)
            else:
                sch[res_ub_rf].reorder(
                    res_batch_m_dim_fused,
                    res_group_dim_axis,
                    res_n_dim_axis,
                    singlecore_out2bl0_loopg_axis,
                    out2al1_loopbatch_axis,
                    singlecore_out2bl1_loopn_axis,
                    singlecore_out2al1_loopm_axis,
                    res_bl0_group_opt_axis,
                    res_al1_batch_axis,
                    bl12bl0_loopn_axis,
                    al12al0_loopm_axis,
                    cl02cub_loopn_axis,
                    cl02cub_loopm_axis,
                    res_nc_factor_axis,
                    res_m_factor_axis,
                    res_c0_axis)

            if self._inner_batch.flag:
                sch[res_ub_rf].reorder(
                    bl12bl0_loopn_axis,
                    al12al0_loopm_axis,
                    cl02cub_loopn_axis,
                    cl02cub_loopm_axis,
                    res_nc_factor_axis,
                    res_al1_batch_axis)

            blocks = batch_dim*n_dim*m_dim*group_dim
            sch[sum_x_global].bind(multicore_axis, tvm.thread_axis("blockIdx.x"))
            if blocks == batch_dim:
                sch[sum_x_global].pragma(bn_bl12bl0_loopn_axis, 'json_info_batchBindOnly', 1)

            out2al1_loopbatch_axis_ori = out2al1_loopbatch_axis
            out2al1_loopbatch_axis, res_batch_1_axis = sch[res_ub_rf].split(out2al1_loopbatch_axis_ori, factor=1)
            reorder_mn_flag = get_reorder_mn_flag()
            if not self._inner_batch.flag:
                if reorder_mn_flag:
                    sch[res_ub_rf].reorder(
                        out2al1_loopbatch_axis,
                        singlecore_out2al1_loopm_axis,
                        res_batch_1_axis,
                        bl12bl0_loopn_axis,
                        singlecore_out2bl1_loopn_axis,
                        res_bl0_group_opt_axis,
                        res_al1_batch_axis)
                else:
                    sch[res_ub_rf].reorder(
                        out2al1_loopbatch_axis,
                        singlecore_out2bl1_loopn_axis,
                        singlecore_out2al1_loopm_axis,
                        bl12bl0_loopn_axis,
                        res_batch_1_axis,
                        res_bl0_group_opt_axis,
                        res_al1_batch_axis)
            if self._tiling["n_bef_batch_flag"] and not reorder_mn_flag and not self._inner_batch.flag:
                sch[res_ub_rf].reorder(singlecore_out2bl1_loopn_axis, out2al1_loopbatch_axis)

            res_pragma_axis = res_m_factor_axis
            bindcore_axis = out2al1_loopbatch_axis

            def get_bn1_res_attach_axis():
                """
                conv2d+bn1 fusion prepare res attach axis for compute_at
                """
                # get cub_at_res_axis
                cub_at_res_axis = cl02cub_loopm_axis

                # get cl0_at_res_axis
                cl0_at_res_axis = al12al0_loopm_axis

                # get bl0_at_res_axis
                bl0_at_res_axis = singlecore_out2bl0_loopg_axis

                # get al1_at_res_axis
                al1_at_res_axis = out2al1_loopbatch_axis
                if self._lx_fusion.l1_fusion_type == DEPTH_L1_FUSION:
                    al1_at_res_axis = singlecore_out2bl0_loopg_axis
                if al1_tiling or self._conv1d.flag:
                    al1_at_res_axis = singlecore_out2al1_loopm_axis

                # get bl1_at_res_axis
                bl1_at_res_axis = singlecore_out2bl0_loopg_axis
                if bl1_tiling:
                    bl1_at_res_axis = singlecore_out2bl1_loopn_axis
                    if group_cl0 == 1 and multi_bl0_group:
                        bl1_at_res_axis = singlecore_out2bl0_loopg_axis
                if self._inner_batch.flag:
                    bl1_at_res_axis = singlecore_out2bl1_loopn_axis

                # get sum_x_global_pragma_axis
                if cub_channel_wise_flag:
                    sum_x_global_pragma_axis = bn_nc_cl0_axis
                else:
                    sum_x_global_pragma_axis = bn_bl12bl0_loopn_axis

                # get res_ub_rf_at_sum_x_axis
                res_ub_rf_at_sum_x_axis = multicore_axis
                if group_opt > 1:
                    res_ub_rf_at_sum_x_axis = bn_singlecore_group_opt_axis
                if cub_channel_wise_flag:
                    res_ub_rf_at_sum_x_axis = bn_singlecore_out2bl1_loopn_axis

                self._convbn1.set_sum_x_global(sum_x_global, sum_x_global_pragma_axis, res_ub_rf_at_sum_x_axis)

                return [cub_at_res_axis, cl0_at_res_axis, bl0_at_res_axis,
                        al1_at_res_axis, bl1_at_res_axis]

            res_axis_list = get_bn1_res_attach_axis()

            return res_ub_rf, res_axis_list, bindcore_axis, res_pragma_axis

        def tile_tensor_cl0():
            """
            tile tensor cl0
            """
            cl0_k1, cl0_k0 = cl0.op.reduce_axis

            # split ma*m0
            cl0_ma_factor = ma_al0
            cl0_ma_factor = self._pooling_fusion.modify_cl0_m_factor(cl0_ma_factor)
            cl0_mo, cl0_mi = sch[cl0].split(sch[cl0].op.axis[3], cl0_ma_factor * self._block_m0)

            if bl0_tiling == []:
                cl0_co, cl0_ci = sch[cl0].split(sch[cl0].op.axis[2], nparts=1)
            else:
                cl0_co, cl0_ci = sch[cl0].split(sch[cl0].op.axis[2], nb_bl0)

            if multi_cl0_group and multi_bl0_group:
                cl0_go, cl0_gi = sch[cl0].split(cl0.op.axis[0], factor=group_bl0)

            # for reduce axis, al0 and bl0 should be the same
            cl0_ko, cl0_ki = sch[cl0].split(cl0_k1, ka_al0)
            cl0_no, cl0_ni = sch[cl0].split(cl0.op.axis[1], 1)
            cl0_co0 = cl0.op.axis[4]

            if self._inner_batch.flag:
                sch[cl0].reorder(cl0_ko,
                                 cl0_co,
                                 cl0_no,
                                 cl0_mo,
                                 cl0_ni,
                                 cl0_ci,
                                 cl0_mi,
                                 cl0_co0,
                                 cl0_ki,
                                 cl0_k0)
            elif self._pooling_fusion.flag:
                cl0_cio, cl0_cii = sch[cl0].split(cl0_ci, 1)
                sch[cl0].reorder(cl0_ko,
                                 cl0_co,
                                 cl0_mo,
                                 cl0_cio,
                                 cl0_ni, # 1
                                 cl0_cii,
                                 cl0_mi, # ma*m0
                                 cl0_co0, # co0
                                 cl0_ki, # ka = kb
                                 cl0_k0) # k0
            else:
                sch[cl0].reorder(cl0_ko,
                                 cl0_co,
                                 cl0_mo,
                                 #=======L0C tiling=========
                                 cl0_ni, # 1
                                 cl0_ci, # nb
                                 cl0_mi, # ma*m0
                                 cl0_co0, # co0
                                 cl0_ki, # ka = kb
                                 cl0_k0) # k0

            if multi_cl0_group and multi_bl0_group:
                if self._inner_batch.flag:
                    sch[cl0].reorder(cl0_go,
                                     cl0_co,
                                     cl0_gi,
                                     cl0_ko,
                                     cl0_no,
                                     cl0_mo)
                else:
                    sch[cl0].reorder(cl0_go,
                                     cl0_no,
                                     cl0_co,
                                     cl0_gi,
                                     cl0_ko,
                                     cl0_mo)

            outer_factor = max(al1_nparts[0], bl1_nparts[0])
            inner_factor = min(al1_nparts[0], bl1_nparts[0])

            if outer_factor % inner_factor != 0:
                err_man.raise_err_specific("conv2d", "illegal value of AL1_shape & BL1_shape")

            if al1_nparts[0] > bl1_nparts[0]:
                cl0_koo, cl0_koi = sch[cl0].split(cl0_ko, nparts=al1_nparts[0])
                cl0_kooo, cl0_kooi = sch[cl0].split(cl0_koo, nparts=bl1_nparts[0])
            else:
                cl0_koo, cl0_koi = sch[cl0].split(cl0_ko, nparts=bl1_nparts[0])
                cl0_kooo, cl0_kooi = sch[cl0].split(cl0_koo, nparts=al1_nparts[0])

            def get_cl0_attach_axis():
                """
                prepare cl0 attach axis for compute_at
                """
                # get al0_at_cl0_axis
                al0_at_cl0_axis = cl0_mo

                # get bl0_at_cl0_axis
                bl0_at_cl0_axis = cl0_co
                if self._pooling_fusion.flag:
                    bl0_at_cl0_axis = cl0_cio

                # get al1_at_cl0_axis and bl1_at_cl0_axis
                if al1_nparts[0] > bl1_nparts[0]:
                    al1_at_cl0_axis = cl0_kooi
                    bl1_at_cl0_axis = cl0_kooo
                else:
                    al1_at_cl0_axis = cl0_kooo
                    bl1_at_cl0_axis = cl0_kooi

                if bl1_tiling:
                    if bl1_nparts[0] == 1 and multi_cl0_group:
                        bl1_at_cl0_axis = cl0_co

                if bl1_tiling == [] and multi_cl0_group:
                    bl1_at_cl0_axis = cl0_co

                return [al0_at_cl0_axis, bl0_at_cl0_axis,
                        al1_at_cl0_axis, bl1_at_cl0_axis]

            k_outer_list = [cl0_kooo, cl0_kooi, cl0_koi]
            cl0_axis_list = get_cl0_attach_axis()
            cl0_pragma_axis = cl0_ni

            attach_axis_dict.update(
                {
                "cl0_mo": cl0_mo
                }
                )

            return k_outer_list, cl0_axis_list, cl0_pragma_axis

        def tile_tensor_fixpipe_res():
            """
            tile tensor fixpipe_res
            """
            cub_pragma_axis = None

            if self._eltwise_ub_fusion.flag:
                cub = self._fixpipe_res
                if self._fixpipe_fusion.nz2nd_flag:
                    cub_n1_axis, cub_n0_axis = sch[cub].split(cub.op.axis[2], self._block_n0)
                    cub_m_axis = cub.op.axis[1]
                else:
                    cub_n1_axis = cub.op.axis[1]
                    cub_m_axis = cub.op.axis[2]

                cub_out2ub_loopn_axis, cub_nc_factor_axis = sch[cub].split(cub_n1_axis, nc_factor_cub)
                cub_out2ub_loopm_axis, cub_m_factor_axis = sch[cub].split(cub_m_axis, nparts=1)

                if self._fixpipe_fusion.nz2nd_flag:
                    sch[cub].reorder(
                        cub_out2ub_loopn_axis,
                        cub_out2ub_loopm_axis,
                        cub_m_factor_axis,
                        cub_nc_factor_axis,
                        cub_n0_axis
                        )
                    cub_pragma_axis = cub_m_factor_axis
                else:
                    sch[cub].reorder(
                        cub_out2ub_loopn_axis,
                        cub_out2ub_loopm_axis,
                        cub_nc_factor_axis,
                        cub_m_factor_axis
                        )
                    cub_pragma_axis = cub_nc_factor_axis
            return cub_pragma_axis

        def bl0_compute_at():
            """
            Handle bl0 attach.
            """
            if self._pooling_fusion.flag:
                sch[bl0].compute_at(sch[cl0], bl0_at_cl0_axis)
            elif bl0_tiling or (bl0_tiling == [] and multi_cl0_group):
                if multi_bl0_group and group_cl0 == 1:
                    sch[bl0].compute_at(sch[res], bl0_at_res_axis)
                else:
                    sch[bl0].compute_at(sch[cl0], bl0_at_cl0_axis)
            else:
                sch[bl0].compute_at(sch[res], bl0_at_res_axis)

        def al1_compute_at():
            """
            Handle al1 attach.
            """
            def get_al1_attach_info():
                """
                Get the consumer tensor of al1 and the target axis to be attached.
                """
                if multi_cl0_group:
                    return cl0, al1_at_cl0_axis

                if al1_tiling:
                    if al1_nparts[0] != 1:
                        return cl0, al1_at_cl0_axis
                return res, al1_at_res_axis

            consumer, target_axis = get_al1_attach_info()
            sch[al1].compute_at(sch[consumer], target_axis)
            if fmap_row_major is not None:
                sch[fmap_row_major].compute_at(sch[consumer], target_axis)

        def bl1_compute_at():
            """
            Handle bl1 attach.
            """
            def get_bl1_attach_info():
                """
                Get the consumer tensor of bl1 and the target axis to be attached.
                """
                if self._inner_batch.flag:
                    return res, bl1_at_res_axis

                if bl1_tiling:
                    if bl1_nparts[0] != 1:
                        # cl0_gi is under cl0_ko
                        return cl0, bl1_at_cl0_axis
                    if bl1_nparts[0] == 1 and multi_cl0_group:
                        return cl0, bl1_at_cl0_axis
                    return res, bl1_at_res_axis

                if bl1_tiling == [] and multi_cl0_group:
                    return cl0, bl1_at_cl0_axis

                return res, bl1_at_res_axis

            if bl1_tiling is not None:
                consumer, target_axis = get_bl1_attach_info()
                sch[bl1].compute_at(sch[consumer], target_axis)

        def bias_compute_at():
            """
            Handle bias attach.
            """
            if self._bias_flag:
                if cub_channel_wise_flag:  # use INPUT_L1_BT_param later
                    sch[bias_l1].compute_at(sch[res], cl0_at_res_axis)
                else:
                    sch[bias_l1].compute_at(sch[res], bindcore_axis)
                sch[bias_bt].compute_at(sch[res], cl0_at_res_axis)

        def anti_quant_spilt_flag(res):
            """
            fp16 and not nd2nz and anti_quant, split 2 for pass claculate c0 stride.
            """
            if res.dtype == "float16" and not self._fixpipe_fusion.nz2nd_flag and \
                    self._fixpipe_fusion.anti_quant_flag:
                return True

            return False

        #==========================parse tiling==================================
        al1_tiling = self._tiling["AL1_shape"]
        bl1_tiling = self._tiling["BL1_shape"]
        al0_tiling = self._tiling["AL0_matrix"]
        bl0_tiling = self._tiling["BL0_matrix"]
        cl0_tiling = self._tiling["CL0_matrix"]
        cub_tiling = self._tiling["CUB_matrix"]
        pingpong_buffer = self._tiling["manual_pingpong_buffer"]
        cub_channel_wise_flag = self._tiling["CUB_channel_wise_flag"]
        batch = self._batch
        kernel_h = self._kernel_h
        kernel_w = self._kernel_w
        dilate_h = self._dilate_h
        dilate_w = self._dilate_w
        block_k0 = self._block_k0
        group_opt = self._group_opt
        ci1_opt = self._ci1_opt
        co1_opt = self._co1_opt
        out_hw = self._out_hw

        # only bl1_tiling can be None, only al0_tiling cannot be []

        if al0_tiling:
            ma_al0, ka_al0, _, _, _, _ = al0_tiling

        if bl0_tiling:
            _, nb_bl0, _, _, _, group_bl0 = bl0_tiling

        if cl0_tiling:
            nc_cl0, mc_cl0, m0_cl0, _, batch_cl0, group_cl0 = cl0_tiling

        if cub_tiling:
            nc_factor_cub, _, _, _, _, _ = cub_tiling

        batch_al1 = 1

        if al1_tiling:
            k_al1, multi_m_al1, batch_al1, _ = al1_tiling
            k1_al1 = k_al1 // (((kernel_h - 1)*dilate_h + 1)*((kernel_w - 1)*dilate_w + 1)*block_k0)
            if k1_al1 == 0:
                k1_al1 = 1

        if bl1_tiling:
            k_bl1, multi_n_bl1, _, _ = bl1_tiling
            k1_bl1 = k_bl1 // (kernel_h*kernel_w*block_k0)

        batch_dim, n_dim, m_dim, group_dim = self._tiling["block_dim"]

        #==========================calculate various coefficients==================================
        multi_bl0_group = bl0_tiling and group_bl0 > 1
        multi_cl0_group = group_cl0 > 1

        if multi_bl0_group or multi_cl0_group:
            cl0_factor = [1, ceil_div(out_hw, mc_cl0*m0_cl0)]
        else:
            cl0_factor = [ceil_div(co1_opt, nc_cl0), ceil_div(out_hw, mc_cl0*m0_cl0)]

        cl0_factor = self._pooling_fusion.modify_cl0_factor(cl0_factor)

        if al1_tiling:
            al1_nparts = [ci1_opt // k1_al1, ceil_div(cl0_factor[1], multi_m_al1)]
        else: # al1 full load
            al1_nparts = [1, 1]

        if bl1_tiling:
            if cl0_factor[0] % multi_n_bl1 != 0:
                err_man.raise_err_specific("conv2d", "second value of BL1_shape should be factor of n block num")
            if multi_n_bl1 > 1 and multi_n_bl1 % 2 != 0:
                err_man.raise_err_specific("conv2d", "second value of BL1_shape better to be even number")

            bl1_nparts = [(ci1_opt + k1_bl1 - 1) // k1_bl1,
                          (cl0_factor[0] + multi_n_bl1 - 1) // multi_n_bl1]
        else:
            bl1_nparts = [1, n_dim]

        #===========================split and compute at=========================================
        sch = self._sch
        al1 = tensor_param["al1"]
        bl1 = tensor_param["bl1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]
        fmap_row_major = tensor_param["fmap_row_major"]
        bias_l1 = tensor_param["bias_l1"]
        bias_bt = tensor_param["bias_bt"]

        attach_axis_dict = {}
        tiling_param = {}

        # tile
        #===================================tile al0============================================
        al0_axis_list, dynamic_al0_pragma_axis = tile_tensor_al0()

        #===================================tile res============================================
        if self._convbn1.flag:
            res, res_axis_list, bindcore_axis, res_pragma_axis = bn1fusion_tile_tensor_res()
        else:
            res_axis_list, bindcore_axis, res_pragma_axis = tile_tensor_res()
        cub_at_res_axis, cl0_at_res_axis, bl0_at_res_axis, al1_at_res_axis, bl1_at_res_axis = res_axis_list

        #=====================================tile cl0==============================================
        k_outer_list, cl0_axis_list, cl0_pragma_axis = tile_tensor_cl0()
        al0_at_cl0_axis, bl0_at_cl0_axis, al1_at_cl0_axis, bl1_at_cl0_axis = cl0_axis_list

        #=======================tile fixpipe_res==============================
        cub_pragma_axis = tile_tensor_fixpipe_res()

        #=================tile reform_by_vmuls/reform_by_vadds in quant op==================

        self._quant_fusion.split_reform_axis(sch)

        #===============================attach=======================================
        # fixpipe
        fixpipe_slice_axis = cub_at_res_axis if group_opt*co1_opt > 16 else bindcore_axis
        self._fixpipe_fusion.fixpipe_inputs_compute_at(sch, res, fixpipe_slice_axis, cub_at_res_axis)

        # ub fusion
        self._eltwise_ub_fusion.ub_tensors_attach(sch, res, cub_at_res_axis)

        # quant fusion
        self._quant_fusion.quant_tensors_attach(sch, res, cub_at_res_axis)

        # bn1 fusion
        self._convbn1.bn1fusion_compute_at(sch, res, cub_at_res_axis)

        sch[cl0].compute_at(sch[res], cl0_at_res_axis)
        sch[al0].compute_at(sch[cl0], al0_at_cl0_axis)
        bl0_compute_at()
        al1_compute_at()
        bl1_compute_at()
        bias_compute_at()

        tiling_param.update(
            {
            "al1_tiling": al1_tiling,
            "al0_tiling": al0_tiling,
            "bl1_tiling": bl1_tiling,
            "bl0_tiling": bl0_tiling,
            "cl0_tiling": cl0_tiling,
            "al1_nparts": al1_nparts,
            "stride_h_update": self._strideh_opti.stride_h_update,
            "fmap_5hd_shape": self._para_dict["a_shape"],
            "blocks": batch_dim*n_dim*m_dim*group_dim,
            "m_cl0": mc_cl0*m0_cl0,
            "out_hw": self._out_hw
            }
            )
        if al1_tiling:
            tiling_param.update({"multi_m_al1": multi_m_al1,
                                 "k_al1": k_al1,
                                 "k1_al1": k1_al1})

        emit_insn_dict = {"al0_axis_list": al0_axis_list,
                          "bindcore_axis": bindcore_axis,
                          "k_outer": k_outer_list,
                          "cl0_pragma_axis": cl0_pragma_axis,
                          "res_pragma_axis": res_pragma_axis,
                          "cub_pragma_axis": cub_pragma_axis,
                          #================dynamic shape======================
                          "dynamic_al0_pragma_axis": dynamic_al0_pragma_axis,
                         }

        return res, tiling_param, emit_insn_dict, attach_axis_dict

    def special_process_post(self, res, conv_param, tensor_param, tiling_param, emit_insn_dict, attach_axis_dict):
        """
        Special process before tiling is parsed.
        """
        #===========================prepare params================================================
        fmap = tensor_param["fmap"]
        fmap_row_major = tensor_param["fmap_row_major"]
        al1 = tensor_param["al1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]
        cub = self._eltwise_ub_fusion.cub

        pingpong_buffer = self._tiling["manual_pingpong_buffer"]
        _, _, m_dim, _ = self._tiling["block_dim"]

        cl0_tiling = tiling_param["cl0_tiling"]
        blocks = tiling_param["blocks"]

        res_pragma_axis = emit_insn_dict["res_pragma_axis"]
        bindcore_axis = emit_insn_dict["bindcore_axis"]

        cub_at_res_axis = attach_axis_dict.get("cub_at_res_axis")
        singlecore_out2al1_loopm_axis = attach_axis_dict.get("singlecore_out2al1_loopm_axis")
        al12al0_loopm_axis = attach_axis_dict.get("al12al0_loopm_axis")
        batchbindonly_pragma_axis = attach_axis_dict.get("batchbindonly_pragma_axis")
        res_m_dim_axis = attach_axis_dict.get("res_m_dim_axis")
        cl0_mo = attach_axis_dict.get("cl0_mo")

        sch = self._sch

        # parse the tbe compile parameter
        sch.tbe_compile_para, preload_flag = util.parse_tbe_compile_para(self._tiling.get("compile_para"))
        if pingpong_buffer["CL0_pbuffer"] == 2 and preload_flag:
            sch[cl0].preload()

        #=================================CL0 buffer align======================================
        # CL0 shape: [group_opt, batch, co1_opt, howo, co0]
        sch[cl0].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (self._block_m0, self._block_m0),
            (self._block_n0, self._block_n0),
            (1, 1), # reduce_axis_k1
            (self._block_k0, self._block_k0) # reduce_axis_k0
            )

        # align cl0 memory allocation when channel merging
        if res.dtype in ("int4", "int8"):
            sch[cl0].set_buffer_size(reduce((lambda x, y: x*y), cl0_tiling))

        #=================================lxfusion======================================
        self._lx_fusion.config_l1_tensormap(sch, fmap, al1, self._op_graph)

        #=================================dynamic shape process=====================================
        self._dynamic_shape.set_al1_bound(sch, al1, conv_param, tiling_param,
                                          self._l0a_load2d.flag, self._strideh_opti.flag)
        self._dynamic_shape.set_cl0_bound(sch, cl0, cl0_tiling)
        self._dynamic_shape.res_hw_dynamic_pragma(sch, res, res_pragma_axis)

        #=================================maxpool fusion=====================================
        self._pooling_fusion.set_build_cfg()

        self._pooling_fusion.process_maxpool_bl0(
            sch, bl0, res, blocks,
            batchbindonly_pragma_axis,
            res_m_dim_axis,
            al12al0_loopm_axis,
            singlecore_out2al1_loopm_axis,
            cl0_mo
            )

        self._pooling_fusion.process_maxpool_ub_tensors(
            sch, res,
            cub_at_res_axis, bindcore_axis, singlecore_out2al1_loopm_axis, al12al0_loopm_axis,
            self._out_width, m_dim, pingpong_buffer
            )

        self._pooling_fusion.maxpool_buffertile(
            sch, fmap_row_major, al1, al0, cl0, cub, res,
            m_dim, self._stride_h, self._kernel_h, self._out_width, self._conv_param,
            singlecore_out2al1_loopm_axis, al12al0_loopm_axis, cl0_mo
            )

    def double_buffer(self, tensor_param):
        """
        Enable pingpong buffer.
        """
        pingpong_buffer = self._tiling["manual_pingpong_buffer"]

        del pingpong_buffer["AUB_pbuffer"]
        if "BUB_pbuffer" in pingpong_buffer:
            del pingpong_buffer["BUB_pbuffer"]

        del pingpong_buffer["UBG_pbuffer"]

        al1 = tensor_param["al1"]
        bl1 = tensor_param["bl1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]

        pingpong_map = {"AL1_pbuffer": al1,
                        "BL1_pbuffer": bl1,
                        "AL0_pbuffer": al0,
                        "BL0_pbuffer": bl0,
                        "CL0_pbuffer": cl0}

        # need to do bt doublebuffer with INPUT_L1_BT_pbuffer

        if pingpong_buffer["CUB_pbuffer"] == 2:
            for tensor in list(self._eltwise_ub_fusion.ub_body_tensors) + self._eltwise_ub_fusion.cache_write_tensors:
                self._sch[tensor].double_buffer()
            if self._eltwise_ub_fusion.flag:
                self._sch[self._fixpipe_res].double_buffer()

        del pingpong_buffer["CUB_pbuffer"]

        for key, value in pingpong_buffer.items():
            if value == 2 and pingpong_map[key] is not None:
                self._sch[pingpong_map[key]].double_buffer()

        self._pooling_fusion.maxpool_al1_preload(self._sch, pingpong_buffer, al1)

    def map_insn(self, res, tensor_param, tiling_param, emit_insn_dict):
        """
        Emit insn for each tensor.
        """
        def im2col_emit_insn():
            """
            Emit insn for AL1/Row_major/AL0.
            """
            def config_setfmatrix():
                """
                Emit insn for row_major tensor.
                """
                if self._im2col_dma.flag:
                    return None

                setfmatrix_dict = {
                    "conv_kernel_h": conv_param.filter_h,
                    "conv_kernel_w": conv_param.filter_w,
                    "conv_padding_top": conv_param.padding[0],
                    "conv_padding_bottom": conv_param.padding[1],
                    "conv_padding_left": conv_param.padding[2],
                    "conv_padding_right": conv_param.padding[3],
                    "conv_stride_h": stride_h_update,
                    "conv_stride_w": conv_param.stride_w,
                    "conv_dilation_h": conv_param.dilate_h,
                    "conv_dilation_w": conv_param.dilate_w}

                setfmatrix_dict["conv_fm_c"] = al1.shape[1]*al1.shape[4]
                setfmatrix_dict["conv_fm_h"] = al1.shape[2]
                setfmatrix_dict["conv_fm_w"] = al1.shape[3]

                return setfmatrix_dict

            def al1_emit_insn():
                """
                Emit insn for al1.
                """
                def al1_common_emit_insn(sch, al1):
                    """
                    Emit insn for al1 in common usage.
                    """
                    sch[al1].emit_insn(al1.op.axis[0],
                                       "dma_copy",
                                       {"mem_align": 1})

                if self._im2col_dma.flag:
                    return None
                if self._input_nd2nz.flag:
                    return self._input_nd2nz.al1_nd2nz_emit_insn(sch, al1)
                if self._aipp_fusion.flag:
                    return self._aipp_fusion.al1_aipp_emit_insn(sch, al1)

                return al1_common_emit_insn(sch, al1)

            def al0_emit_insn():
                """
                Emit insn for al0.
                """
                def al0_common_emit_insn(sch, fmap_row_major, al0, setfmatrix_dict, al0_axis_list):
                    """
                    Emit insn for al0 and row major tensor in common usage.
                    """
                    sch[fmap_row_major].emit_insn(fmap_row_major.op.axis[1], 'set_fmatrix', setfmatrix_dict)
                    sch[al0].emit_insn(al0_axis_list[3], 'im2col')

                if self._im2col_dma.flag:
                    return self._im2col_dma.im2col_dma_emit_insn(sch, al1_im2col, al0, al0_axis_list)

                return al0_common_emit_insn(sch, fmap_row_major, al0, setfmatrix_dict, al0_axis_list)

            if self._dynamic_shape.flag:
                self._dynamic_shape.dynamic_mode_im2col_v2(
                    sch, conv_param, tensor_param, tiling_param,
                    emit_insn_dict, self._input_nd2nz.flag, self._l0a_load2d.flag)
            elif self._l0a_load2d.flag:
                self._l0a_load2d.load2d_emit_insn(sch, al1, al0)
            else:
                setfmatrix_dict = config_setfmatrix()
                al1_emit_insn()
                self._lx_fusion.al1_l1fusion_pragma(sch, al1)
                al0_emit_insn()
            if not self._dynamic_shape.flag:
                get_weight_repeat_number()

        def bl1_emit_insn():
            """
            Emit insn for bl1.
            """
            def bl1_common_emit_insn():
                """
                Emit insn for bl1 in common usage.
                """
                if bl1_tiling is not None:
                    sch[bl1].emit_insn(bl1.op.axis[0], "dma_copy")

            if self._weight_nd2nz.flag:
                return self._weight_nd2nz.bl1_nd2nz_emit_insn(sch, bl1)

            return bl1_common_emit_insn()

        def cl0_emit_insn():
            """
            Emit insn for cl0.
            """
            mad_dict = {"mad_pattern": 2, "k_outer": k_outer}

            if self._fmap_dtype == "float32" and conv_param.impl_mode == "high_performance":
                if tbe.common.platform.platform_info.intrinsic_check_support("Intrinsic_mmad", "h322f32"):
                    mad_dict["hf32"] = 1
                    log.debug("enable HF32 mode")

            sch[cl0].emit_insn(cl0_pragma_axis, 'mad', mad_dict)

        def get_weight_repeat_number():
            """
            Get load repeat number of weight tensor.
            """
            if bl1_tiling is None:
                if bl0_tiling == []:
                    weight_repeat_load_num = 1
                else:
                    weight_repeat_load_num = al1_nparts[1]
            elif bl1_tiling == []:
                weight_repeat_load_num = 1
            else:
                weight_repeat_load_num = al1_nparts[1]
            sch[res].pragma(bindcore_axis, "json_info_weight_repeat", weight_repeat_load_num)

        def fixpipe_res_emit_insn():
            """
            Emit insn for fixpipe_res tensor.
            """
            layout_transform_dict = {
                "int4": "channel_merge",
                "int8": "channel_merge",
                "float32": "channel_split"
            }

            def get_res_insn_str():
                """
                Get res emit insn pragma.
                """
                if is_support_fixpipe_op():
                    return "fixpipe_op"
                return "dma_copy"

            def res_merge_split_emit_insn():
                """
                Emit insn for res tensor in channel merge/split situation.
                """
                sch[self._fixpipe_res].emit_insn(res_pragma_axis, "dma_copy",
                                                 attrs={"layout_transform": layout_transform_dict[res.dtype]})

            def res_common_emit_insn():
                """
                Emit insn for res tensor in common usage.
                """
                fixpipe_res_pragma_axis = cub_pragma_axis if self._eltwise_ub_fusion.flag else res_pragma_axis
                sch[self._fixpipe_res].emit_insn(fixpipe_res_pragma_axis, get_res_insn_str())

            if self._convbn1.flag:
                return self._convbn1.bn1fusion_cub_emit_insn(sch)

            if is_support_fixpipe_op():
                return res_common_emit_insn()

            if self._output_nz2nd.flag:
                return self._output_nz2nd.res_nz2nd_emit_insn(sch, self._fixpipe_res, res_pragma_axis)
            if res.dtype in ("int4", "int8", "float32"):
                return res_merge_split_emit_insn()

            return res_common_emit_insn()

        def res_emit_insn():
            """
            Emit insn for res tensor.
            """
            if self._convbn1.flag:
                self._convbn1.bn1fusion_res_emit_insn(sch, res, res_pragma_axis, self._dynamic_shape.flag)
            elif self._eltwise_ub_fusion.flag:
                sch[res].emit_insn(res_pragma_axis, "dma_copy")

        def bias_emit_insn():
            """
            Emit insn for bias.
            """
            if self._bias_flag:
                sch[bias_l1].emit_insn(bias_l1.op.axis[0], "dma_copy", attrs={"layout_transform": "nd2nz"})
                sch[bias_bt].emit_insn(bias_bt.op.axis[0], "dma_copy")

        #=============================prepare params=========================================
        conv_param = self._conv_param
        sch = self._sch
        al1 = tensor_param["al1"]
        bl1 = tensor_param["bl1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]
        al1_im2col = tensor_param["al1_im2col"]
        fmap_row_major = tensor_param["fmap_row_major"]
        bias_l1 = tensor_param["bias_l1"]
        bias_bt = tensor_param["bias_bt"]

        bl1_tiling = tiling_param["bl1_tiling"]
        bl0_tiling = tiling_param["bl0_tiling"]
        al1_nparts = tiling_param["al1_nparts"]
        stride_h_update = tiling_param["stride_h_update"]

        al0_axis_list = emit_insn_dict["al0_axis_list"]
        bindcore_axis = emit_insn_dict["bindcore_axis"]
        k_outer = emit_insn_dict["k_outer"]
        cl0_pragma_axis = emit_insn_dict["cl0_pragma_axis"]
        res_pragma_axis = emit_insn_dict["res_pragma_axis"]
        cub_pragma_axis = emit_insn_dict["cub_pragma_axis"]

        #=============================emit insn=========================================
        im2col_emit_insn()

        bl1_emit_insn()

        sch[bl0].emit_insn(bl0.op.axis[0], "dma_copy")

        cl0_emit_insn()

        fixpipe_res_emit_insn()

        res_emit_insn()

        bias_emit_insn()

        self._fixpipe_fusion.fixpipe_inputs_emit_insn(sch)

        self._eltwise_ub_fusion.ub_tensors_emit_insn(sch, res)

        self._quant_fusion.quant_tensors_emit_insn(sch)


def conv_v220_schedule(sch, res, spec_node_list, sch_list, conv_param, op_graph, tiling_dict_flag, tiling_case=None, var_range=None):
    """
    Schedule for Conv2d v220.
    """
    schedule = Conv2dSchedule(sch, res, spec_node_list, conv_param, op_graph, tiling_dict_flag, tiling_case, var_range)

    info_dict = schedule.fetch_info_dict(tiling_case)

    if conv_param.dynamic_flag and tiling_dict_flag:
        return info_dict

    schedule.fetch_tiling(info_dict, tiling_case)

    schedule.verify_tiling()

    tensor_param = schedule.config_scope()

    schedule.special_process_pre(res, tensor_param)

    res, tiling_param, emit_insn_dict, attach_axis_dict = schedule.tile_attach_tensor(res, sch_list, tensor_param)

    schedule.special_process_post(res, conv_param, tensor_param, tiling_param, emit_insn_dict, attach_axis_dict)

    schedule.double_buffer(tensor_param)

    schedule.map_insn(res, tensor_param, tiling_param, emit_insn_dict)

    return None
