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
conv2d backprop filter schudule.
"""

from __future__ import absolute_import
from __future__ import print_function
from te import tvm
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import scope_ca
from te.platform.cce_params import scope_cb
from te.platform.cce_params import scope_cc
from te.platform.cce_params import scope_cbuf
from te.platform.cce_conf import get_soc_spec
from tbe.common.tiling.get_tiling import get_tiling
from te.lang.cce.te_compute.cube_util import shape_to_list
from tbe.dsl.static_schedule.util import parse_tbe_compile_para

from tbe.common.utils.errormgr import error_manager_util
from te.lang.cce.te_compute.conv2d_backprop_filter_compute \
    import DynamicConv2dBpFilterParams
# for debug, delete before publish
DEBUG_MODE = False
# disable double buffer, set True
DEBUG_DOUBLE_BUFFER_OFF = False

CUBE_DIM = 16
FLOAT16_SIZE = 2
FLOAT32_SIZE = 4
CUBE_MUL_SHAPE = 256
OPEN_DOUBLE_BUFFER = 2
DEFAULT_TILING_CASE = 32
LOOSE_LINE_CONDITION = 2
KB_2_B = 1024


def _ceil_div(dividend, divisor):
    """
    do division and round up to an integer

    """
    if divisor == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        error_manager_util.raise_runtime_error(dict_args)
    return (dividend + divisor - 1) // divisor


def _align(x_1, x_2):
    """
    do align

    """
    if x_2 == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        error_manager_util.raise_runtime_error(dict_args)
    return (x_1 + x_2 - 1) // x_2*x_2


class CceConv2dBackpropFilterOp:  # pylint: disable=too-few-public-methods
    """
    CceConv2dBackpropFilterOp: schedule definition of conv2d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    schedule : schedule definition of conv2d_backprop_filter

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
        self.cube_vector_split = get_soc_spec("CUBE_VECTOR_SPLIT")
        self.l1_size = get_soc_spec("L1_SIZE")  # L1 size
        self._corenum = get_soc_spec("CORE_NUM")
        self._loc_size = get_soc_spec("L0C_SIZE")
        self._lob_size = get_soc_spec("L0B_SIZE")

    def schedule(self,  # pylint: disable=R0914,R0915
                 res, spec_node_list, sch_list, dynamic_para=None):
        """
        schedule definition of conv2d_backprop_filter

        Parameters:
        ----------
        res :

        spec_node_list :

        sch_list:

        dynamic_para : A dict of dynamic shape parameters

        Returns
        -------
        None
        """

        self.spec_node_list = spec_node_list

        def _get_dyn_mode(dynamic_para):
            if not dynamic_para:
                return None
            var_range = dynamic_para.get("var_range")
            if not var_range:
                return None
            if "batch" in var_range and len(var_range) == 1:
                return "dynamic_batch"
            for var in ("dedy_h", "dedy_w", "fmap_h", "fmap_w"):
                if var in var_range:
                    return "dynamic_hw"
            return None

        def _tiling_shape_check(): # pylint: disable=R0915
            """
            do tiling shape paramters general check

            """

            al1_shape = tiling.get("AL1_shape")
            bl1_shape = tiling.get("BL1_shape")
            al0_matrix = tiling.get("AL0_matrix")
            bl0_matrix = tiling.get("BL0_matrix")
            cl0_matrix = tiling.get("CL0_matrix")
            if al1_shape:
                if (al1_shape[0]//CUBE_DIM) % al0_matrix[1] != 0:
                    dict_args = dict()
                    dict_args['errCode'] = "E64006"
                    dict_args['param_1'] = "AL1_shape(AL1_shape[0]//16)"
                    dict_args['param_2'] = "AL0_matrix(ALO_matrix[1])"
                    dict_args['value_1'] = str(al1_shape[0]//16)
                    dict_args['value_2'] = str(al0_matrix[1])
                    error_manager_util.raise_runtime_error(dict_args)
                if al1_shape[1] < 1:
                    dict_args = dict()
                    dict_args['errCode'] = "E64007"
                    dict_args['axis_name'] = "m"
                    dict_args['param_name'] = "AL1_shape[1]"
                    dict_args['param_value'] = str(al1_shape[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if bl1_shape:
                if (bl1_shape[0]//CUBE_DIM) % bl0_matrix[0] != 0:
                    dict_args = dict()
                    dict_args['errCode'] = "E64006"
                    dict_args['param_1'] = "BL1_shape(BL1_shape[0]//16)"
                    dict_args['param_2'] = "BL0_matrix(BL0_matrix[0])"
                    dict_args['value_1'] = str(bl1_shape[0]//16)
                    dict_args['value_2'] = str(bl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)
                if bl1_shape[1] < 1:
                    dict_args = dict()
                    dict_args['errCode'] = "E64007"
                    dict_args['axis_name'] = "n"
                    dict_args['param_name'] = "BL1_shape[1]"
                    dict_args['param_value'] = str(bl1_shape[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if al0_matrix:
                if al0_matrix[0] != cl0_matrix[1]:
                    dict_args = dict()
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'm_axis'
                    dict_args['param_1'] = "AL0_matrix"
                    dict_args['param_2'] = "CL0_matrix"
                    dict_args['value_1'] = str(al0_matrix[0])
                    dict_args['value_2'] = str(cl0_matrix[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if bl0_matrix:
                if bl0_matrix[1] != cl0_matrix[0]:
                    dict_args = dict()
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'n_axis'
                    dict_args['param_1'] = "BL0_matrix"
                    dict_args['param_2'] = "CL0_matrix"
                    dict_args['value_1'] = str(bl0_matrix[1])
                    dict_args['value_2'] = str(cl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)

            if al0_matrix and bl0_matrix:
                if al0_matrix[1] != bl0_matrix[0]:
                    dict_args = dict()
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'k_axis'
                    dict_args['param_1'] = "AL0_matrix"
                    dict_args['param_2'] = "BL0_matrix"
                    dict_args['value_1'] = str(al0_matrix[1])
                    dict_args['value_2'] = str(bl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)

        def _tiling_buffer_check():
            """
            Do buffer paramters general check

            """
            block_cout = tiling.get("block_dim")

            al1_pbuff = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
            bl1_pbuff = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
            al0_pbuff = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
            bl0_pbuff = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
            l0c_pbuff = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
            cub_pbuff = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
            cl0_matrix = tiling.get("CL0_matrix")
            cub_matrix = tiling.get("CUB_matrix")
            if (not self.cube_vector_split
                and (cl0_matrix[0] % cub_matrix[0] != 0 or cl0_matrix[1] != cub_matrix[1])):
                dict_args = dict()
                dict_args['errCode'] = "E64009"
                error_manager_util.raise_runtime_error(dict_args)
            # blockIdx must be positive int
            dim_i = 0
            for dim_x in block_cout:
                if dim_x < 1:
                    dict_args = dict()
                    dict_args["errCode"] = "E64004"
                    dict_args["param_name"] = "tiling.block_dim"
                    dict_args["axis_rule"] = "positive int"
                    dict_args["wrong_axis"] = str(dim_i)
                    dict_args["actual_value"] = str(dim_x)
                    error_manager_util.raise_runtime_error(dict_args)
                dim_i = dim_i + 1

            def _gen_dict_args(name, value):
                dict_args = dict()
                dict_args["errCode"] = "E64010"
                dict_args["buffer_name"] = name
                dict_args["value"] = str(value)
                return dict_args
            # only support no dbuffer/ dbuffer
            if al1_pbuff not in (1, 2):
                dict_args = _gen_dict_args("AL1_pbuffer", al1_pbuff)
                error_manager_util.raise_runtime_error(dict_args)
                error_manager_util.raise_runtime_error(dict_args)

            if bl1_pbuff not in (1, 2):
                dict_args = _gen_dict_args("BL1_pbuffer", bl1_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if al0_pbuff not in (1, 2):
                dict_args = _gen_dict_args("AL0_pbuffer", al0_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if bl0_pbuff not in (1, 2):
                dict_args = _gen_dict_args("BL0_pbuffer", bl0_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if l0c_pbuff not in (1, 2):
                dict_args = _gen_dict_args("L0C_pbuffer", l0c_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if cub_pbuff not in (1, 2):
                dict_args = _gen_dict_args("CUB_pbuffer", cub_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

        def _l1_limit_check():
            """
            do L1 size limit check

            """
            al1_min_byte = CUBE_DIM * CUBE_DIM * FLOAT16_SIZE
            if not flag_conv1d_case:
                kl1_min = width_fmap
            else:
                kl1_min = (CUBE_DIM - 1) * stride_width + kw_dilation
            if width_grads >= CUBE_DIM:
                if width_grads % CUBE_DIM == 0:
                    bl1_min_byte = kernel_height * kl1_min * CUBE_DIM *\
                        FLOAT16_SIZE
                else:
                    bl1_min_byte = (kernel_height+stride_height) \
                        * kl1_min * CUBE_DIM * FLOAT16_SIZE
            else:
                bl1_align_factor = _ceil_div(CUBE_DIM, width_grads)
                if CUBE_DIM % width_grads == 0:
                    bl1_min_byte = (kernel_height+(bl1_align_factor-1)
                                    * stride_height) * kl1_min *\
                        CUBE_DIM * FLOAT16_SIZE
                else:
                    bl1_min_byte = (kernel_height +
                                    bl1_align_factor * stride_height) \
                        * kl1_min * CUBE_DIM * FLOAT16_SIZE
            if (al1_min_byte + bl1_min_byte) > self.l1_size:
                dict_args = dict()
                dict_args["errCode"] = "E60026"
                error_manager_util.raise_runtime_error(dict_args)

        def _atomic_add(sch, res_cc, res_ub, res_ddr):
            """
            achieve atomic add according to refactor dw_cc

            """

            # redefine dw_ddr, dw_ub, dw_cc to achieve atomic write
            ub_reduce = res_ub
            ddr_reduce = res_ddr
            batch, real_k = sch[res_cc].op.reduce_axis
            batch_dim_factor = _ceil_div(batch_fmap, block_dim_batch)
            batch_dim_factor = tvm.max(1, batch_dim_factor)
            if self.dynamic_mode == "dynamic_batch":
                batch_core, batch_in = sch[res_cc].split(
                    batch, factor=batch_dim_factor)
            else:
                batch_core, batch_in = sch[res_cc].split(
                        batch, nparts=block_dim_batch)

            if self.dynamic_mode and \
                not DynamicConv2dBpFilterParams.flag_all_one_case:
                # for dynamic hw, the reduce axis of res_cc dose not cut k0
                hw_single_core_factor = _ceil_div(hw_pad_1 * CUBE_DIM, \
                                                    block_dim_hw)
                hw_single_core_factor = _align(hw_single_core_factor, \
                                                    dw_k * CUBE_DIM)
                k_1_multicore, real_k = sch[res_cc].split(real_k, \
                                                    hw_single_core_factor)

                sch[res_cc].reorder(k_1_multicore, batch_core, batch_in,
                                    real_k)
            else:
                real_k, k_in = sch[res_cc].split(real_k, CUBE_DIM)
                k_1_multicore, real_k = sch[res_cc].split(real_k,
                                                          nparts=block_dim_hw)

                sch[res_cc].reorder(k_1_multicore, batch_core, batch_in,
                                    real_k, k_in)
            fused_atomic_write = sch[res_cc].fuse(k_1_multicore, batch_core)

            # after rfactor op, dw_cc becomes dw_ddr, original dw_ub and dw_ddr
            # will be dropped
            res_ddr = res_cc
            res_cc = sch.rfactor(res_ddr, fused_atomic_write)
            sch[res_cc].set_scope(scope_cc)
            if self.cube_vector_split:
                res_ub = None
            else:
                res_ub = sch.cache_read(res_cc, scope_ubuf, [res_ddr])
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
            if tiling.get("BL1_shape") and \
                    tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1] \
                    % (kernel_height * kernel_width) != 0:
                tiling["BL1_shape"][1] = _align(tiling.get("BL1_shape")[1] *
                                                tiling.get("BL0_matrix")[1],
                                                kernel_height * kernel_width) \
                    // tiling.get("BL0_matrix")[1]

            # whether axis K is fully loaded in L0A and L0B
            # excluding axis batch
            if not tiling["AL0_matrix"]:
                full_k_l0a = 1
            else:
                full_k_l0a = tiling["AL0_matrix"][1] \
                    // _ceil_div(hw_pad_1, block_dim_hw)
            if DEBUG_MODE:
                print("full_k_in_l0a", full_k_l0a)

            if not tiling["BL0_matrix"]:
                full_k_l0b = 1
            else:
                full_k_l0b = tiling["BL0_matrix"][0] \
                    // _ceil_div(hw_pad_1, block_dim_hw)
            if DEBUG_MODE:
                print("full_k_in_l0b", full_k_l0b)
            return full_k_l0a, full_k_l0b

        def _compute_tiling_parts():
            """
            compute the parts or the factors of tensors

            """

            if not tiling["AL0_matrix"]:  # if grads no tiling in L0A
                tiling["AL1_shape"] = []  # then no tiling in L1

            # dw_cc is (fmap_channel_1*kernel_height*kernel_width,
            #          grads_channel_1, C0_grads, C0_fmap)
            dw_tiling_factor = [tiling["CL0_matrix"][0],
                                tiling["CL0_matrix"][1]]
            # nparts N, nparts M
            # dw_tiling_nparts only describe the nparts from single core to L0
            dw_tiling_nparts = \
                [_ceil_div(fkk // block_dim_cin, dw_tiling_factor[0]),
                 _ceil_div(_ceil_div(cout_g // c0_grads, dw_tiling_factor[1]),
                           block_dim_cout)]

            # tiling parameters of dw_ub
            dw_ub_tiling_factor = [tiling["CUB_matrix"][0],
                                   tiling["CUB_matrix"][1]]
            dw_ub_tiling_nparts = [_ceil_div(dw_tiling_factor[0],
                                             dw_ub_tiling_factor[0]),
                                   _ceil_div(dw_tiling_factor[1],
                                             dw_ub_tiling_factor[1])]

            # only support loading one batch to L1 at a time for now
            # cout:out->single core(sc)->L1
            if tiling["AL1_shape"]:  # if grads needs tiling in L1
                if len(tiling["AL1_shape"]) == 1:  # but no C_1 tiling info
                    tiling["AL1_shape"] = \
                        tiling["AL1_shape"] + [1]
                # nparts K1 in L1, nparts M1 in L1
                grads_l1_tiling_nparts = [_ceil_div(hw_pad_1, block_dim_hw) //
                                          (tiling["AL1_shape"][0] // CUBE_DIM),
                                          dw_tiling_nparts[1]
                                          // tiling["AL1_shape"][1]]
            else:
                grads_l1_tiling_nparts = [1, 1]

            if tiling["BL1_shape"]:  # if fmap needs tiling in L1
                if len(tiling["BL1_shape"]) == 1:  # but no fkk tiling info
                    tiling["BL1_shape"] = \
                        tiling["BL1_shape"] + [1]  # tiling fkk=1
                # DDR to L1 [nparts K1, nparts N1]
                fmap_l1_tiling_nparts = [_ceil_div(hw_pad_1, block_dim_hw) //
                                         (tiling["BL1_shape"][0] // CUBE_DIM),
                                         dw_tiling_nparts[0]
                                         // tiling["BL1_shape"][1]]
            else:
                fmap_l1_tiling_nparts = [1, 1]

            # during L1 to L0 [nparts N1, nparts M1]
            l1_2_l0_tiling_nparts = \
                [dw_tiling_nparts[0] // fmap_l1_tiling_nparts[1],
                 dw_tiling_nparts[1] // grads_l1_tiling_nparts[1]]
            # ka and kb may be different,
            # the min value corresponds to one MMAD,
            # the larger one is []
            if tiling["AL0_matrix"]:  # dw_k equals to ka if L0A needs tiling
                dw_k = tiling["AL0_matrix"][1]
            elif tiling["BL0_matrix"]:
                dw_k = tiling["BL0_matrix"][0]
            else:  # both fully loaded
                dw_k = hw_pad_1 // block_dim_hw

            if DEBUG_MODE:
                print("dw_tiling_factor", dw_tiling_factor)
                print("dw_tiling_nparts", dw_tiling_nparts)
                print("dw_ub_tiling_factor", dw_ub_tiling_factor)
                print("dw_ub_tiling_nparts", dw_ub_tiling_nparts)
                print("grads_l1_tiling_nparts", grads_l1_tiling_nparts)
                print("fmap_l1_tiling_nparts", fmap_l1_tiling_nparts)
                print("l1_2_l0_tiling_nparts", l1_2_l0_tiling_nparts)
                print("dw_k", dw_k)

            tiling_parts_dict = dict()
            tiling_parts_dict["dw_tiling_factor"] = dw_tiling_factor
            tiling_parts_dict["dw_tiling_nparts"] = dw_tiling_nparts
            tiling_parts_dict["dw_ub_tiling_factor"] = dw_ub_tiling_factor
            tiling_parts_dict["dw_ub_tiling_nparts"] = dw_ub_tiling_nparts
            tiling_parts_dict["grads_l1_tiling_nparts"] = \
                                                    grads_l1_tiling_nparts
            tiling_parts_dict["fmap_l1_tiling_nparts"] = fmap_l1_tiling_nparts
            tiling_parts_dict["l1_2_l0_tiling_nparts"] = l1_2_l0_tiling_nparts
            tiling_parts_dict["dw_k"] = dw_k
            return tiling_parts_dict

        def _compute_tiling_factors():
            fmap_l1_tiling_factor_k, grads_l1_tiling_factor_k = None, None
            if self.dynamic_mode and not flag_all_one_case:
                if reduce_split_mode:
                    if tiling["AL1_shape"]:
                        grads_l1_tiling_factor_k = \
                            tiling["AL1_shape"][0] // (dw_k * CUBE_DIM)
                    if tiling["BL1_shape"] and tiling["AL1_shape"]:
                        fmap_l1_tiling_factor_k = \
                            tiling["BL1_shape"][0] // tiling["AL1_shape"][0]
                else:
                    if tiling["BL1_shape"]:
                        fmap_l1_tiling_factor_k = \
                            tiling["BL1_shape"][0] // (dw_k * CUBE_DIM)
                    if tiling["BL1_shape"] and tiling["AL1_shape"]:
                        grads_l1_tiling_factor_k = \
                            tiling["AL1_shape"][0] // tiling["BL1_shape"][0]
            return grads_l1_tiling_factor_k, fmap_l1_tiling_factor_k

        def _reduce_split_mode():
            reduce_split_mode = True
            if self.dynamic_mode == "dynamic_hw":
                if tiling["AL1_shape"] and tiling["BL1_shape"]:
                    # grads and fmap need tiling in L1
                    reduce_split_mode = \
                              tiling["AL1_shape"][0] < tiling["BL1_shape"][0]
                elif tiling["AL1_shape"]:
                    # only grads needs tiling in L1
                    reduce_split_mode = True
                elif tiling["BL1_shape"]:
                    # only fmap needs tiling in L1
                    reduce_split_mode = False
                else:
                    # Neither grads nor fmap need tiling in L1
                    reduce_split_mode = False
            else:
                reduce_split_mode = \
                          grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]
            return reduce_split_mode

        def _l0_attach():
            """
            achieve Al0 and Bl0 compute at loc or ddr

            """
            if self.dynamic_mode and not flag_all_one_case:
                l0a_attach_mode = (dynamic_l0a_attach == "dw_ddr")
                l0b_attach_mode = (dynamic_l0b_attach == "dw_ddr")
            else:
                l0a_attach_mode = \
                         ((batch_num_sc == 1) and (full_k_in_l0a == 1))
                l0b_attach_mode = \
                         ((batch_num_sc == 1) and (full_k_in_l0b == 1))

            if tiling["AL0_matrix"]:
                if l0a_attach_mode:
                    # L0A data is more than that L0C needed, attach to dw_ddr
                    sch[grads_fractal].compute_at(sch[dw_ddr], c_grads_mad_at)
                else:
                    sch[grads_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
            else:  # else: fully load, attach to thread_axis
                sch[grads_fractal].compute_at(sch[dw_ddr], fused_multi_core)

            if tiling["BL0_matrix"]:
                if l0b_attach_mode:
                    sch[fmap_fractal].compute_at(sch[dw_ddr], c_fmap_mad_at)
                else:
                    sch[fmap_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
            else:  # else: fully load, attach to thread_axis
                sch[fmap_fractal].compute_at(sch[dw_ddr], fused_multi_core)

        def _al1_attach():
            """
            achieve Al1 compute at l0c or ddr

            """
            if self.dynamic_mode and not flag_all_one_case:
                al1_attach_mode = (dynamic_al1_attach == "dw_cc")
            else:
                al1_attach_mode = \
                        (grads_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)

            if tiling["AL1_shape"]:
                # if axis K needs split, then attach to dw_cc
                if al1_attach_mode:
                    sch[grads_matrix].compute_at(sch[dw_cc], al1_at_axis)
                else:  # if axis K fully load in L1, attach to dw_ddr
                    sch[grads_matrix].compute_at(sch[dw_ddr], c_grads_l1_at)
            else:  # else: fully load, attach to thread_axis
                sch[grads_matrix].compute_at(sch[dw_ddr], fused_multi_core)

        def _bl1_attach():
            """
            achieve Bl1 compute at l0c or ddr

            """
            fmap_matrix_flag = not self.dynamic_mode or flag_all_one_case
            if self.dynamic_mode and not flag_all_one_case:
                bl1_attach_mode = (dynamic_bl1_attach == "dw_cc")
            else:
                bl1_attach_mode = \
                         (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)

            if tiling["BL1_shape"]:
                # if axis K needs split, then attach to dw_cc
                if  bl1_attach_mode:
                    if fmap_matrix_flag:
                        sch[fmap_matrix].compute_at(sch[dw_cc], bl1_at_axis)
                    if not flag_all_one_case:
                        sch[fmap_l1].compute_at(sch[dw_cc], bl1_at_axis)
                else:  # if axis K fully load in L1, attach to dw_ddr
                    if fmap_matrix_flag:
                        sch[fmap_matrix].compute_at(sch[dw_ddr], c_fmap_l1_at)
                    if not flag_all_one_case:
                        sch[fmap_l1].compute_at(sch[dw_ddr], c_fmap_l1_at)

            else:  # else: fully load, attach to thread_axis
                if fmap_matrix_flag:
                    sch[fmap_matrix].compute_at(sch[dw_ddr], fused_multi_core)
                if not flag_all_one_case:
                    sch[fmap_l1].compute_at(sch[dw_ddr], fused_multi_core)

        def _double_buffer():
            """
            achieve double_buffer

            """
            if not DEBUG_DOUBLE_BUFFER_OFF:
                if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[grads_matrix].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    if not flag_all_one_case:
                        sch[fmap_l1].double_buffer()
                    else:
                        sch[fmap_matrix].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[grads_fractal].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[fmap_fractal].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[dw_cc].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
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
            mad_dict = {"mad_pattern": 3,
                        "k_outer":
                            [batch_insn_o, hw_mad_1_l1_out_at,
                            hw_mad_1_l1_in_at, hw_mad_1_mad_at]}

            if self.dynamic_mode and not flag_all_one_case:
                setfmatrix_dict["set_fmatrix"] = 1
                setfmatrix_dict["conv_fm_c1"] = c1_fmap
                setfmatrix_dict["conv_fm_c0"] = c0_fmap
            else:
                setfmatrix_dict["conv_dilation_h"] = dilation_height
                setfmatrix_dict["conv_dilation_w"] = dilation_width
                mad_dict["mad_pattern"] = 2

            setfmatrix_dict_0 = dict()
            setfmatrix_dict_0["conv_kernel_h"] = kernel_height
            setfmatrix_dict_0["conv_kernel_w"] = kernel_width
            setfmatrix_dict_0["conv_padding_top"] = pad_up
            setfmatrix_dict_0["conv_padding_bottom"] = pad_down
            setfmatrix_dict_0["conv_padding_left"] = pad_left
            setfmatrix_dict_0["conv_padding_right"] = pad_right
            setfmatrix_dict_0["conv_stride_h"] = stride_height
            setfmatrix_dict_0["conv_stride_w"] = stride_width
            setfmatrix_dict_0["conv_fm_c"] = featuremap_channel
            setfmatrix_dict_0["conv_fm_h"] = featuremap_height
            setfmatrix_dict_0["conv_fm_w"] = featuremap_width
            setfmatrix_dict_0["group_flag"] = 1

            if self.dynamic_mode and not flag_all_one_case:
                setfmatrix_dict_0["set_fmatrix"] = 0
                setfmatrix_dict_0["conv_fm_c1"] = c1_fmap
                setfmatrix_dict_0["conv_fm_c0"] = c0_fmap
            else:
                setfmatrix_dict_0["conv_dilation_h"] = dilation_height
                setfmatrix_dict_0["conv_dilation_w"] = dilation_width

            # move grads from ddr to L1
            sch[grads_matrix].emit_insn(grads_matrix.op.axis[0], 'dma_copy')
            # move grads from L1 to L0A
            sch[grads_fractal].emit_insn(grads_fractal.op.axis[0], 'dma_copy')

            # move fmap from ddr to L1
            if not flag_all_one_case:
                if self.dynamic_mode:
                    sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                                'dma_copy', setfmatrix_dict)
                    sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[1],
                                                'im2col_v2', setfmatrix_dict_0)
                else:
                    sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')
                    sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0],
                                            'set_fmatrix', setfmatrix_dict)
                    sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[1],
                                            'im2col')
            else:
                sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0], 'dma_copy')
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0],
                                            'dma_copy')

            # move dw from L0C to UB
            if not self.cube_vector_split:
                sch[dw_ub].emit_insn(dw_ub.op.axis[0], 'dma_copy')

            sch[dw_cc].emit_insn(batch_insn, 'mad', mad_dict)

            # move dw form UB to ddr
            sch[dw_ddr].emit_insn(c_fmap_2_ub_insn, 'dma_copy')

            sch[dw_ddr_reduce].emit_insn(dw_ddr_reduce.op.axis[0],
                                         'phony_insn')
            sch[dw_ub_reduce].emit_insn(dw_ub_reduce.op.axis[0], 'phony_insn')

            sch_list.append(dw_ddr)

        def _get_value(ele):
            res_ele = [ele.value if isinstance(ele, tvm.expr.IntImm) else \
                                                                    ele][0]
            return res_ele

        def _get_para(stride_height, stride_width, pad_up, pad_down, \
                      pad_left, pad_right, kernel_height, kernel_width, \
                      dilation_height, dilation_width):
            return stride_height, stride_width, pad_up, pad_down, pad_left, \
                   pad_right, kernel_height, kernel_width, dilation_height, \
                   dilation_width

        def _get_load3d_para():
            # load_3d parameters
            if self.dynamic_mode and not \
                                DynamicConv2dBpFilterParams.flag_all_one_case:
                stride_height, stride_width, pad_up, pad_down, pad_left, \
                pad_right, kernel_height, kernel_width, dilation_height, \
                dilation_width = _get_para( \
                    _get_value(fmap_fractal.op.attrs['stride'][0]), \
                    _get_value(fmap_fractal.op.attrs['stride'][1]), \
                    _get_value(fmap_fractal.op.attrs['pad'][0]), \
                    _get_value(fmap_fractal.op.attrs['pad'][1]), \
                    _get_value(fmap_fractal.op.attrs['pad'][2]), \
                    _get_value(fmap_fractal.op.attrs['pad'][3]), \
                    _get_value(fmap_fractal.op.attrs['kernel_size'][2]), \
                    _get_value(fmap_fractal.op.attrs['kernel_size'][3]), \
                    _get_value(fmap_fractal.op.attrs['dilation'][2]), \
                    _get_value(fmap_fractal.op.attrs['dilation'][3]))
            else:
                stride_height, stride_width, pad_up, pad_down, pad_left, \
                pad_right, kernel_height, kernel_width, dilation_height, \
                dilation_width = _get_para( \
                    fmap_matrix.op.attrs['stride'][0].value, \
                    fmap_matrix.op.attrs['stride'][1].value, \
                    fmap_matrix.op.attrs['pad'][0].value, \
                    fmap_matrix.op.attrs['pad'][1].value, \
                    fmap_matrix.op.attrs['pad'][2].value, \
                    fmap_matrix.op.attrs['pad'][3].value, \
                    fmap_matrix.op.attrs['kernel_size'][2].value, \
                    fmap_matrix.op.attrs['kernel_size'][3].value, \
                    fmap_matrix.op.attrs['dilation'][2].value, \
                    fmap_matrix.op.attrs['dilation'][3].value)
            return stride_height, stride_width, pad_up, pad_down, pad_left, \
                   pad_right, kernel_height, kernel_width, dilation_height, \
                   dilation_width

        def _set_var_range():
            if self.dynamic_mode == "dynamic_hw":
                var_range = dynamic_para.get("var_range")
                sch.set_var_range(height_fmap, *var_range.get('fmap_h'))
                sch.set_var_range(width_fmap, *var_range.get('fmap_w'))
                sch.set_var_range(height_grads, *var_range.get('dedy_h'))
                sch.set_var_range(width_grads, *var_range.get('dedy_w'))
            elif self.dynamic_mode == "dynamic_batch":
                var_range = dynamic_para.get("var_range")
                sch.set_var_range(batch_fmap, *var_range.get('batch'))

        def _get_tiling():
            if not self.dynamic_mode:
                _l1_limit_check()
                info_dict = {
                    "op_type": "conv2d_backprop_filter",
                    "A_shape": list(grads_shape),
                    "B_shape": list(fmap_shape),
                    "C_shape": list(weight_shape),
                    "A_dtype": str(grads.dtype),
                    "B_dtype": str(fmap.dtype),
                    "C_dtype": str(dw_cc.dtype),
                    "mad_dtype": str(dw_cc.dtype),
                    "padl": pad_left,
                    "padr": pad_right,
                    "padu": pad_up,
                    "padd": pad_down,
                    "strideH": stride_height,
                    "strideW": stride_width,
                    "strideH_expand": 1,
                    "strideW_expand": 1,
                    "dilationH": dilation_height,
                    "dilationW": dilation_width,
                    "group": real_g,
                    "bias_flag": 0,
                    "fused_double_operand_num": 0,
                    "fusion_type": 0,
                    "kernel_name": kernel_name.value,
                }
                tiling = get_tiling(info_dict)
            else:
                tiling = dynamic_para.get('tiling')
            return tiling


        def _handle_tbe_compile_para():
            tbe_compile_para = tiling.get("tbe_compile_para")
            sch.tbe_compile_para, preload = parse_tbe_compile_para(tbe_compile_para)
            if preload:
                if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
                    sch[dw_cc].preload()


        def _get_attach_flag():
            dynamic_l0a_attach = None
            dynamic_l0b_attach = None
            dynamic_al1_attach = None
            dynamic_bl1_attach = None
            if self.dynamic_mode and \
                not DynamicConv2dBpFilterParams.flag_all_one_case:
                dynamic_l0a_attach = dynamic_para.get('dynamic_l0a_attach')
                dynamic_l0b_attach = dynamic_para.get('dynamic_l0b_attach')
                dynamic_al1_attach = dynamic_para.get('dynamic_al1_attach')
                dynamic_bl1_attach = dynamic_para.get('dynamic_bl1_attach')
            return dynamic_l0a_attach, dynamic_l0b_attach, \
                   dynamic_al1_attach, dynamic_bl1_attach

        # for dynamic
        self.dynamic_mode = _get_dyn_mode(dynamic_para)

        # ####################### get computing graph #######################
        dw_ddr = res  # pylint: disable=too-many-statements
        dw_ub = dw_ddr.op.input_tensors[0]
        dw_cc = dw_ub.op.input_tensors[0]
        grads_fractal = dw_cc.op.input_tensors[0]
        fmap_fractal = dw_cc.op.input_tensors[1]
        grads_matrix = grads_fractal.op.input_tensors[0]
        fmap_matrix = fmap_fractal.op.input_tensors[0]
        fmap = fmap_matrix.op.input_tensors[0]
        grads = grads_matrix.op.input_tensors[0]
        group_dict = fmap_matrix.op.attrs['group_dict']
        kernel_name = dw_ddr.op.attrs["kernel_name"]

        # ########################extract parameters##########################

        def _get_default_tiling():

            def _get_factors(val, val_max):
                """
                get the factor of val that smaller than val_max
                """
                factor_max = min(val, val_max)
                for m_fac in range(factor_max, 0, -1):
                    if val % m_fac == 0:
                        return m_fac

            def _get_nbl0():
                """
                nbl0 can not too large to over space.
                nbl0 can be 1, kernel_width, kernel_width * kernel_height
                """
                if kernel_width * kernel_height <= self._lob_size // KB_2_B:
                    nbl0 = kernel_width * kernel_height
                elif kernel_width <= self._lob_size // KB_2_B:
                    return kernel_width
                else:
                    nbl0 = 1
                return nbl0

            def _get_kbl1():
                """
                give the max kbl1 that fmap load kernel_h
                or kernel_h+stride_h in the hi direction in L1
                """
                if width_grads < CUBE_DIM:
                    return CUBE_DIM
                elif width_grads % CUBE_DIM == 0:
                    return width_grads
                else:
                    kbl1_before = width_grads // CUBE_DIM
                    # if kbl1_before is not factor of K, it needs to recalculate
                    c_k1 = _ceil_div(hw_pad_1, CUBE_DIM)
                    k_npart = _ceil_div(c_k1, kbl1_before)
                    kbl1 = c_k1 // k_npart * CUBE_DIM
                    return kbl1

            def _cal_bl1size():
                """
                calculate bl1_size, if kbl1 is a factor of Wo,
                it must loads Hk in H direction
                else it must loads Hk+stride_h in H direction
                """
                if width_grads % kbl1 == 0:
                    phol1 = 1
                else:
                    phol1 = 2
                pbl1hi = (phol1 - 1) * stride_height + (kernel_height - 1) * dilation_height + 1
                bl1_size = pbl1hi * width_fmap * CUBE_DIM * FLOAT16_SIZE
                return bl1_size

            nbl0 = _get_nbl0()
            l0c_mal0_max = self._loc_size // KB_2_B // nbl0
            loa_mal0_max = self._lob_size // KB_2_B
            mal0 = _get_factors(c1_grads, min(l0c_mal0_max, loa_mal0_max))
            kbl1 = _get_kbl1()

            block_batch = _get_factors(batch_grads, self._corenum)
            if nbl0 * mal0 * CUBE_DIM * CUBE_DIM * FLOAT32_SIZE * OPEN_DOUBLE_BUFFER < self._loc_size:
                cl0_pbuffer = 2
            else:
                cl0_pbuffer = 1
            bl1_size = _cal_bl1size()
            al1_size_double = mal0 * CUBE_MUL_SHAPE * FLOAT16_SIZE * OPEN_DOUBLE_BUFFER
            if al1_size_double + bl1_size <= self.l1_size:
                bl1_pbuffer = 2
            else:
                bl1_pbuffer = 1

            default_tiling = {
                'AUB_shape': None, 'BUB_shape': None,
                'AL1_shape': [CUBE_DIM, 1, 1, 1], 'BL1_shape': [kbl1, 1, 1, 1],
                'AL0_matrix': [mal0, 1, CUBE_DIM, CUBE_DIM, 1],
                'BL0_matrix': [1, nbl0, CUBE_DIM, CUBE_DIM, 1],
                'CL0_matrix': [nbl0, mal0, CUBE_DIM, CUBE_DIM, 1],
                'CUB_matrix': [nbl0, mal0, CUBE_DIM, CUBE_DIM, 1],
                'block_dim': [block_batch, 1, 1, 1],
                'cout_bef_batch_flag': 0,
                'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
                'manual_pingpong_buffer': {
                    'AUB_pbuffer': 1, 'BUB_pbuffer': 1,
                    'AL1_pbuffer': 1,
                    'BL1_pbuffer': bl1_pbuffer,
                    'AL0_pbuffer': 2, 'BL0_pbuffer': 2,
                    'CL0_pbuffer': cl0_pbuffer,
                    'CUB_pbuffer': cl0_pbuffer,
                    'UBG_pbuffer': 1}}
            return default_tiling

        cin1_g = group_dict["cin1_g"].value
        cout_g = group_dict["cout_g"].value
        real_g = group_dict["real_g"].value

        batch_grads, c1_grads, height_grads, width_grads, c0_grads \
            = shape_to_list(grads.shape)
        grads_shape = [batch_grads, cout_g // c0_grads,
                       height_grads, width_grads, c0_grads]
        batch_fmap, c1_fmap, height_fmap, width_fmap, c0_fmap \
            = shape_to_list(fmap.shape)
        fmap_shape = [batch_fmap, cin1_g, height_fmap, width_fmap, c0_fmap]
        grads_matrix_batch, grads_matrix_c1, \
        grads_matrix_howo, grads_matrix_c0 = shape_to_list(grads_matrix.shape)
        _, fkk, _, _ = shape_to_list(dw_cc.shape)
        _, _, hw_pad_1, _, _, _ = shape_to_list(fmap_fractal.shape)

        stride_height, stride_width, pad_up, pad_down, pad_left, \
        pad_right, kernel_height, kernel_width, dilation_height, \
        dilation_width \
            = _get_load3d_para()

        featuremap_channel = c1_fmap*c0_fmap
        featuremap_height = height_fmap
        featuremap_width = width_fmap
        kw_dilation = (kernel_width - 1) * dilation_width + 1
        weight_shape = [cout_g, cin1_g,
                        kernel_height, kernel_width, c0_fmap]

        sch = sch_list[0]
        _set_var_range()
        dynamic_l0a_attach, dynamic_l0b_attach, dynamic_al1_attach, \
        dynamic_bl1_attach \
            = _get_attach_flag()
        def _flag_all_one():
            # special supporting for a unique case, there are 2 conditions:
            # (1) height & weight of x/output_backprop/filter are all 1
            # (2) strides is [1,1]
            flag_all_one_case = False
            flag_conv1d_case = False

            height_all_one = False
            width_all_one = False
            if stride_height == 1 and height_grads == 1 and height_fmap == 1 \
                    and kernel_height == 1:
                height_all_one = True
            if stride_width == 1 and width_grads == 1 and width_fmap == 1 \
                    and kernel_width == 1:
                width_all_one = True
            if height_all_one and width_all_one:
                flag_all_one_case = True
                if DEBUG_MODE:
                    print("schedule: this is all one case,"
                          " using special branch")
            if height_all_one and not width_all_one:
                flag_conv1d_case = True
            if self.dynamic_mode == "dynamic_hw":
                dynamic_hw_tiling = dynamic_para.get("tiling")
                flag_conv1d_case = dynamic_hw_tiling.get("flag_conv1d_case")
            return flag_all_one_case, flag_conv1d_case
        flag_all_one_case, flag_conv1d_case = _flag_all_one()
        tiling = _get_tiling()

        if DEBUG_MODE:
            print("grads_shape to tiling_query", grads_shape)
            print("fmap_shape to tiling_query", fmap_shape)
            print("weight_shape to tiling_query", weight_shape)
            print("pad to tiling_query", pad_left, pad_right, pad_up, pad_down)
            print("stride to tiling_query", stride_height, stride_width)
            print("dilation to tiling_query", dilation_height, dilation_width)
            print("Conv2dBackpropFilter: returned from auto_tiling", tiling)
            print("kernel_name:", kernel_name)
        _tiling_shape_check()
        _tiling_buffer_check()
        # if no valid tiling found, the flag is as follows
        if tiling["AL0_matrix"][2] == DEFAULT_TILING_CASE:
            tiling = _get_default_tiling()

        batch_num = batch_grads

        def _get_block_dim():
            if tiling.get("AUB_shape"):
                block_dim_hw = tiling.get("AUB_shape")[0]
            else:
                block_dim_hw = 1
            block_dim_batch = tiling.get("block_dim")[0]
            if self.dynamic_mode == "dynamic_batch":
                block_dim_batch = tvm.min(block_dim_batch, batch_fmap)
            block_dim_cout = tiling.get("block_dim")[2]
            block_dim_cin = tiling.get("block_dim")[1]
            block_dim_group = tiling.get("block_dim")[3]
            return block_dim_hw, block_dim_batch, \
                   block_dim_cout, block_dim_cin, block_dim_group

        block_dim_hw, block_dim_batch, \
        block_dim_cout, block_dim_cin, block_dim_group \
            = _get_block_dim()

        sch[grads_matrix].set_scope(scope_cbuf)
        sch[grads_matrix].storage_align(
            sch[grads_matrix].op.axis[1], CUBE_MUL_SHAPE, 0)

        sch[grads_fractal].set_scope(scope_ca)
        sch[grads_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                        (1, CUBE_DIM), (1, CUBE_DIM))

        def _load3d_fmap_l1_process():
            # shape info:
            # fmap_shape_original_matrix is (batch_size,
            #                               grads_height*grads_width,
            #                               fmap_channel_1,
            #                               kernel_height,
            #                               kernel_width,
            #                               C0_fmap)
            if not self.dynamic_mode:
                fmap_l1 = sch.cache_read(fmap, scope_cbuf, [fmap_matrix])
                if not flag_conv1d_case:
                    sch[fmap_matrix].buffer_align((1, 1),
                                                (width_grads, width_grads),
                                                (1, 1),
                                                (kernel_height, kernel_height),
                                                (kernel_width, kernel_width),
                                                (1, CUBE_DIM))
                else:
                    sch[fmap_matrix].buffer_align((1, 1), (1, 1), (1, 1),
                                              (1, 1),
                                              (kernel_width, kernel_width),
                                              (1, CUBE_DIM))
            else:
                fmap_l1 = fmap_matrix
            return fmap_l1

        if not flag_all_one_case:
            fmap_l1 = _load3d_fmap_l1_process()
        else:
            sch[fmap_matrix].storage_align(
                sch[fmap_matrix].op.axis[1], CUBE_MUL_SHAPE, 0)

        sch[fmap_matrix].set_scope(scope_cbuf)

        sch[fmap_fractal].set_scope(scope_cb)
        sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                       (1, CUBE_DIM), (1, CUBE_DIM))

        full_k_in_l0a, full_k_in_l0b = _full_k_check()

        tiling_parts_dict = _compute_tiling_parts()
        dw_tiling_factor = tiling_parts_dict["dw_tiling_factor"]
        dw_tiling_nparts = tiling_parts_dict["dw_tiling_nparts"]
        dw_ub_tiling_factor = tiling_parts_dict["dw_ub_tiling_factor"]

        grads_l1_tiling_nparts = tiling_parts_dict["grads_l1_tiling_nparts"]
        fmap_l1_tiling_nparts = tiling_parts_dict["fmap_l1_tiling_nparts"]

        l1_2_l0_tiling_nparts = tiling_parts_dict["l1_2_l0_tiling_nparts"]
        dw_k = tiling_parts_dict["dw_k"]
        reduce_split_mode = _reduce_split_mode()
        grads_l1_tiling_factor_k, fmap_l1_tiling_factor_k = \
                                                    _compute_tiling_factors()

        dw_cc, dw_ub, dw_ddr, dw_ub_reduce, dw_ddr_reduce = \
            _atomic_add(sch, dw_cc, dw_ub, dw_ddr)
        # #######################tiling parameters analyze####################
        batch_num_sc = batch_num//block_dim_batch
        if DEBUG_MODE:
            print("start analyzing tiling parameters")
            print("axis K: block_dim_batch", block_dim_batch)
            print("axis K: block_dim_hw", block_dim_hw)
            print("axis N: block_dim_cin", block_dim_cin)
            print("axis M: block_dim_cout", block_dim_cout)
            print("axis G: block_dim_group", block_dim_group)
            print("batch_num_sc", batch_num_sc)

        # #############################split axis N##########################
        # dw_shape is (fmap_channel_1*kernel_height*kernel_width,
        #             grads_channel_1, C0_grads, C0_fmap)
        g_multicore, g_axis = sch[dw_ddr].split(sch[dw_ddr].op.axis[0],
                                                nparts=block_dim_group)
        c_fmap_multicore, c_fmap_mad_at \
            = sch[dw_ddr].split(sch[dw_ddr].op.axis[1], nparts=block_dim_cin)

        c_fmap_mad_at, c_fmap_mad_insn \
            = sch[dw_ddr].split(c_fmap_mad_at, nparts=dw_tiling_nparts[0])

        c_fmap_l1_ori, c_fmap_mad_at \
            = sch[dw_ddr].split(c_fmap_mad_at, nparts=fmap_l1_tiling_nparts[1])

        def _ddr_n_split():
            # for N axis, if Hk and Wk needs split, do explict split
            if not flag_all_one_case:
                if tiling.get("BL1_shape"):
                    # n1 in L1
                    nc_cc = tiling.get("CL0_matrix")[0] *\
                        tiling.get("BL1_shape")[1]
                else:  # "BL1_shape" is []
                    nc_cc = c1_fmap*kernel_width*kernel_height//block_dim_cin

                factor_kw = _ceil_div(kernel_width, nc_cc)
                factor_kh = _ceil_div(
                    kernel_width*kernel_height, nc_cc)//factor_kw
                factor_c1 = \
                    _ceil_div(c1_fmap*kernel_width*kernel_height //
                              block_dim_cin, nc_cc) // factor_kw // factor_kh

                if DEBUG_MODE:
                    print("N axis split in L1",
                          factor_c1, factor_kh, factor_kw)

                c_fmap_l1_out, c_fmap_l1_at \
                    = sch[dw_ddr].split(c_fmap_l1_ori, factor_kw)

                c_fmap_l1_c1, c_fmap_l1_kh \
                    = sch[dw_ddr].split(c_fmap_l1_out, factor_kh)
            else:
                c_fmap_l1_out, c_fmap_l1_at \
                    = sch[dw_ddr].split(c_fmap_l1_ori, 1)

                c_fmap_l1_c1, c_fmap_l1_kh \
                    = sch[dw_ddr].split(c_fmap_l1_out, 1)
            return c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at
        c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at = _ddr_n_split()

        # split axis M
        c_grads_mad_at, c_grads_mad_insn \
            = sch[dw_ddr].split(sch[dw_ddr].op.axis[2],
                                dw_tiling_factor[1]*CUBE_DIM)

        c_grads_multicore, c_grads_mad_at \
            = sch[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)

        c_grads_l1_at, c_grads_mad_at = \
            sch[dw_ddr].split(c_grads_mad_at, nparts=grads_l1_tiling_nparts[1])

        # reorder according to requirments of mmad EmitInsn
        sch[dw_ddr].reorder(sch[dw_ddr].op.reduce_axis[0],
                            g_multicore,
                            c_grads_multicore, c_fmap_multicore, g_axis,
                            c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                            c_grads_l1_at,
                            c_fmap_mad_at, c_grads_mad_at,
                            c_fmap_mad_insn, c_grads_mad_insn)

        def _ub_and_cc_attach():
            # optimization by move small loops to outer
            reorder_flag = False
            # during L1 to L0, if M loop is smaller, then move to outer
            if l1_2_l0_tiling_nparts[0] > l1_2_l0_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_mad_at, c_fmap_mad_at)
                reorder_flag = True
            # during sc to L1, if M loop is smaller, then move to outer
            if fmap_l1_tiling_nparts[1] > grads_l1_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_l1_at,
                                    c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at)

            if not self.cube_vector_split:
                # dw_ub attach
                # dw_ub split
                c_fmap_2_ub_at, c_fmap_2_ub_insn \
                    = sch[dw_ddr].split(c_fmap_mad_insn, dw_ub_tiling_factor[0])
                # dw_ub attach
                sch[dw_ub].compute_at(sch[dw_ddr], c_fmap_2_ub_at)
            else:
                c_fmap_2_ub_insn = c_fmap_mad_insn

            # dw attach
            if reorder_flag:
                sch[dw_cc].compute_at(sch[dw_ddr], c_fmap_mad_at)
            else:
                sch[dw_cc].compute_at(sch[dw_ddr], c_grads_mad_at)
            return c_fmap_2_ub_insn
        c_fmap_2_ub_insn = _ub_and_cc_attach()

        def _dw_cc_split():
            # get the 3 reduce axis of dw_cc
            batch_axis_sc, k_1_axis_sc, k_0 = sch[dw_cc].op.reduce_axis
            # dw_k is the part for one MMAD
            hw_mad_1_mad_at, hw_mad_1_mad_insn \
                = sch[dw_cc].split(k_1_axis_sc, dw_k)
            # mad_pattern :2 , the 1st axis should be 1, so do a fake split
            batch_insn_o, batch_insn = sch[dw_cc].split(batch_axis_sc, 1)

            # K of AL1 and BL1 can be different, there are 2 split methods
            # on which one is larger
            if reduce_split_mode:
                hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                    hw_mad_1_mad_at, nparts=grads_l1_tiling_nparts[0])
                hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                    hw_mad_1_l1_at, nparts=fmap_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_in_at
                bl1_at_axis = hw_mad_1_l1_out_at
            else:
                hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                    hw_mad_1_mad_at, nparts=fmap_l1_tiling_nparts[0])
                hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                    hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_out_at
                bl1_at_axis = hw_mad_1_l1_in_at

            # split dw_cc.op.axis[0](N1), factor is one MMAD
            fkk_mad_at, fkk_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[2], dw_tiling_factor[0])

            # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
            lc_mad_at, lc_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[3],
                                dw_tiling_factor[1] * CUBE_DIM)
            sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, sch[dw_cc].op.axis[0],
                               batch_insn_o, hw_mad_1_l1_out_at,
                               hw_mad_1_l1_in_at, hw_mad_1_mad_at,
                               batch_insn, fkk_mad_insn, lc_mad_insn,
                               sch[dw_cc].op.axis[4], hw_mad_1_mad_insn, k_0)
            return al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
                   hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn

        def _dynamic_hw_dw_cc_split():
            # get the 2 reduce axis of dw_cc
            batch_axis_sc, k_1_axis_sc = sch[dw_cc].op.reduce_axis
            # dw_k is the part for one MMAD
            hw_mad_1_mad_at, hw_mad_1_mad_insn \
                = sch[dw_cc].split(k_1_axis_sc, dw_k * CUBE_DIM)
            # mad_pattern :2 , the 1st axis should be 1, so do a fake split
            batch_insn_o, batch_insn = sch[dw_cc].split(batch_axis_sc, 1)
            if reduce_split_mode:
                # the factor of grads_l1 is smaller than fmap_l1
                hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                    hw_mad_1_mad_at, grads_l1_tiling_factor_k)
                if tiling["AL1_shape"] and tiling["BL1_shape"]:
                    # grads and fmap need tiling in L1
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, fmap_l1_tiling_factor_k)
                elif tiling["AL1_shape"]:
                    # only grads needs tiling in L1
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, nparts=fmap_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_in_at
                bl1_at_axis = hw_mad_1_l1_out_at
            else:
                # the factor of fmap_l1 is smaller than grads_l1
                if tiling["AL1_shape"] and tiling["BL1_shape"]:
                    # grads and fmap need tiling in L1
                    hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                        hw_mad_1_mad_at, fmap_l1_tiling_factor_k)
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, grads_l1_tiling_factor_k)
                elif tiling["BL1_shape"]:
                    # only fmap needs tiling in L1
                    hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                        hw_mad_1_mad_at, fmap_l1_tiling_factor_k)
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
                else:
                    # Neither grads nor fmap need tiling in L1
                    hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                        hw_mad_1_mad_at, nparts=fmap_l1_tiling_nparts[0])
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_out_at
                bl1_at_axis = hw_mad_1_l1_in_at
            # split dw_cc.op.axis[0](N1), factor is one MMAD
            fkk_mad_at, fkk_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[2], dw_tiling_factor[0])

            # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
            lc_mad_at, lc_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[3],
                                dw_tiling_factor[1] * CUBE_DIM)
            sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, sch[dw_cc].op.axis[0],
                               batch_insn_o, hw_mad_1_l1_out_at,
                               hw_mad_1_l1_in_at, hw_mad_1_mad_at,
                               batch_insn, fkk_mad_insn, lc_mad_insn,
                               sch[dw_cc].op.axis[4], hw_mad_1_mad_insn)
            return al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
                   hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn

        if self.dynamic_mode and not flag_all_one_case:
            al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn \
                = _dynamic_hw_dw_cc_split()
        else:
            al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn \
                = _dw_cc_split()
        # #############################multi core#############################
        def _bind_core():
            fused_multi_core = \
                sch[dw_ddr].fuse(sch[dw_ddr].op.reduce_axis[0], g_multicore,
                                 c_grads_multicore, c_fmap_multicore)
            fused_multi_core, pragma_at = \
                sch[dw_ddr].split(fused_multi_core, 1)
            block = tvm.thread_axis("blockIdx.x")
            sch[dw_ddr].bind(fused_multi_core, block)

            blocks = (block_dim_batch * block_dim_cin *
                        block_dim_cout * block_dim_hw * block_dim_group)
            if blocks == block_dim_batch:
                sch[dw_ddr].pragma(pragma_at,
                                'json_info_batchBindOnly')
            return fused_multi_core
        fused_multi_core = _bind_core()

        def _split_w_for_conv1d():
            if flag_conv1d_case:
                # the offset according to multicore
                block_div = (block_dim_batch * block_dim_cout *
                                block_dim_cin * block_dim_group)
                hw_block_offset = fused_multi_core // block_div * \
                    _ceil_div(hw_pad_1, block_dim_hw) * CUBE_DIM
                if self.dynamic_mode:
                    bool_dw_cc = dynamic_bl1_attach == "dw_cc"
                else:
                    bool_dw_cc = tiling["BL1_shape"] and\
                        (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)
                if bool_dw_cc:
                    if self.dynamic_mode == "dynamic_hw":
                        bl1_k = tiling.get("BL1_shape")[0]
                        al1_k = grads_matrix_howo
                        if tiling.get("AL1_shape"):
                            al1_k = tiling.get("AL1_shape")[0]
                        if bl1_at_axis == hw_mad_1_l1_in_at:
                            # hw splited two times before BL1 attach
                            hw_parts_offset = (hw_mad_1_l1_out_at.var * al1_k +
                                          hw_mad_1_l1_in_at.var * bl1_k)
                        else:
                            # hw splited one time before BL1  attach
                            hw_parts_offset = hw_mad_1_l1_out_at.var * bl1_k
                    else:
                        if grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]:
                            # hw splited one time before BL1  attach
                            hw_parts_offset = \
                                hw_mad_1_l1_out_at * tiling["BL1_shape"][0]
                        else:
                            # hw splited 2 times before BL1 attach
                            hw_parts_offset = \
                                ((hw_mad_1_l1_out_at * fmap_l1_tiling_nparts[0] //
                                 grads_l1_tiling_nparts[0] + hw_mad_1_l1_in_at) * 
                                 tiling["BL1_shape"][0])

                    hw_offset = hw_block_offset + hw_parts_offset
                else:
                    # k axis is full load
                    hw_offset = hw_block_offset
                # the offset of w according to that of k_axis
                hw_offset_with_pad = stride_width * hw_offset - pad_left
                # the extend of w according to that of k_axis
                if not tiling["BL1_shape"]:
                    kbl1_data = _ceil_div(hw_pad_1 * CUBE_DIM, block_dim_hw)
                else:
                    kbl1_data = tiling["BL1_shape"][0]
                hw_extend =\
                    (kbl1_data-1)*stride_width + kw_dilation
                sch[fmap_l1].buffer_tile((None, None),
                                         (None, None),
                                         (None, None),
                                         (hw_offset_with_pad, hw_extend),
                                         (None, None))

        def _get_bl1_bound():
            """
            for bl1_bound set for dynamic

            Returns :
            bl1_bound: dynamic_mode param for storage_bound

            additional_rows: int
                param for buffer_tile in multi-core cases
                -1 indicates not used

            ho_len: int
                actually number of lines loaded in dynamic_batch
                -1 indicates not used
            """

            def _set_additional_rows(bl1_k, width_grads):
                """
                additional rows set for load3d

                Returns : 1. Exp for dynamic_hw
                    2. int for dynamic_batch
                """

                if self.dynamic_mode == "dynamic_hw":
                    # dynamic_hw returns exp
                    # generally fix shape:
                    # 1. kbl1 < wo: ho = 1 if wo % kbl1 == 0 else ho = 2
                    # 2. kbl1 >= wo: ho = ceilDiv(kbl1, wo) if kbl1%wo == 0 else ho = ceilDiv(kbl1,wo) + 1
                    # dynamic_hw need to consider multi_core tail blocks: hw_single_core%wo == 0
                    return tvm.select(
                               bl1_k < width_grads,
                               tvm.select(
                                   tvm.all(
                                       tvm.floormod(width_grads, bl1_k) == 0,
                                       tvm.floormod(hw_single_core_factor,
                                                    width_grads) == 0),
                                   1,
                                   LOOSE_LINE_CONDITION),
                               tvm.select(
                                   tvm.all(
                                       tvm.floormod(bl1_k, width_grads) == 0,
                                       tvm.floormod(hw_single_core_factor,
                                                    width_grads) == 0),
                                   0,
                                   tvm.select(
                                       tvm.floormod(bl1_k*2, width_grads) == 0,
                                       1,
                                       LOOSE_LINE_CONDITION)))

                if  bl1_k % width_grads == 0:
                    # fully loaded dont extra line
                    return 0

                if (bl1_k * 2 % width_grads == 0
                    or bl1_k % width_grads == 1):
                    # special cases need only 1 extra line
                    return 1

                # other situations need 2 extra lines
                return 2

            if tiling.get("BL1_shape"):
                bl1_k = tiling.get("BL1_shape")[0]
                bl1_k_full = bl1_k
                if flag_all_one_case:
                    additional_rows = -1
                    ho_len = -1
                elif (self.dynamic_mode == "dynamic_batch"
                      and bl1_k < width_grads):
                    additional_rows = -1
                    ho_len = 2
                    if not flag_conv1d_case:
                        # check if only need to load int times of bl0
                        ho_len = 1 if (width_grads % bl1_k == 0) \
                            else LOOSE_LINE_CONDITION
                        hi_max = kernel_height + (ho_len - 1) * stride_height
                        bl1_k_full = width_fmap * hi_max
                else:
                    # load3d can not split width_grads
                    additional_rows = _set_additional_rows(bl1_k, width_grads)

                    ho_len = tvm.floordiv(bl1_k, width_grads) + additional_rows
                    hi_max = kernel_height + (ho_len - 1) * stride_height
                    bl1_k_full = hi_max * width_fmap

                bl1_bound = bl1_k_full * tiling.get("BL1_shape")[1] * \
                            tiling.get("BL0_matrix")[1] // \
                            (kernel_height * kernel_width) * c0_fmap

            else:
                bl1_k_full = _ceil_div(width_fmap * height_fmap, block_dim_hw)
                bl1_k_full = _align(bl1_k_full, CUBE_DIM)
                bl1_bound = bl1_k_full * c1_fmap * c0_fmap
                ho_len = height_fmap
                additional_rows = -1

            if flag_conv1d_case:
                bl1_hi = 1
                if not tiling["BL1_shape"]:
                    kbl1_data = _ceil_div(hw_pad_1 * CUBE_DIM, block_dim_hw)
                    nbl1_c1 = c1_fmap
                else:
                    kbl1_data = tiling["BL1_shape"][0]
                    nbl1_c1 = ((tiling.get("BL1_shape")[1] * 
                               tiling.get("BL0_matrix")[1]) // 
                               (kernel_height * kernel_width))
                bl1_wi = (kbl1_data - 1) * stride_width + kw_dilation
                bl1_k_full = bl1_hi * bl1_wi
                bl1_bound = bl1_k_full * nbl1_c1 * c0_fmap

            return bl1_bound, additional_rows, ho_len

        def _get_al1_bound():
            # al1 set storage bound
            if tiling.get("AL1_shape"):
                al1_m = tiling.get("AL1_shape")[1] * \
                              tiling.get("AL0_matrix")[0] * CUBE_DIM
                al1_k = tiling.get("AL1_shape")[0]
                al1_bound = al1_k * al1_m
            else:
                al1_m = grads_matrix_c1 * grads_matrix_c0
                al1_bound = grads_matrix_howo * al1_m
            return al1_bound

        def _dynamic_bl1_buffer_tile(ho_len):
            # buffer_tile for dynamic mode

            def _set_tile_mode():
                # set buffer tile mode for dynamic

                if dynamic_bl1_attach == "dw_cc":
                    # hw need both multi-core offset and k_axis offset
                    return "tile_h_dw_cc"

                elif dynamic_bl1_attach == "dw_ddr":
                    # hw only need multi_core offset
                    return "tile_h_dw_ddr"

                return "None"

            def _set_tile_params(ho_len, tile_mode):
                ho_min = 0

                # axis_k offset
                wi_min = -pad_left
                wi_extent = width_fmap + pad_left + pad_right

                bl1_k = tiling.get("BL1_shape")[0]
                al1_k = grads_matrix_howo
                if tiling.get("AL1_shape"):
                    al1_k = tiling.get("AL1_shape")[0]

                if bl1_at_axis == hw_mad_1_l1_in_at:
                    # hw splited two times before BL1 attach
                    axis_k_var = (hw_mad_1_l1_out_at.var * al1_k +
                                  hw_mad_1_l1_in_at.var * bl1_k)
                else:
                    # hw splited one time before BL1 attach
                    axis_k_var = hw_mad_1_l1_out_at.var * bl1_k

                if self.dynamic_mode == "dynamic_batch":
                    # multi_core offset
                    multi_core_offset = tvm.floordiv(
                                            tvm.floordiv(fused_multi_core,
                                                         block_dim_cout *
                                                         block_dim_cin),
                                            tvm.floordiv(batch_fmap - 1,
                                                         _ceil_div(batch_fmap,
                                                                   block_dim_batch)) +
                                            1) * hw_single_core_factor

                    if tile_mode == "tile_h_dw_cc":
                        ho_min = tvm.floordiv(multi_core_offset + axis_k_var,
                                              width_grads)

                    elif tile_mode == "tile_h_dw_ddr":
                        ho_min = tvm.floordiv(multi_core_offset, width_grads)

                elif self.dynamic_mode == "dynamic_hw":
                    # multi_core offset
                    block_div = (block_dim_batch * block_dim_cout * block_dim_cin)
                    multi_core_offset = fused_multi_core // block_div * \
                                      hw_single_core_factor

                    if tile_mode == "tile_h_dw_cc":
                        hw_block_offset = multi_core_offset + axis_k_var
                        ho_min = tvm.floordiv(hw_block_offset, width_grads)

                    elif tile_mode == "tile_h_dw_ddr":
                        hw_block_offset = multi_core_offset
                        ho_min = tvm.floordiv(hw_block_offset, width_grads)

                    # dynamic_hw need process multi_core tail blocks
                    ho_len = _ceil_div(tvm.floormod(hw_block_offset,
                                                     width_grads) +
                                       bl1_k,
                                       width_grads)

                hi_min = ho_min * stride_height - pad_up

                # Calculate the min and extent of the h dimension bound
                hi_extent = kernel_height + (ho_len - 1) * stride_height

                return hi_min, hi_extent, wi_min, wi_extent

            if not flag_all_one_case and not flag_conv1d_case:
                tile_mode = _set_tile_mode()

                hi_min, hi_extent, wi_min, wi_extent = \
                    _set_tile_params(ho_len, tile_mode)

                if "tile_h" in tile_mode:
                    hi_min, hi_extent = hi_min, hi_extent
                else:
                    hi_min, hi_extent = None, None

                sch[fmap_l1].buffer_tile((None, None), (None, None),
                                         (hi_min, hi_extent),
                                         (wi_min, wi_extent),
                                         (None, None))

            # mem management in dynamic mode
            sch[grads_matrix].set_storage_bound(al1_bound)
            sch[fmap_matrix].set_storage_bound(bl1_bound)

        def _dynamic_memory_management():
            # disable_allocate
            sch.disable_allocate(scope_cbuf)
            sch.disable_allocate(scope_ca)
            sch.disable_allocate(scope_cb)
            sch.disable_allocate(scope_cc)
            if not self.cube_vector_split:
                sch.disable_allocate(scope_ubuf)

            # mem_unique
            sch[grads_matrix].mem_unique()
            sch[fmap_matrix].mem_unique()
            sch[grads_fractal].mem_unique()
            sch[fmap_fractal].mem_unique()
            sch[dw_cc].mem_unique()
            if not self.cube_vector_split:
                sch[dw_ub].mem_unique()

        _split_w_for_conv1d()
        _l0_attach()
        _al1_attach()
        _bl1_attach()
        _double_buffer()
        _emit_insn()
        _handle_tbe_compile_para()
        if self.dynamic_mode:
            hw_single_core_factor = _ceil_div(hw_pad_1, block_dim_hw) * \
                                    CUBE_DIM
            hw_single_core_factor = _align(hw_single_core_factor,
                                           dw_k * CUBE_DIM)
            al1_bound = _get_al1_bound()
            bl1_bound, additional_rows, ho_len = _get_bl1_bound()
            _dynamic_bl1_buffer_tile(ho_len)
            _dynamic_memory_management()

        return True
