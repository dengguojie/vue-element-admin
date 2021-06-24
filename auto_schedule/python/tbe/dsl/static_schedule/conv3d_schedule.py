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
Schedule of conv3d.
"""
from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from tbe.common.tiling.get_tiling import get_tiling
from tbe.dsl.compute import conv3d_compute
from tbe.dsl.compute import util as compute_util


# tiling check
_TILING_L1_SHAPE_DIM = 4
_TILING_AL0_MATRIX_DIM = [6]
_TILING_BL0_MATRIX_DIM = [6, 0]
_TILING_CL0_MATRIX_DIM = [6]
_TILING_CUB_MATRIX_DIM = [6]
_TILING_BLOCK_DIM_DIM = [4]
_TILING_FLOAT16_MKN = 16
_VALID_TILING_NUM = 32
_CONV_NUM = 5


class CceConv3dOp:
    """
    cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing paagma when using calculate

    Returns
    -------
    cceop_instance : instance of cceop

    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._scope = scope
        self._schedule = None
        self._tensor_map = conv3d_compute.Conv3DParam._TENSOR_MAP
        self._dim_map = conv3d_compute.Conv3DParam.dim_map
        self._tiling = conv3d_compute.Conv3DParam.tiling
        self._fused_op_num = 0
        self._res_tensor = None
        self.body_ops = []
        self.input_ops = []
        self.output_ops = []
        self._has_vector_flag = False
        self._in_quant_sqrt_flag = False
        self.dsl_flag = self._tensor_map["dsl_flag"]
        self.var_map = conv3d_compute.Conv3DParam.var_map
        self.tiling_case = {}
        self.var_range = {}

    def _get_value(self, ele):
        res_ele = [ele.value if isinstance(ele, tvm.expr.IntImm) else \
                                                                ele][0]
        return res_ele

    def _int_ceil_div_tvm(self, num_a, num_b):
        """
        tvm.floordiv result
        """
        return tvm.floordiv((num_a + num_b - 1), num_b)

    def _weight_to_bl1(self, tiling, filter_matrix, weight, c_col):
        """
        weight to bl1.

        Parameters
        ----------
        tiling : case tiling

        filter_matrix : full load

        weight: input weight tensor

        c_col: stage loc

        Returns
        -------
        bl1
        """
        sch = self._schedule
        if tiling["BL0_matrix"] == filter_matrix:
            tiling["BL0_matrix"] = []

        if tiling["BL0_matrix"] == []:
            tiling["BL1_shape"] = None

        if tiling["BL1_shape"] is not None:
            bl1 = sch.cache_read(weight, tbe_platform_info.scope_cbuf, [c_col])
        else:
            # tiling["BL1_shape"] is not None ---> weight from OUT To l0b directly
            bl1 = weight
        return bl1

    def _factor_al1_bl1(self, tiling, c_factor):
        """
        get al1_factor and bl1_factor.

        Parameters
        ----------
        tiling : case tiling

        c_factor : [cout//nc//n0, howo//mc//mo]

        Returns
        -------
        al1_factor, bl1_factor

        """
        if len(tiling["AL1_shape"]) == 1:
            tiling["AL1_shape"] = tiling["AL1_shape"] + [1]
        equivalent_k = self._tensor_map["group_dict"]["cin1_g"] * self._tensor_map["filter_d"]
        if tiling["AL1_shape"]:
            al1_factor = [
                equivalent_k // tiling["AL1_shape"][0],
                compute_util.int_ceil_div(c_factor[1], tiling["AL1_shape"][1])
            ]
        else:
            al1_factor = [1, 1]

        if tiling["BL1_shape"]:
            if len(tiling["BL1_shape"]) > 1:
                if c_factor[0] % tiling["BL1_shape"][1] != 0:
                    cube_err.raise_err_one_para('E62301', 'conv3d', str(tiling["BL1_shape"][1]))

            if len(tiling["BL1_shape"]) == 1:
                tiling["BL1_shape"] = tiling["BL1_shape"] + [1]
            bl1_factor = [
                compute_util.int_ceil_div(equivalent_k, tiling["BL1_shape"][0]),
                compute_util.int_ceil_div(c_factor[0], tiling["BL1_shape"][1])
            ]
        else:
            bl1_factor = [1, tiling["block_dim"][1]]

        outer_factor = max(al1_factor[0], bl1_factor[0])
        inner_factor = min(al1_factor[0], bl1_factor[0])
        if outer_factor % inner_factor != 0:
            cube_err.raise_err_two_paras('E62303', 'conv3d', str(al1_factor[0]), str(bl1_factor[0]))

        return al1_factor, bl1_factor

    def _reorder_axis(self, tiling, al1_factor, bl1_factor, double_buffer_flag,
                      reorder_axis_dict, res_c):
        """
        reorder axis.

        Parameters
        ----------
        tiling : case tiling

        al1_factor : al1 split factor [c1//kAl1, howo//mc//m0//m_Al1]

        bl1_factor : bl1 split factor [c1//kBl1, howo//nc//n0//n_Bl1]

        double_buffer_flag : flag of double buffer

        reorder_axis_dict : axis to reorder

        res_c : c ddr

        Returns
        -------
        reorder flag

        """
        sch = self._schedule
        reorder_flag = False
        noi = reorder_axis_dict["noi"]
        m_outer_outer_outer_inner = reorder_axis_dict[
            "m_outer_outer_outer_inner"]
        c_outer_outer_outer_inner = reorder_axis_dict[
            "c_outer_outer_outer_inner"]

        if not tiling["BL1_shape"]:
            reorder_flag = True
        elif double_buffer_flag["AL1_pbuffer"] == double_buffer_flag["BL1_pbuffer"] and \
                isinstance(al1_factor[1], int):
            if bl1_factor[1] >= al1_factor[1]:
                reorder_flag = True
        elif double_buffer_flag["BL1_pbuffer"] == 2:
            reorder_flag = True

        if reorder_flag:
            sch[res_c].reorder(m_outer_outer_outer_inner, noi,
                               c_outer_outer_outer_inner)
        else:
            sch[res_c].reorder(c_outer_outer_outer_inner,
                               m_outer_outer_outer_inner, noi)
        return reorder_flag

    def _attach_bl0(self, tiling, stage_dict, bl0, coo, noo):
        """
        bl0 compute at.

        Parameters
        ----------
        tiling : case tiling

        stage_dict : c_col res_c

        bl0 : loc axis

        coo : loc axis

        noo : res axis

        Returns
        -------

        """
        sch = self._schedule
        res_c = stage_dict["res_c"]
        c_col = stage_dict["c_col"]
        if tiling["BL0_matrix"]:
            sch[bl0].compute_at(sch[c_col], coo)
        else:
            sch[bl0].compute_at(sch[res_c], noo)

        return True

    def _al1_bl1_axis(self, stage_dict, al1_factor, bl1_factor, k_outer_outer):
        """
        splite al1 and bl1 k_axis.

        Parameters
        ----------
        stage_dict : c_col res_c

        al1_factor : al1 split factor [c1//kAl1, howo//mc//m0//m_Al1]

        bl1_factor : bl1 split factor [c1//kBl1, howo//nc//n0//n_Bl1]

        k_outer_outer : loc axis

        Returns
        -------
        al1_at_ccol_axis, bl1_at_ccol_axis, k_axis_dict

        """
        c_col = stage_dict["c_col"]
        sch = self._schedule
        if al1_factor[0] > bl1_factor[0]:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_col].split(
                k_outer_outer, nparts=al1_factor[0])
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[
                c_col].split(k_outer_outer_outer, nparts=(bl1_factor[0]))
            al1_at_ccol_axis = k_outer_outer_outer_inner
            bl1_at_ccol_axis = k_outer_outer_outer_outer
        else:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_col].split(
                k_outer_outer, nparts=bl1_factor[0])
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[
                c_col].split(k_outer_outer_outer, nparts=(al1_factor[0]))
            al1_at_ccol_axis = k_outer_outer_outer_outer
            bl1_at_ccol_axis = k_outer_outer_outer_inner

        k_axis_dict = {
            "k_outer_outer_outer_outer": k_outer_outer_outer_outer,
            "k_outer_outer_outer_inner": k_outer_outer_outer_inner,
            "k_outer_outer_inner": k_outer_outer_inner
        }

        return al1_at_ccol_axis, bl1_at_ccol_axis, k_axis_dict

    def _get_nbuffer_al1_flag(self, tiling, compute_al1_axis, buffer_dict,
                              k_outer_outer_inner, k_outer_outer_inner_size,
                              shape_w):
        """
        get al1 nbuffer flag.

        Parameters
        ----------
        tiling : case tiling

        compute_al1_axis : al1 compute at axis

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        k_outer_outer_inner : loc axis

        k_outer_outer_inner_size : k_outer_outer_inner size

        shape_w : weight shape

        Returns
        -------
        nbuffer_flag_al1, compute_al1_axis, nbuffer_axis

        """
        sch = self._schedule
        c_col = buffer_dict["c_col"]
        nbuffer_flag_al1 = False
        nbuffer_axis = {}
        nbuffer_size = 1
        if tiling["A_overhead_opt_flag"]:
            if (shape_w[-3] * shape_w[-2]) % tiling["AL0_matrix"][1] == 0:
                nbuffer_size = shape_w[-3] * shape_w[-2] // tiling["AL0_matrix"][1]
            else:
                nbuffer_size = shape_w[-3] * shape_w[-2]
            if int(k_outer_outer_inner_size % nbuffer_size
                   ) == 0 and k_outer_outer_inner_size > nbuffer_size:
                k_outer_outer_inner_outer, k_outer_outer_inner_inner = sch[
                    c_col].split(k_outer_outer_inner, nbuffer_size)
                nbuffer_flag_al1 = True
                compute_al1_axis[
                    "k_outer_outer_inner_outer"] = k_outer_outer_inner_outer
                nbuffer_axis = {
                    "k_outer_outer_inner_outer": k_outer_outer_inner_outer,
                    "k_outer_outer_inner_inner": k_outer_outer_inner_inner
                }

        return nbuffer_flag_al1, compute_al1_axis, nbuffer_axis, k_outer_outer_inner_size // nbuffer_size
    
    def _get_cyclebuffer_flag(self, tiling, shape_w, w_dtype, shape_fmap, stride_d,
                              pad_d, l0a_load2d_flag, dilation_h, dilation_w):
        """
        calculate whether to do cyclebuffer
        
        Parameters
        ----------
        tiling : tiling_new
        
        shape_w : filter shape

        shape_fmap : fmap shape

        stride_d : d channel stride

        pad_d : pad of d direction

        l0a_load2d_flag : whether fmap to load2d

        dilation_h : dilation on h direction

        dilation_w : dilation on w direction
        
        return
        ---------
        cyclebuffer_flag
        """
        cyclebuffer_flag = False
        filter_d = shape_w[1]
        filter_h = shape_w[3]
        filter_w = shape_w[4]
        fmap_d = shape_fmap[1]
        channel_c1 = shape_fmap[2]
        d_dim = tiling["block_dim"][-1]
        matrix_ka = tiling["AL0_matrix"][1] * tiling["AL0_matrix"][-1]
        d_out = (fmap_d + pad_d[0] + pad_d[1] - filter_d) // stride_d + 1
        cyc_size = 0
        dilated_k_w = (filter_w - 1) * dilation_w + 1
        dilated_k_h = (filter_h - 1) * dilation_h + 1
        if tiling["AL1_shape"]:
            cyc_size = int(tiling["AL1_shape"][0] * tiling["AL1_shape"][-1] // \
                           (dilated_k_w * dilated_k_h * tbe_platform.CUBE_MKN[w_dtype]['mac'][1]))
        
        if cyc_size == filter_d * channel_c1:
            cyclebuffer_flag = True

        if l0a_load2d_flag or filter_d <= stride_d or d_out == d_dim:
            cyclebuffer_flag = False
        
        if channel_c1 * filter_h * filter_w % matrix_ka != 0:
            cyclebuffer_flag = False
        
        return cyclebuffer_flag

    def _cachebuffer(self, spec_node_list):
        """
        tensor not for conv set scope.

        Parameters
        ----------
        bodyops : body dict

        inputops : input dict

        spec_node_list : res tensor

        Returns
        -------

        """
        for lop in self.body_ops:
            if (("conv3d" not in lop["op"] or
                 (self.dsl_flag and (lop["op"] == "conv3d_C"))) and
                lop['dst_buffer'] not in spec_node_list):
                self._schedule[lop["dst_buffer"]].set_scope(tbe_platform_info.scope_ubuf)
        for lop in self.input_ops:  # not for A, B, DeqScale, ReqScale,
            if "conv3d" in lop["op"]:
                continue
            if ("bias_tensor" in lop["dst_buffer"].name) and (
                    "bias" in self._tensor_map.keys()):
                continue
            tmp_read_map = []
            for nop in lop["next_op"]:
                tmp_read_map.append(nop["dst_buffer"])
            tmp_cache_buffer = self._schedule.cache_read(
                lop["dst_buffer"], tbe_platform_info.scope_ubuf,
                list(set(tmp_read_map)))
            lop["cache_buffer"] = tmp_cache_buffer

        return True

    def _tiling_l0a_l0b(self, partial_ab, full_c, instr):
        """
        reduce factor.

        Parameters
        ----------
        partial_ab : tiling["AL0_matrix"] or tiling["BL0_matrix"]

        full_c : tiling["CL0_matrix"]

        instr: "A" or "B"

        Returns
        -------
        axis_factor, reduce factor
        """
        reduce_dim = [
            self._dim_map["fmap_matrix_dim"][-3],
            self._dim_map["fmap_matrix_dim"][-1]
        ]

        if instr == 'A':
            full_ab = [full_c[-3], reduce_dim[-2], full_c[-2], reduce_dim[-1]]
        elif instr == 'B':
            full_ab = [reduce_dim[-2], full_c[-4], full_c[-1], reduce_dim[-1]]

        partial_ab = list(partial_ab) if partial_ab else full_ab
        i_axis = 0
        for i_axis in range(len(partial_ab))[::-1]:
            if partial_ab[i_axis] != full_ab[i_axis]:
                break

        axis_factor = {}
        reduce_factor = {}
        red_axis = 0

        if instr == 'A':
            axis_map_a2c = {0: 1, 2: 2}
            axis_factor = {axis_map_a2c[0]: full_ab[0]}
            reduce_factor[0] = full_ab[1]
            for i in range(i_axis + 1):
                if i in [0, 2]:
                    axis_factor[axis_map_a2c[i]] = partial_ab[i]
                else:
                    reduce_factor[red_axis] = partial_ab[i]
                    red_axis += 1
        elif instr == 'B':
            axis_map_b2c = {1: 0, 2: 3}
            axis_factor = {axis_map_b2c[1]: full_ab[1]}
            reduce_factor[0] = full_ab[0]
            for i in range(i_axis + 1):
                reduce_factor[red_axis] = partial_ab[i]
                red_axis += 1
        axis_factor_for_batch = {}
        for i in axis_factor:
            axis_factor_for_batch[i + 1] = axis_factor[i]

        return {
            "axis_factor": axis_factor_for_batch,
            "reduce_factor": reduce_factor
        }

    def _check_tiling(self, tiling, w_dtype):
        """
        default tiling check

        Returns
        -------
        true for auto tiling, false for default tiling
        """
        if tiling["AL0_matrix"][2] == _VALID_TILING_NUM:
            return False

        l1_shape = ["AL1_shape", "BL1_shape"]
        for shape in l1_shape:
            if tiling[shape] and len(tiling[shape]) != _TILING_L1_SHAPE_DIM:
                cube_err.raise_err_three_paras('E62304', 'conv3d', shape,
                        str(_TILING_L1_SHAPE_DIM), str(len(tiling[shape])))

        matrix_list = [
            "AL0_matrix", "BL0_matrix", "CL0_matrix", "CUB_matrix", "block_dim"
        ]
        matrix_dim = [
            _TILING_AL0_MATRIX_DIM, _TILING_BL0_MATRIX_DIM,
            _TILING_CL0_MATRIX_DIM, _TILING_CUB_MATRIX_DIM, _TILING_BLOCK_DIM_DIM
        ]
        for matrix, dim in zip(matrix_list, matrix_dim):
            if len(tiling[matrix]) not in dim:
                cube_err.raise_err_three_paras('E62304', 'conv3d', matrix,
                        str(dim[0]), str(len(tiling[matrix])))

        matrix_cab = ["CL0_matrix", "AL0_matrix", "BL0_matrix"]
        for index0, index1 in zip(matrix_list[0:3], matrix_cab):
            if compute_util.get_and_res(tiling[index0], tiling[index1]):
                if tiling[index0][0] != tiling[index1][1]:
                    cube_err.raise_err_specific('conv3d',
                        "tiling['%s'][0] must equal to tiling['%s'][1]" % (index0, index1))

        if w_dtype != "float16":
            dict_args = {
                'errCode': 'E60005',
                'param_name': 'weight',
                'expected_dtype_list': 'float16',
                'dtype': w_dtype
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        for matrix in matrix_list[0:3]:
            if tiling[matrix] != []:
                if tiling[matrix][2] != _TILING_FLOAT16_MKN:
                    dict_args = {
                        'errCode': 'E62305',
                        'param_name': matrix,
                        'expect_value': str(_TILING_FLOAT16_MKN),
                        'value': str(tiling[matrix][2])
                    }
                    raise RuntimeError(dict_args,
                                       error_manager_util.get_error_message(dict_args))

                if tiling[matrix][3] != _TILING_FLOAT16_MKN:
                    dict_args = {
                        'errCode': 'E62305',
                        'param_name': matrix,
                        'expect_value': str(_TILING_FLOAT16_MKN),
                        'value': str(tiling[matrix][3])
                    }
                    raise RuntimeError(dict_args,
                                       error_manager_util.get_error_message(dict_args))

        return True

    def _tiling_fetch(self):
        """
        get tiling.

        Returns
        -------
        tiling
        """
        fmap_shape_ndc1hwc0 = conv3d_compute.Conv3DParam.tiling_info_dict["a_shape"]
        shape_w_ndc1hwc0 = conv3d_compute.Conv3DParam.tiling_info_dict["b_shape"]
        in_dtype = conv3d_compute.Conv3DParam.tiling_info_dict["a_dtype"]
        w_dtype = conv3d_compute.Conv3DParam.tiling_info_dict["b_dtype"]
        res_dtype = conv3d_compute.Conv3DParam.tiling_info_dict["c_dtype"]
        mad_dtype = conv3d_compute.Conv3DParam.tiling_info_dict["mad_dtype"]
        padd = conv3d_compute.Conv3DParam.tiling_info_dict["pad"][0:2]
        padh = conv3d_compute.Conv3DParam.tiling_info_dict["pad"][2:4]
        padw = conv3d_compute.Conv3DParam.tiling_info_dict["pad"][4:6]
        strided = conv3d_compute.Conv3DParam.tiling_info_dict["stride"][0]
        strideh = conv3d_compute.Conv3DParam.tiling_info_dict["stride"][1]
        stridew = conv3d_compute.Conv3DParam.tiling_info_dict["stride"][2]
        dilationd = conv3d_compute.Conv3DParam.tiling_info_dict["dilation"][0]
        dilationh = conv3d_compute.Conv3DParam.tiling_info_dict["dilation"][1]
        dilationw = conv3d_compute.Conv3DParam.tiling_info_dict["dilation"][2]
        bias_flag = conv3d_compute.Conv3DParam.tiling_info_dict["bias_flag"]
        batch_size = fmap_shape_ndc1hwc0[0]
        in_size_w = fmap_shape_ndc1hwc0[-2]
        kernel_d = shape_w_ndc1hwc0[1]
        kernel_h = shape_w_ndc1hwc0[-3]
        kernel_w = shape_w_ndc1hwc0[-2]

        if self.var_map:
            tiling_new = self.tiling_case
        else:
            info_dict = {
                "op_type": "convolution_3d",
                "a_shape": fmap_shape_ndc1hwc0,
                "b_shape": shape_w_ndc1hwc0,
                "a_dtype": in_dtype,
                "b_dtype": w_dtype,
                "c_dtype": res_dtype,
                "mad_dtype": mad_dtype,
                "pad": [padd[0], padd[1], padh[0], padh[1], padw[0], padw[1]],
                "stride": [strided, strideh, stridew],
                "dilation": [1, dilationh, dilationw],
                "bias_flag": bias_flag,
                "fused_coefficient": [0, 0, self._fused_op_num],
                "group": self._tensor_map["group_dict"]["real_g"],
                "kernel_name": self._tensor_map["kernel_name"],
            }
            tiling_new = get_tiling(info_dict)

        l0a_load2d_flag = self._tensor_map["l0a_load2d_flag"]
        self._schedule.set_var_range(self._tensor_map["d_dim"],
                                    int(tiling_new["block_dim"][-1]),
                                    int(tiling_new["block_dim"][-1]))
        cyclebuffer_flag = self._get_cyclebuffer_flag(tiling_new, shape_w_ndc1hwc0,
                                                        w_dtype, fmap_shape_ndc1hwc0,
                                                        strided, padd, l0a_load2d_flag,
                                                        dilationh, dilationw)

        tiling_ok_flag = self._check_tiling(tiling_new, w_dtype)

        tiling = {}
        tiling["AL0_matrix"] = tiling_new["AL0_matrix"][0:4]
        tiling["AL0_matrix"][
            1] = tiling["AL0_matrix"][1] * tiling_new["AL0_matrix"][-1]
        tiling["CL0_matrix"] = tiling_new["CL0_matrix"][0:4]
        tiling["CUB_matrix"] = tiling_new["CUB_matrix"][0:4]
        tiling["A_overhead_opt_flag"] = tiling_new["A_overhead_opt_flag"]
        tiling["B_overhead_opt_flag"] = tiling_new["B_overhead_opt_flag"]

        tiling["BL0_matrix"] = []
        if tiling_new["BL0_matrix"]:
            tiling["BL0_matrix"] = tiling_new["BL0_matrix"][0:4]
            tiling["BL0_matrix"][
                1] = tiling["BL0_matrix"][1] * tiling_new["BL0_matrix"][-1]

        tiling["manual_pingpong_buffer"] = tiling_new["manual_pingpong_buffer"]
        tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]

        tiling["AL1_shape"] = []
        if tiling_new["AL1_shape"]:
            tiling["AL1_shape"] = tiling_new["AL1_shape"][0:2]
            tiling["AL1_shape"][0] = int(
                tiling["AL1_shape"][0] * tiling_new["AL1_shape"][-1] /
                (((kernel_h - 1) * dilationh + 1) * ((kernel_w - 1) * dilationw + 1) *
                tbe_platform.CUBE_MKN[w_dtype]['mac'][1]))

        if compute_util.get_or_res(tiling_new["BL1_shape"] == [],
                              tiling_new["BL1_shape"] is None):
            tiling["BL1_shape"] = tiling_new["BL1_shape"]
        else:
            tiling["BL1_shape"] = tiling_new["BL1_shape"][0:2]
            tiling["BL1_shape"][0] = int(
                tiling["BL1_shape"][0] * tiling_new["BL1_shape"][-1] /
                (kernel_h * kernel_w * tbe_platform.CUBE_MKN[w_dtype]['mac'][1]))

        tiling["block_dim"] = tiling_new["block_dim"]
        tiling["block_dim"][0] = tiling["block_dim"][0] * tiling["block_dim"][-1]
        tiling["scale_drq_split_flag"] = False
        tiling["bias_split_flag"] = False

        if compute_util.get_or_res(tiling_ok_flag is False,
                              conv3d_compute.Conv3DParam.tiling_info_dict["default_tiling"]):
            tiling = {}
            config = tbe_platform.CUBE_MKN[w_dtype]
            ci0 = config['mac'][1]
            l1_buffer_size = tbe_platform_info.get_soc_spec("L1_SIZE")
            m_bit_length = {
                "float32": 32,
                "float16": 16,
                "uint8": 8,
                "int8": 8,
                "uint4": 4,
                "int4": 4
            }
            m_bit_ratio = {
                "int32": 4,
                "float32": 4,
                "float16": 2,
                "uint8": 1,
                "int8": 1,
                "uint4": 1.0 / 2,
                "int4": 1.0 / 2
            }
            input_data_type = in_dtype
            w_out = (in_size_w + padw[0] + padw[1] - ((kernel_w - 1) * dilationw + 1)) // stridew + 1
            if "fmap_w" in self.var_map:
                w_out = self.var_range.get('w_out')[1]
                in_size_w = self.var_range.get('fmap_w')[1]

            for m_target in range(32, 0, -1):
                tmp1 = (
                    (m_target * m_bit_length['float16']) + w_out - 1) // w_out
                tmp2 = ((tmp1 * strideh) + (kernel_h - 1) * dilationh + 1) * in_size_w
                max_feature_map = 1 * ci0 * tmp2 * 2 * m_bit_ratio[input_data_type]
                if max_feature_map < l1_buffer_size:
                    break

            tiling_m = m_target
            tiling_k = 1
            tiling_n = 1
            tiling["AL1_shape"] = [1]
            tiling["BL1_shape"] = None
            tiling["AL0_matrix"] = [tiling_m, tiling_k, 16, 16]
            tiling["BL0_matrix"] = [tiling_k, tiling_n, 16, 16]
            tiling["CL0_matrix"] = [tiling_n, tiling_m, 16, 16]
            tiling["CUB_matrix"] = [tiling_n, tiling_m, 16, 16]
            tiling["manual_pingpong_buffer"] = {
                'AL1_pbuffer': 1,
                'BL1_pbuffer': 1,
                'AL0_pbuffer': 1,
                'BL0_pbuffer': 1,
                'CL0_pbuffer': 1,
                'CUB_pbuffer': 1,
                'UBG_pbuffer': 1,
            }
            tiling["block_dim"] = [1, 1, 1]
            device_core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            if "batch_n" in self.var_map:
                tiling["block_dim"][0] = device_core_num
            elif compute_util.get_and_res(batch_size > 1, device_core_num > 1):
                if batch_size <= device_core_num:
                    tiling["block_dim"][0] = batch_size
                else:
                    for i in range(device_core_num, 0, -1):
                        if batch_size % i == 0:
                            break
                    tiling["block_dim"][0] = i
            else:
                tiling["block_dim"][0] = 1
            tiling["scale_drq_split_flag"] = True
            tiling["bias_split_flag"] = True
            tiling["A_overhead_opt_flag"] = 0
            tiling["B_overhead_opt_flag"] = 0
            tiling["n_bef_batch_flag"] = 0

        def _g_dim_tiling():
            # g_dim
            if (tiling_new["BUB_shape"] is None
                    or tiling_new["BUB_shape"][0] is None
                    or tiling_new["BUB_shape"][0] == 0):
                tiling["g_dim"] = 1
            else:
                tiling["g_dim"] = tiling_new["BUB_shape"][0]

        _g_dim_tiling()

        return tiling, cyclebuffer_flag

    def _double_buffer(self, buffer_dict, double_buffer_flag):
        """
        double buffer.

        Parameters
        ----------
        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        double_buffer_flag : flag for double buffer

        Returns
        -------

        """
        sch = self._schedule
        cyclebuffer_flag = self._tensor_map["cyclebuffer_flag"]
        # al1
        if double_buffer_flag["AL1_pbuffer"] == 2:
            if cyclebuffer_flag and not self.var_map:
                # cycle_double_buffer is not used in the dynamic shape
                sch[buffer_dict["al1"]].cycle_double_buffer()
            else:
                sch[buffer_dict["al1"]].double_buffer()
        # bl1
        if double_buffer_flag["BL1_pbuffer"] == 2:
            sch[buffer_dict["bl1"]].double_buffer()
        # l0a
        if double_buffer_flag["AL0_pbuffer"] == 2:
            sch[buffer_dict["fmap_col"]].double_buffer()
        # l0b
        if double_buffer_flag["BL0_pbuffer"] == 2:
            sch[buffer_dict["bl0"]].double_buffer()
        # L0C
        if double_buffer_flag["CL0_pbuffer"] == 2:
            sch[buffer_dict["c_col"]].double_buffer()
        # CUB
        if double_buffer_flag["CUB_pbuffer"] == 2:
            sch[buffer_dict["c_ub"]].double_buffer()

    def _condition_cycle_buffer_dynamic(self, cyclebuffer_flag,
                                        al1, c_col, c_ub, tiling, cin1_g):
        sch = self._schedule
        d_out = self._tensor_map["d_out"]
        pad_head = c_col.op.attrs['pad_head']
        stride_d = c_col.op.attrs['stride_d']
        kernel_d = c_ub.op.attrs['kernel_d']

        if cyclebuffer_flag:
            # set_store_predicate for cycle buffer
            _, dc_index = sch[al1].split(al1.op.axis[2], nparts=1)
            _, n_index = sch[al1].split(al1.op.axis[1], nparts=1)
            d_index = n_index % d_out * stride_d + (dc_index // cin1_g +
                        n_index % d_out * (kernel_d - stride_d)) % kernel_d - pad_head
            # condition_update is a refers to the conditions that
            # need to be updated during this load and the last load
            condition_update = d_index + pad_head > (n_index % d_out - 1) * \
                                stride_d + kernel_d - 1
            cyclebuffer_factor = self._int_ceil_div_tvm(self._tensor_map.get("d_out"),
                                                        self._tensor_map.get("d_dim"))
            db_expr = tvm.select(tvm.convert(cyclebuffer_factor) == 1,
                                    0,
                                    tvm.floormod(n_index,
                                                 cyclebuffer_factor))
            # 1: Full load is required for the first load in a single core
            # 2: In the case of db, both ping and pong must be full loaded for the first time
            if tiling["manual_pingpong_buffer"]['AL1_pbuffer'] == 1:
                condition_db = (db_expr == 0).asnode()
            else:
                condition_db = tvm.any((db_expr == 0).asnode(),
                                        (db_expr == (cyclebuffer_factor + 1) // 2).asnode())
            condition_cycle = tvm.any(condition_update, condition_db)
            sch[al1].set_store_predicate(condition_cycle)

    def _intrin_mapping(self, fmap, mad_dict, buffer_dict, new_fmap_col_axis,
                        tiling, cn_axis, l0a_load2d_flag):
        """
        intrin_mapping.

        Parameters
        ----------
        famp : input tensor

        mad_dict : for mad pragma

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        new_fmap_col_axis : fmap_col axis

        tiling : case tiling

        cn_axis : loc axis

        l0a_load2d_flag : true or false

        Returns
        -------

        """
        sch = self._schedule
        al1 = buffer_dict["al1"]
        bl1 = buffer_dict["bl1"]
        fmap_col = buffer_dict["fmap_col"]
        bl0 = buffer_dict["bl0"]
        c_col = buffer_dict["c_col"]
        c_ub = buffer_dict["c_ub"]

        cin1_g = self._tensor_map["group_dict"]["cin1_g"]
        setfmatrix_dict = {
            "conv_kernel_h": c_ub.op.attrs['kernel_h'],
            "conv_kernel_w": c_ub.op.attrs['kernel_w'],
            "conv_padding_top": c_ub.op.attrs['padding'][0],
            "conv_padding_bottom": c_ub.op.attrs['padding'][1],
            "conv_padding_left": c_ub.op.attrs['padding'][2],
            "conv_padding_right": c_ub.op.attrs['padding'][3],
            "conv_stride_h": c_ub.op.attrs['stride'][0],
            "conv_stride_w": c_ub.op.attrs['stride'][1],
            "conv_dilation_h": c_ub.op.attrs['dilation'][0],
            "conv_dilation_w":  c_ub.op.attrs['dilation'][1],
            "conv_fm_c": cin1_g * fmap.op.shape[5],
            "conv_fm_h": fmap.op.shape[3],
            "conv_fm_w": fmap.op.shape[4]
        }

        stride_d = c_col.op.attrs['stride_d']
        cyclebuffer_flag = self._tensor_map["cyclebuffer_flag"]
        kernel_d = c_ub.op.attrs['kernel_d']
        if cyclebuffer_flag and not self.var_map:
            # Specifies AL1 memory
            sch[al1].mem_unique()
            sch[al1].emit_insn(al1.op.axis[1], 'dma_copy')
        elif self.var_map and not l0a_load2d_flag:
            self._condition_cycle_buffer_dynamic(cyclebuffer_flag, al1, c_col,
                                                 c_ub, tiling, cin1_g)
        elif self.var_map and l0a_load2d_flag:
            sch[al1].emit_insn(al1.op.axis[1], 'dma_copy')
        else:
            sch[al1].emit_insn(al1.op.axis[0], 'dma_copy')

        if l0a_load2d_flag:
            sch[fmap_col].emit_insn(new_fmap_col_axis[3], 'dma_copy')
        elif self.var_map:
            stride_update = 1 if self._tensor_map["opti_h_flag"] else c_ub.op.attrs['stride'][0]
            im2col_attr = {
                'set_fmatrix': 1,
                'conv_kernel_d': kernel_d,
                'conv_kernel_h': c_ub.op.attrs['kernel_h'],
                'conv_kernel_w': c_ub.op.attrs['kernel_w'],
                'conv_padding_top': c_ub.op.attrs['padding'][0],
                'conv_padding_bottom': c_ub.op.attrs['padding'][1],
                'conv_padding_left': c_ub.op.attrs['padding'][2],
                'conv_padding_right': c_ub.op.attrs['padding'][3],
                'conv_stride_d': stride_d,
                'conv_stride_h': stride_update,
                'conv_stride_w': c_ub.op.attrs['stride'][1],
                'conv_fm_c': cin1_g * fmap.op.shape[5],
                'conv_fm_c1': cin1_g,
                'conv_fm_h': fmap.shape[3],
                'conv_fm_w': fmap.shape[4],
                'conv_fm_c0': fmap.shape[5],
                'group_flag': 1,
                'l1_group_flag': 1,
                'circular_buf': cyclebuffer_flag,
                'conv_batch': fmap_col.op.axis[1],
                'conv_intrin_batch': new_fmap_col_axis[-5],
                'conv_d_out': self._tensor_map.get("d_out"),
            }
            if self._tensor_map["opti_h_flag"]:
                im2col_attr["conv_stride_h"] = 1
            sch[fmap_col].emit_insn(new_fmap_col_axis[-5], 'im2col_v2', im2col_attr)
            sch[al1].emit_insn(al1.op.axis[3], 'dma_copy')
        else:
            if self._tensor_map["opti_h_flag"]:
                setfmatrix_dict["conv_stride_h"] = 1
            fmap_col_before = buffer_dict["fmap_col_before"]
            sch[fmap_col_before].emit_insn(fmap_col_before.op.axis[0],
                                           'set_fmatrix', setfmatrix_dict)
            sch[fmap_col].emit_insn(new_fmap_col_axis[-5], 'im2col')

        if tiling["BL1_shape"] is not None:
            sch[bl1].emit_insn(sch[bl1].op.axis[0], 'dma_copy')
        sch[bl0].emit_insn(bl0.op.axis[0], 'dma_copy')

        sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
        sch[c_col].emit_insn(cn_axis, 'mad', mad_dict)

    def _attach_at(self, bodyops, inputops, compute_at_buffer, compute_at_axis,
                   tiling):
        """
        tensor not for conv compute at.

        Parameters
        ----------
        bodyops : body dict

        inputops : input dict

        compute_at_buffer : col res_c

        compute_at_axis : axis for compute at

        tiling : case tiling

        Returns
        -------

        """
        for lop in bodyops:
            if "conv3d" not in lop["op"] or "convolution_A" in lop["op"] or \
                (self.dsl_flag and (lop["op"] == "conv3d_C")):
                if lop["op"] == "conv_vector_remove_pad":
                    continue
                self._schedule[lop["dst_buffer"]].compute_at(
                    self._schedule[compute_at_buffer[1]],
                    compute_at_axis[1])
                self._schedule[lop["dst_buffer"]].buffer_align(
                    (1, 1), (1, 1),
                    (1, tiling["CL0_matrix"][1] * tiling["CL0_matrix"][2]),
                    (1, 1))

        for lop in inputops:
            if "conv3d" in lop["op"]:
                continue
            if ("bias_tensor" in lop["op"]) and (
                    "bias" in self._tensor_map.keys()):
                continue
            self._schedule[lop["cache_buffer"]].compute_at(
                self._schedule[compute_at_buffer[0]], compute_at_axis[1])

    def _to_pragma(self, bodyops, inputops, c_outer_inner_inner):
        """
        tensor not for conv to pragma.

        Parameters
        ----------
        bodyops : body dict

        inputops : input dict

        fmap : input tensor

        c_ub : conv res in ub

        c_outer_inner_inner : res axis

        Returns
        -------

        """
        for lop in bodyops:
            if "conv3d" not in lop["op"] or "conv3d_A" in lop["op"] or \
                (self.dsl_flag and (lop["op"] == "conv3d_C")):
                lop["tensorize_axis"] = self._schedule[
                    lop["dst_buffer"]].op.axis[0]
                if "Before" in lop["op"]:
                    lop["op"] = lop["op"].replace("_Before", "")
                if "_conv3d_A" in lop["op"]:
                    lop["op"] = lop["op"].replace("_conv3d_A", "")
                self.__pragma_for_op(lop, c_outer_inner_inner)

        for lop in inputops:
            if "conv3d" in lop["op"]:
                continue
            if ("bias_tensor" in lop["op"]) and (
                    "bias" in self._tensor_map.keys()):
                continue
            self._schedule[lop["cache_buffer"]].emit_insn(
                lop["cache_buffer"].op.axis[0], 'dma_copy')

    def _set_al1_at_axis(self, l0a_load2d_flag, nbuffer_flag_al1, reorder_flag,
                         tiling, al1_factor, compute_axis, run_once_axis,
                         allocate_axis, index_axis, buffer_dict, stage):
        """
        al1 compute_at.

        Parameters
        ----------
        l0a_load2d_flag : true or false

        nbuffer_flag_al1 : true or false

        reorder_flag : true or false

        tiling : case tiling

        al1_factor : al1 split factor [c1//kAl1, howo//mc//m0//m_Al1]

        compute_axis : al1 axis to compute at

        run_once_axis : al1 axis to run once

        allocate_axis : al1 axis to allocate at

        index_axis : al1 index to stage

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        stage : c_col res_c

        Returns
        -------

        """
        index = int(al1_factor[0] == 1) if tiling["AL1_shape"] else 2
        cyclebuffer_flag = self._tensor_map["cyclebuffer_flag"]
        sch = self._schedule
        al1_allocate_axis = None
        al1_run_once_axis = []
        compute_stage = None
        allocate_stage = None
        al1 = buffer_dict["al1"]
        if not l0a_load2d_flag and not self.var_map:
            fmap_col_before = buffer_dict["fmap_col_before"]
        if compute_util.get_and_res(l0a_load2d_flag, nbuffer_flag_al1):
            al1_compute_axis = compute_axis["k_outer_outer_inner_outer"]
            compute_stage = stage[0]
            al1_allocate_axis = allocate_axis[index_axis[index]]
            allocate_stage = stage[index]
            run_flag = compute_util.get_and_res(index == 1, reorder_flag)
            if compute_util.get_or_res(run_flag, index == 2):
                al1_run_once_axis = [
                    run_once_axis["c_outer_outer_inner"],
                    run_once_axis["c_outer_outer_outer_inner"]
                ]
        elif nbuffer_flag_al1:
            if index == 0:
                al1_compute_axis = compute_axis["k_outer_outer_inner_outer"]
                compute_stage = stage[0]
                al1_allocate_axis = allocate_axis[index_axis[0]]
                allocate_stage = stage[0]
            else:
                al1_compute_axis = compute_axis[index_axis[index]]
                compute_stage = stage[index]
        else:
            al1_compute_axis = compute_axis[index_axis[index]]
            compute_stage = stage[index]

        if l0a_load2d_flag:
            sch[al1].compute_at(sch[compute_stage], al1_compute_axis)
        else:
            sch[al1].compute_at(sch[compute_stage], al1_compute_axis)
            if not self.var_map:
                sch[fmap_col_before].compute_at(sch[compute_stage],
                                                al1_compute_axis)

        if not self.var_map:
            do_num = al1.op.axis[0].dom.extent.value
            if compute_util.get_and_res(cyclebuffer_flag, do_num != 1):
                cyclebuffer_factor = self._tensor_map["d_out"] // self._tensor_map["d_dim"]
                expr = tvm.select(tvm.convert(cyclebuffer_factor) == 1,
                                  al1.op.axis[0].var,
                                  tvm.floormod(al1.op.axis[0].var,
                                               cyclebuffer_factor))
                sch[al1].pragma(al1.op.axis[0],
                                "cyclebuffer",
                                (expr == 0).asnode())

            if al1_run_once_axis:
                sch[al1].allocate_at(sch[allocate_stage],
                                     al1_allocate_axis,
                                     run_once_axes=al1_run_once_axis)
            elif al1_allocate_axis is not None:
                sch[al1].allocate_at(sch[allocate_stage], al1_allocate_axis)

        return True

    def _set_bl1_at_axis(self, reorder_flag, tiling, bl1_factor,
                         compute_bl1_axis, run_once_bl1_axis, allocate_bl1_axis,
                         bl1_index_dict, buffer_dict, stage):
        """
        bl1 compute_at.

        Parameters
        ----------
        reorder_flag : true or false

        tiling : case tiling

        bl1_factor : bl1 split factor [c1//kBl1, cout//nc//n0//m_Bl1]

        compute_bl1_axis : bl1 axis to compute at

        run_once_bl1_axis : bl1 axis to run once

        allocate_bl1_axis : bl1 axis to allocate at

        bl1_index_axis : bl1 index to stage

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        stage : c_col res_c

        Returns
        -------

        """
        index = 2 if (tiling["BL1_shape"] == []
                      or tiling["BL1_shape"] is None) else int(
                          bl1_factor[0] == 1)
        sch = self._schedule
        bl1_compute_axis = None
        bl1_allocate_axis = None
        bl1_run_once_axis = []
        bl1_compute_stage = None
        bl1_allocate_stage = None
        bl1 = buffer_dict["bl1"]
        if tiling["B_overhead_opt_flag"]:
            if index == 0 or (index == 1 and reorder_flag):
                bl1_compute_axis = compute_bl1_axis["coo"]
                bl1_compute_stage = stage[0]
                bl1_allocate_axis = allocate_bl1_axis[bl1_index_dict[index]]
                bl1_allocate_stage = stage[index]
                if index == 1 and reorder_flag:
                    bl1_run_once_axis = [
                        run_once_bl1_axis["m_outer_outer_inner"]
                    ]
            else:
                bl1_compute_axis = compute_bl1_axis[bl1_index_dict[index]]
                bl1_compute_stage = stage[index]
        else:
            bl1_compute_axis = compute_bl1_axis[bl1_index_dict[index]]
            bl1_compute_stage = stage[index]

        sch[bl1].compute_at(sch[bl1_compute_stage], bl1_compute_axis)
        if bl1_run_once_axis:
            sch[bl1].allocate_at(sch[bl1_allocate_stage],
                                 bl1_allocate_axis,
                                 run_once_axes=bl1_run_once_axis)
        elif bl1_allocate_axis is not None:
            sch[bl1].allocate_at(sch[bl1_allocate_stage], bl1_allocate_axis)

        return True

    def do_schedule(self, res, spec_node_list, sch_list, dynamic_para=None):
        """
        auto_schedule for cce AI-CORE.
        For now, only one convolution operation is supported.

        Parameters
        ----------
        res : tvm.tensor

        spec_node_list : same as other template in cce_schedule

        sch_list: use sch_list[0] to return conv schedule

        Returns
        -------
        True for sucess, False for no schedule
        """
        opti_flag = False

        tensor_map = self._tensor_map
        dim_map = self._dim_map
        c_ub = tensor_map["c_ub"]

        sch = sch_list[0]
        self._schedule = sch

        color_op = AutoScheduleOp(res)
        self.body_ops = color_op.body_ops
        self.input_ops = color_op.input_ops
        self.output_ops = color_op.output_ops

        self._res_tensor = res
        res_c = self._res_tensor
        self._cachebuffer(spec_node_list)
        l0a_load2d_flag = tensor_map["l0a_load2d_flag"]
        bias_flag = conv3d_compute.Conv3DParam.tiling_info_dict["bias_flag"]

        def _get_fused_op_num():
            fuse_op_num = len(self.body_ops) - _CONV_NUM
            if not l0a_load2d_flag:
                fuse_op_num -= 1
            if bias_flag:
                fuse_op_num -= 1
            return fuse_op_num

        self._fused_op_num = _get_fused_op_num()
        if self.var_map:
            self.tiling_case = dynamic_para.get("tiling")
            self.var_range = dynamic_para.get("var_range")

        fmap = tensor_map["fmap"]
        weight = tensor_map["filter"]
        c_col = tensor_map["c_col"]
        stage_dict = {"res_c": res_c, "c_col": c_col}

        config = tbe_platform.CUBE_MKN[weight.dtype]

        pad_right = c_ub.op.attrs['padding'][2]
        pad_left = c_ub.op.attrs['padding'][3]
        kernel_w = c_ub.op.attrs['kernel_w']
        kernel_h = c_ub.op.attrs['kernel_h']
        kernel_d = c_ub.op.attrs['kernel_d']

        fmap_w = fmap.shape[-2] if fmap.op.input_tensors else fmap.op.shape[-2]
        fmap_cin1_ori = fmap.shape[2] if fmap.op.input_tensors else fmap.op.shape[2]
        stride_w = c_ub.op.attrs['stride'][1]
        dilation_w = c_ub.op.attrs['dilation'][1]

        if "fmap_d" in self.var_map:
            sch.set_var_range(self.var_map.get("fmap_d"), *self.var_range.get('fmap_d'))
            sch.set_var_range(self.var_map.get("d_out"), *self.var_range.get('d_out'))
        if "fmap_h" in self.var_map:
            sch.set_var_range(self.var_map.get("fmap_h"), *self.var_range.get('fmap_h'))
            sch.set_var_range(self.var_map.get("h_out"), *self.var_range.get('h_out'))
        if "fmap_w" in self.var_map:
            sch.set_var_range(self.var_map.get("fmap_w"), *self.var_range.get('fmap_w'))
            sch.set_var_range(self.var_map.get("w_out"), *self.var_range.get('w_out'))
        if "batch_n" in self.var_map:
            sch.set_var_range(self.var_map.get("batch_n"), *self.var_range.get('batch_n'))

        if "fmap_w" in self.var_map:
            w_out = self.var_map.get("w_out")
        else:
            w_out = (fmap_w + pad_left + pad_right - ((kernel_w - 1) * dilation_w + 1)) // stride_w + 1

        def _load2d_process():
            if l0a_load2d_flag:
                _al1 = tensor_map["al1_load2d"]
                al0 = tensor_map["al0_load2d"]
                sch[_al1].storage_align(sch[_al1].op.axis[1], 256, 0)
                _fmap_col = al0
                sch[_al1].set_scope(tbe_platform_info.scope_cbuf)
                _fuse_fmap_tensor = 0
                _fmap_col_before = 0
            else:
                _fuse_fmap_tensor = tensor_map["fmap_do_tensor"]
                _fmap_col_before = 0
                if not self.var_map:
                    _fmap_col_before = tensor_map["fmap_im2col_row_major_res"]
                    sch[_fmap_col_before].buffer_align(
                        (1, 1), (w_out, w_out), (1, 1), (1, 1), (1, 1),
                        (1, tbe_platform.CUBE_MKN[_fmap_col_before.dtype]["mac"][1]))
                    sch[_fmap_col_before].set_scope(tbe_platform_info.scope_cbuf)
                _fmap_col = tensor_map["fmap_im2col_fractal_res"]
                sch[_fuse_fmap_tensor].set_scope(tbe_platform_info.scope_cbuf)
                _al1 = _fuse_fmap_tensor
            return _fuse_fmap_tensor, _fmap_col_before, _fmap_col, _al1

        fuse_fmap_tensor, fmap_col_before, fmap_col, al1 = _load2d_process()

        if 'bias' in tensor_map.keys() and opti_flag:
            bias = tensor_map["bias"]
            bias_l0c = tensor_map["bias_l0c"]
            c_col_bias = tensor_map["c_col_bias"]
            bias_ub_brc = tensor_map["bias_ub_brc"]

            sch[bias_l0c].set_scope(tbe_platform_info.scope_cc)
            sch[c_col_bias].set_scope(tbe_platform_info.scope_cc)
            sch[bias_ub_brc].set_scope(tbe_platform_info.scope_ubuf)
            bias_ub = sch.cache_read(bias, tbe_platform_info.scope_ubuf,
                                     [bias_ub_brc])

        sch[c_ub].buffer_align((1, 1), (1, 1),
                               (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][0]),
                               (1, tbe_platform.CUBE_MKN[c_ub.dtype]["mac"][2]))
        # for fusion vector
        if self.dsl_flag:
            res_ub = sch.cache_write(res, tbe_platform_info.scope_ubuf)
            self.output_ops[0]["tensorize_axis"] = \
                self._schedule[res_ub].op.axis[0]
            self.output_ops[0]["dst_buffer"] = res_ub

        tiling, cyclebuffer_flag = self._tiling_fetch()

        self._tensor_map["cyclebuffer_flag"] = cyclebuffer_flag
        cyclebuffer = tensor_map["cycle_flag_info"]
        specified_bound = int(cyclebuffer_flag)
        sch.set_var_range(cyclebuffer, specified_bound, specified_bound)

        filter_matrix = list(dim_map["filter_matrix_dim"])
        filter_matrix[1] = filter_matrix[1] // tiling["block_dim"][1]

        bl1 = self._weight_to_bl1(tiling, filter_matrix, weight, c_col)
        bl0 = sch.cache_read(bl1, tbe_platform_info.scope_cb, [c_col])

        sch[c_col].set_scope(tbe_platform_info.scope_cc)
        sch[c_ub].set_scope(tbe_platform_info.scope_ubuf)

        compute_at_buffer = []
        compute_at_axis = []

        sch[fmap_col].set_scope(tbe_platform_info.scope_ca)

        factor_m = tiling["AL0_matrix"][0]
        factor_k = tiling["AL0_matrix"][1]

        # split N begin

        a1_axis, a3_axis = sch[fmap_col].split(sch[fmap_col].op.axis[2],
                                               factor_m)
        a2_axis, a4_axis = sch[fmap_col].split(sch[fmap_col].op.axis[3],
                                               factor_k)

        fmap_col_no, fmap_col_ni = sch[fmap_col].split(
            sch[fmap_col].op.axis[1], 1)
        sch[fmap_col].reorder(fmap_col_no, a1_axis, a2_axis, fmap_col_ni,
                              a3_axis, a4_axis, sch[fmap_col].op.axis[4],
                              sch[fmap_col].op.axis[5])
        new_fmap_col_axis = [
            fmap_col_no, a1_axis, a2_axis, fmap_col_ni, a3_axis, a4_axis,
            sch[fmap_col].op.axis[4], sch[fmap_col].op.axis[5]
        ]

        new_c_col_axis = [
            sch[c_col].op.axis[1], sch[c_col].op.axis[2],
            sch[c_col].op.axis[3], sch[c_col].op.axis[4]
        ]


        _, _, _, nn_axis = new_c_col_axis

        c_tiling_factor = [
            tiling["CL0_matrix"][0],
            tiling["CL0_matrix"][1] * tiling["CL0_matrix"][2]
        ]

        n_0 = config["mac"][2]
        c_factor = [
            compute_util.int_ceil_div(self._tensor_map["group_dict"]["cout_g"] // n_0,
                                 c_tiling_factor[0]),
            compute_util.int_ceil_div(dim_map["out_img_shape"][-2],
                                 c_tiling_factor[1])
        ]

        c_ub_tiling_factor = tiling["CUB_matrix"]
        c_ub_factor = [
            compute_util.int_ceil_div(c_tiling_factor[0], c_ub_tiling_factor[0]),
            compute_util.int_ceil_div(c_tiling_factor[1],
                                 c_ub_tiling_factor[1] * c_ub_tiling_factor[2])
        ]

        al1_factor, bl1_factor = self._factor_al1_bl1(tiling, c_factor)

        al0_axis_factor = self._tiling_l0a_l0b(tiling["AL0_matrix"],
                                               tiling["CL0_matrix"], 'A')

        bl0_axis_factor = self._tiling_l0a_l0b(tiling["BL0_matrix"],
                                               tiling["CL0_matrix"], 'B')

        # --------------------------double buffer------------------------
        double_buffer_flag = {
            'AL1_pbuffer': False,
            'BL1_pbuffer': False,
            'AL0_pbuffer': False,
            'BL0_pbuffer': False,
            'CL0_pbuffer': False,
            'CUB_pbuffer': False,
            'UBG_pbuffer': False,
        }

        double_buffer_flag = tiling["manual_pingpong_buffer"]

        # --------------------------tile res_c------------------------
        # to split G
        group_dict = tensor_map["group_dict"]
        cout1_g = group_dict["cout_g"] // n_0
        c_outer_g, c_outer = sch[res_c].split(res_c.op.axis[1], cout1_g)
        c_outer_outer, c_outer_inner = sch[res_c].split(c_outer, c_tiling_factor[0])
        c_outer_g_outer, c_outer_g_inner = sch[res_c].split(c_outer_g, nparts=tiling["g_dim"])

        m_outer_outer, m_outer_inner = sch[res_c].split(res_c.op.axis[2], c_tiling_factor[1])
        sch[res_c].reorder(c_outer_outer, m_outer_outer,
                           c_outer_inner, m_outer_inner)

        m_outer_outer_outer, m_outer_outer_inner = sch[res_c].split(
            m_outer_outer, nparts=al1_factor[1])
        c_outer_outer_outer, c_outer_outer_inner = sch[res_c].split(
            c_outer_outer, nparts=bl1_factor[1])
        c_slice_axis = m_outer_outer_inner

        block_dim = tiling["block_dim"] if "block_dim" in tiling else [1, 1, 1]

        # split batch of res_c
        batch_outer, batch_inner = sch[res_c].split(res_c.op.axis[0],
                                                    nparts=int(block_dim[0]))
        # split cout of res_c
        c_outer_outer_outer_outer, c_outer_outer_outer_inner = sch[
            res_c].split(c_outer_outer_outer, nparts=block_dim[1])
        bl1_at_c_axis = c_outer_outer_outer_inner

        m_outer_outer_outer_size = al1_factor[1]
        block_dim[2] = tvm.min(block_dim[2], m_outer_outer_outer_size)
        m_outer_outer_outer_outer, m_outer_outer_outer_inner = sch[
            res_c].split(m_outer_outer_outer, nparts=block_dim[2])
        al1_at_c_axis = m_outer_outer_outer_inner

        if tensor_map["cyclebuffer_flag"]:
            split_condition = self.var_map and \
                              tiling["manual_pingpong_buffer"]['AL1_pbuffer'] == 2 and \
                              tiling["manual_pingpong_buffer"]['AL0_pbuffer'] == 2
            if split_condition:
                # split + reorder to adjust ping pong data division,
                # instead of cycle double buffer
                batch_inner_outer, batch_inner_inner = sch[res_c].split(
                    batch_inner, nparts=1)
                batch_inner_inner_outer, batch_inner_inner_inner = sch[res_c].split(
                    batch_inner_inner, nparts=2)
                al1_at_c_axis = batch_inner_inner_outer
                sch[res_c].reorder(batch_outer, c_outer_g_outer,
                                   c_outer_outer_outer_outer,
                                   m_outer_outer_outer_outer,
                                   c_outer_g_inner,
                                   batch_inner_outer,
                                   c_outer_outer_outer_inner,
                                   m_outer_outer_outer_inner,
                                   batch_inner_inner_inner,
                                   batch_inner_inner_outer)
                cycbuf_axis = batch_inner_outer
            else:
                batch_inner_outer, batch_inner_inner = sch[res_c].split(
                    batch_inner, nparts=1)
                al1_at_c_axis = batch_inner_inner
                sch[res_c].reorder(batch_outer, c_outer_g_outer,
                                   c_outer_outer_outer_outer,
                                   m_outer_outer_outer_outer,
                                   c_outer_g_inner,
                                   batch_inner_outer,
                                   c_outer_outer_outer_inner,
                                   m_outer_outer_outer_inner, batch_inner_inner)
                cycbuf_axis = batch_inner_outer
        else:
            sch[res_c].reorder(batch_outer, c_outer_g_outer,
                               c_outer_outer_outer_outer,
                               m_outer_outer_outer_outer,
                               c_outer_g_inner,
                               batch_inner,
                               c_outer_outer_outer_inner,
                               m_outer_outer_outer_inner)
            cycbuf_axis = batch_inner

        mc_flag = False
        blocks = block_dim[0] * block_dim[1] * block_dim[2] * tiling["g_dim"]
        noo_true = cycbuf_axis
        block = 1
        if blocks != 1:
            batch_cout_fused = sch[res_c].fuse(batch_outer,
                                               c_outer_g_outer,
                                               c_outer_outer_outer_outer,
                                               m_outer_outer_outer_outer)
            noo_true, _ = sch[res_c].split(batch_cout_fused, nparts=blocks)
            bido, _ = sch[res_c].split(noo_true, 1)
            block = tvm.thread_axis("blockIdx.x")
            sch[res_c].bind(bido, block)
            mc_flag = True

        noi_tree = cycbuf_axis

        noo, noi = sch[res_c].split(noi_tree, factor=1)

        if not mc_flag:
            bido = noo

        reorder_axis_dict = {
            "m_outer_outer_outer_inner": m_outer_outer_outer_inner,
            "c_outer_outer_outer_inner": c_outer_outer_outer_inner,
            "noi": noi
        }

        reorder_flag = self._reorder_axis(tiling, al1_factor, bl1_factor,
                                          double_buffer_flag, reorder_axis_dict,
                                          res_c)
        m_outer_inner_outer, m_outer_inner_inner = sch[res_c].split(
            m_outer_inner, nparts=1)

        # ============ tile CUB ========================
        c_outer_inner_outer, c_outer_inner_inner = sch[res_c].split(
            c_outer_inner, nparts=c_ub_factor[0])

        sch[res_c].reorder(c_outer_inner_outer, m_outer_inner_outer,
                           c_outer_inner_inner, m_outer_inner_inner)
        sch[c_ub].compute_at(sch[res_c], m_outer_inner_outer)
        c_pragma_axis = c_outer_inner_inner

        # ============ tile c_col =======================
        compute_at_buffer.append(res_c)
        compute_at_axis.append(c_slice_axis)
        compute_at_buffer.append(res_c)
        compute_at_axis.append(m_outer_inner_outer)

        sch[c_col].compute_at(sch[res_c], c_slice_axis)

        def _bias_compute_at():
            if 'bias' in tensor_map.keys() and opti_flag:
                sch[bias_l0c].compute_at(sch[res_c], c_slice_axis)
                sch[c_col_bias].compute_at(sch[res_c], c_slice_axis)

        _bias_compute_at()

        _, reduce_kk = sch[c_col].op.reduce_axis

        axis_factor = list(al0_axis_factor["axis_factor"].items())
        # for now

        boo, boi = sch[c_col].split(new_c_col_axis[axis_factor[0][0]],
                                    axis_factor[0][1] * config["mac"][0])

        axis_factor = list(bl0_axis_factor["axis_factor"].items())
        coo, coi = sch[c_col].split(new_c_col_axis[axis_factor[0][0]],
                                    axis_factor[0][1])

        # for reduce axis, al0 and bl0 should be the same
        reduce_axis_factor = list(al0_axis_factor["reduce_factor"].items())

        # k_outer_outer should be no less than kd
        k_outer_outer, k_outer_inner = sch[c_col].split(
            c_col.op.reduce_axis[reduce_axis_factor[0][0]],
            reduce_axis_factor[0][1])
        k_outer_outer_size = c_col.op.reduce_axis[
                                 reduce_axis_factor[0][0]].dom.extent // \
                             reduce_axis_factor[0][1]

        # split N begin
        _, cn_axis = sch[c_col].split(c_col.op.axis[1], 1)

        sch[c_col].reorder(k_outer_outer, coo, boo, cn_axis, coi, boi, nn_axis,
                           k_outer_inner, reduce_kk)
        sch[fmap_col].compute_at(sch[c_col], boo)
        self._attach_bl0(tiling, stage_dict, bl0, coo, noo)

        #  ============ al1 and bl1 slice can be different with CUB & CL0 =====
        al1_at_ccol_axis, bl1_at_ccol_axis, k_axis_dict = self._al1_bl1_axis(
            stage_dict, al1_factor, bl1_factor, k_outer_outer)

        k_outer_outer_outer_outer = k_axis_dict["k_outer_outer_outer_outer"]
        k_outer_outer_outer_inner = k_axis_dict["k_outer_outer_outer_inner"]
        k_outer_outer_inner = k_axis_dict["k_outer_outer_inner"]

        buffer_dict = {
            "al1": al1,
            "bl1": bl1,
            "fmap_col": fmap_col,
            "bl0": bl0,
            "c_col": c_col,
            "c_ub": c_ub,
            "res_c": res_c
        }

        def _update_load2d_buffer_dict():
            if not l0a_load2d_flag:
                buffer_dict["fmap_col_before"] = fmap_col_before

        _update_load2d_buffer_dict()

        # al1 compute_at
        compute_al1_axis = {
            "al1_at_ccol_axis": al1_at_ccol_axis,
            "al1_at_c_axis": al1_at_c_axis,
            "noo": noo
        }
        shape_w = conv3d_compute.Conv3DParam.tiling_info_dict["b_shape"]
        k_outer_outer_inner_size = int(k_outer_outer_size //
                                       max(al1_factor[0], bl1_factor[0]))

        nbuffer_flag_al1, compute_al1_axis, _, k_outer_outer_inner_outer_size \
            = self._get_nbuffer_al1_flag(tiling, compute_al1_axis,
                                         buffer_dict, k_outer_outer_inner,
                                         k_outer_outer_inner_size, shape_w)
        run_once_al1_axis = {
            "c_outer_outer_inner": c_outer_outer_inner,
            "c_outer_outer_outer_inner": c_outer_outer_outer_inner
        }
        allocate_al1_axis = {
            "al1_at_ccol_axis": al1_at_ccol_axis,
            "al1_at_c_axis": al1_at_c_axis,
            "noo": noo
        }
        index_al1_dict = {0: "al1_at_ccol_axis", 1: "al1_at_c_axis", 2: "noo"}
        stage = {0: c_col, 1: res_c, 2: res_c}
        self._set_al1_at_axis(l0a_load2d_flag, nbuffer_flag_al1, reorder_flag,
                              tiling, al1_factor, compute_al1_axis,
                              run_once_al1_axis, allocate_al1_axis,
                              index_al1_dict, buffer_dict, stage)

        # bl1 compute_at
        compute_bl1_axis = {
            "coo": coo,
            "bl1_at_ccol_axis": bl1_at_ccol_axis,
            "bl1_at_c_axis": bl1_at_c_axis,
            "c_outer_g_inner": c_outer_g_inner
        }
        allocate_bl1_axis = {
            "bl1_at_ccol_axis": bl1_at_ccol_axis,
            "bl1_at_c_axis": bl1_at_c_axis,
            "c_outer_g_inner": c_outer_g_inner
        }
        run_once_bl1_axis = {"m_outer_outer_inner": m_outer_outer_inner}
        bl1_index_dict = {0: "bl1_at_ccol_axis", 1: "bl1_at_c_axis", 2: "c_outer_g_inner"}
        self._set_bl1_at_axis(reorder_flag, tiling, bl1_factor,
                              compute_bl1_axis, run_once_bl1_axis,
                              allocate_bl1_axis, bl1_index_dict, buffer_dict,
                              stage)

        ############################ double buffer ###########################
        self._double_buffer(buffer_dict, double_buffer_flag)
        ############################ intrin mapping ###########################
        stride_d = c_col.op.attrs['stride_d']
        pad_head = c_col.op.attrs['pad_head']
        fmap_d = c_col.op.attrs['fmap_d']
        d_out = c_col.op.attrs['d_out']
        w_h = shape_w[-3]
        w_w = shape_w[-2]

        def _get_batch_axis():
            batch_axis = tvm.floordiv(block, block_dim[1] * block_dim[2] * tiling["g_dim"]) * (
                    (self._get_value(fmap_col.shape[1]) + block_dim[0] - 1) //
                    block_dim[0]) + noo
            if tensor_map["cyclebuffer_flag"]:
                batch_axis = tvm.floordiv(block, block_dim[1] * block_dim[2] * tiling["g_dim"]) * \
                             ((self._get_value(fmap_col.shape[1]) + block_dim[0] - 1) //
                              block_dim[0]) + noo + batch_inner_inner
            return batch_axis

        batch_axis = _get_batch_axis()

        outer_factor = max(al1_factor[0], bl1_factor[0])
        inner_factor = min(al1_factor[0], bl1_factor[0])

        x_factor = group_dict["cin1_g"]

        mad_dict = {
            "mad_pattern":
                2,
            "k_outer": [
                k_outer_outer_outer_outer, k_outer_outer_outer_inner,
                k_outer_outer_inner
            ],
            "k_coeff":
                tvm.all(
                    (batch_axis % d_out * stride_d +
                     (((k_outer_outer_outer_outer *
                        (outer_factor // inner_factor) + k_outer_outer_outer_inner)
                       * k_outer_outer_inner_size + k_outer_outer_inner) *
                      reduce_axis_factor[0][1] //
                      (w_h * w_w)) // x_factor >= pad_head),
                    (batch_axis % d_out * stride_d +
                     (((k_outer_outer_outer_outer *
                        (outer_factor // inner_factor) + k_outer_outer_outer_inner)
                       * k_outer_outer_inner_size + k_outer_outer_inner) *
                      reduce_axis_factor[0][1] //
                      (w_h * w_w)) // x_factor < fmap_d + pad_head)),
            "k_cond":
                tvm.any(
                    tvm.all(
                        (batch_axis % d_out * stride_d +
                         (((k_outer_outer_outer_outer *
                            (outer_factor // inner_factor) +
                            k_outer_outer_outer_inner) * k_outer_outer_inner_size +
                           k_outer_outer_inner) * reduce_axis_factor[0][1] //
                          (w_h * w_w)) // x_factor == pad_head),
                        (((k_outer_outer_outer_outer *
                           (outer_factor // inner_factor) +
                           k_outer_outer_outer_inner) * k_outer_outer_inner_size +
                          k_outer_outer_inner) * reduce_axis_factor[0][1] %
                         (w_h * w_w * x_factor) <= 0)),
                    ((k_outer_outer_outer_outer *
                      (outer_factor // inner_factor) + k_outer_outer_outer_inner) *
                     k_outer_outer_inner_size + k_outer_outer_inner) == 0),
        }
        self._intrin_mapping(fmap, mad_dict, buffer_dict, new_fmap_col_axis,
                             tiling, cn_axis, l0a_load2d_flag)

        if self.dsl_flag:
            sch[res_c].emit_insn(c_pragma_axis, 'dma_copy')

        ########################### cube schedule end #########################
        self._attach_at(self.body_ops, self.input_ops, compute_at_buffer,
                        compute_at_axis, tiling)
        self._to_pragma(self.body_ops, self.input_ops, c_outer_inner_inner)

        def _get_al1_bound():
            cin1_g = group_dict["cin1_g"]
            _, _, _, fmap_hi, fmap_wi, fmap_c0 = fmap.shape
            stride_h = conv3d_compute.Conv3DParam.tiling_info_dict["stride"][1]
            l0c_mc, l0c_m0 = tiling['CL0_matrix'][1:3]
            stride_update = 1 if self._tensor_map["opti_h_flag"] else stride_h
            EXTEND_H_CALCULATE_FACTOR = 2
            if tiling["AL1_shape"]:
                al1_m_tiling = tiling["AL1_shape"][1] * c_tiling_factor[1]
                if l0a_load2d_flag:
                    al1_m = al1_m_tiling
                elif "fmap_w" in self.var_map:
                    # dynamic_hw choose the value of additional_rows according to w_out
                    additional_rows = tvm.select(
                        tvm.floormod(al1_m_tiling, w_out) == 0,
                        0,
                        tvm.select(tvm.floormod(al1_m_tiling * EXTEND_H_CALCULATE_FACTOR, w_out) == 0,
                            1, 2))
                    ho_len = tvm.floordiv(al1_m_tiling, self.var_map['w_out']) + additional_rows
                    hi_max = c_ub.op.attrs['kernel_h'] + (ho_len - 1)*stride_update
                    al1_m = hi_max * self.var_map['fmap_w']
                else:
                    if al1_m_tiling % int(w_out) == 0:
                        additional_rows = 0
                    elif al1_m_tiling * EXTEND_H_CALCULATE_FACTOR % int(w_out) == 0:
                        additional_rows = 1
                    else:
                        additional_rows = 2
                    ho_len = tvm.floordiv(al1_m_tiling, w_out) + additional_rows
                    hi_max = c_ub.op.attrs['kernel_h'] + (ho_len - 1)*stride_update
                    al1_m = hi_max * fmap_wi
                return al1_m * tiling["AL1_shape"][0] * fmap_c0
            else:
                if self._tensor_map["opti_h_flag"]:
                    fmap_hi = (fmap_hi - 1) // stride_h + 1
                al1_m = fmap_hi * fmap_wi
                if l0a_load2d_flag:
                    align_util = 16
                    al1_m = compute_util.align(al1_m, align_util)
                return al1_m * cin1_g * fmap_c0 * kernel_d

        if self.var_map:
            sch[al1].set_storage_bound(_get_al1_bound())
            # disable_allocate
            sch.disable_allocate(tbe_platform_info.scope_cbuf)
            sch.disable_allocate(tbe_platform_info.scope_ca)
            sch.disable_allocate(tbe_platform_info.scope_cb)
            sch.disable_allocate(tbe_platform_info.scope_cc)
            sch.disable_allocate(tbe_platform_info.scope_ubuf)

            # mem_unique
            sch[al1].mem_unique()
            sch[fmap_col].mem_unique()
            if tiling["BL1_shape"] is not None:
                sch[bl1].mem_unique()
            sch[bl0].mem_unique()
            sch[c_col].mem_unique()
            sch[c_ub].mem_unique()
        if self.var_map:
            return True
        tensor_map.clear()
        dim_map.clear()
        tiling.clear()
        return True

    def _get_elmwise_instr(self, elm_instr):
        """
        Get the instr for element-wise ops.
        """
        ele_map = {"elewise_single_relu": "vector_relu",
                   "elewise_single_round_d": "vector_conv_round",
                   "elewise_single_VS_max": "vector_maxs",
                   "elewise_single_VS_min": "vector_mins",
                   "elewise_binary_div": "vector_div",
                   "elewise_binary_vcmpv_gt": "vector_gt",
                   "elewise_binary_vcmpv_ge": "vector_ge",
                   "elewise_binary_vcmpv_lt": "vector_lt",
                   "elewise_binary_vcmpv_le": "vector_le",
                   "elewise_binary_vcmpv_eq": "vector_eq",
                   "elewise_binary_vcmpv_ne": "vector_ne",
                   "elewise_binary_cmpsel": "vector_cmpsel",
                   "elewise_binary_add": "vector_add",
                   "elewise_binary_sub": "vector_sub",
                   "elewise_binary_mul": "vector_mul",
                   "elewise_binary_min": "vector_min",
                   "elewise_binary_max": "vector_max",
                   "elewise_binary_or": "vector_or",
                   "elewise_binary_and": "vector_and",
                   "elewise_single_lrelu": "vector_lrelu",
                   "elewise_binary_addrelu": "vector_addrelu",
                   "elewise_binary_subrelu": "vector_subrelu"}
        emit_insn_pragma = ele_map.get(elm_instr)
        if emit_insn_pragma:
            out_instr = emit_insn_pragma
        else:
            out_instr = elm_instr

        return out_instr

    def __pragma_for_op(self, lop, c_outer_inner_inner=None):
        # for not in conv op pragma
        op_cmd = lop["op"].split("_")
        cache_buffer = lop["dst_buffer"]
        tensorize_axis = lop["tensorize_axis"]

        if op_cmd[0].lower() == "elewise":
            ele_instr = self._get_elmwise_instr(lop["op"])
            self._schedule[cache_buffer].emit_insn(tensorize_axis, ele_instr)
        elif lop["op"] == 'conv3d_C':
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], 'dma_copy')
        elif lop["op"] == 'conv_vector_remove_pad':
            self._schedule[cache_buffer].emit_insn(c_outer_inner_inner,
                                                   'dma_copy')
        elif lop["op"] == 'conv_vector_bias_add':
            self._schedule[cache_buffer].emit_insn(tensorize_axis,
                                                   "vector_add")
        elif lop["op"] == 'broadcast_for_tensor':
            self._schedule[cache_buffer].emit_insn(tensorize_axis,
                                                   "vector_auto")
        elif lop["op"] == 'mean_matrix_init':
            self._schedule[cache_buffer].emit_insn(self._schedule[cache_buffer].op.axis[-1],
                                                   "vector_dup")
        elif lop["op"] == 'mean_matrix_fp16':
            self._schedule[cache_buffer].emit_insn(self._schedule[cache_buffer].op.axis[0],
                                                   "vector_auto")
        elif lop["op"] == 'mean_matrix_mul':
            self._schedule[cache_buffer].emit_insn(self._schedule[cache_buffer].op.axis[-1],
                                                   "vector_auto")
        else:
            pass


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

        if "conv3d_A" in tmp_op["op"]:
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
                    if tmp_op["op"] == "conv3d_c_col":
                        i.tag = "conv3d_Input"
                    if tmp_op["op"] == "conv3d_fuse_fmap_tensor":
                        i.tag = "conv3d_A"
                    if tmp_op["op"] == "conv3d_al1_load2d":
                        i.tag = "conv3d_A"
                    if tmp_op["op"] == "conv3d_bias_l0c":
                        i.tag = "conv3d_bias_tensor"
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
        """
        get op by source tensor

        Parameters
        ----------
        src_tensor: the source tensor

        Returns
        -------
        tensor : op
        """
        for i in self._op:
            if i["dst_buffer"].same_as(src_tensor):
                return i
        return None

    def __analyse_input_output(self):
        spec_body_ops = {"mean_matrix_init"}
        input_ops = []
        output_ops = []
        body_ops = []
        input_tensor_name = []
        body_tensor_name = []
        for lop in self._op:
            if not lop["prev_op"] and lop["op"] not in spec_body_ops:
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
