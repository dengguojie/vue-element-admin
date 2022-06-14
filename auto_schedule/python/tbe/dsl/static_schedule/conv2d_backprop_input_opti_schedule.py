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
conv2d_backprop_input schedule
"""
from copy import deepcopy
from functools import reduce

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.utils.errormgr import error_manager_cube
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.compute import cube_util
from tbe.dsl.compute.conv2d_backprop_input_opti_compute import DeConvKernelSize1Pattern
from tbe.dsl.compute.util import int_ceil_div
from tbe.dsl.compute.util import align
from tbe.dsl.static_schedule.util import get_fixpipe_emit_str
from tbe.dsl.static_schedule.util import L1CommonParam
from tbe.dsl.static_schedule.util import parse_tbe_compile_para


# Don't modify,used in log_util
DX_SUPPORT_TAG_LOG_PREFIX = "#Conv2DBackpropInput only support#"
# default False
DEBUG_MODE = 0
CONST_L1_SHAPE_DIM = 4
OUT_OF_ORDER_SHIFT_BIT = 13
DTYPE_BYTE_MAP = {"float16": 2, "float32": 4, "int8": 1, "int32": 4, "bfloat16": 2}
CUB_BUFFER_LIMIT = 4096
TENSOR_MAP = {}
TILING = {}
DIM_MAP = {}
DOUBLE_TENSOR_OUT = []
CUBE_MUL_SHAPE = 256

FUSION_DX_DRELU = "dx+drelu"
FUSION_DX_ELEWISE = "dx+elewise"
FUSION_DX_ADD_DRELU = "dx+vadd+drelu"
FUSION_DX_DEQUANT = "dx+dequant"
FUSION_DX_DEQUANT_QUANT = "dx+dequant+quant"
FUSION_DX_REQUANT = "dx+requant"
FUSION_NONE = ""
FUSION_TYPE_2_OPERAND_NUM = {
    FUSION_NONE: 0,
    FUSION_DX_DRELU: 0.0625,
    FUSION_DX_ELEWISE: 0,
    FUSION_DX_ADD_DRELU: 1.0625,
    FUSION_DX_DEQUANT: 0,
    FUSION_DX_DEQUANT_QUANT: 0,
    FUSION_DX_REQUANT: 0
}

FUSION_TYPE_2_NUM = {
    FUSION_NONE: (1, 2),
    FUSION_DX_ELEWISE: 3,
    FUSION_DX_ADD_DRELU: 4,
    FUSION_DX_DEQUANT: 5,
    FUSION_DX_DEQUANT_QUANT: 6,
    FUSION_DX_REQUANT: 7,
    FUSION_DX_DRELU: 8
}

# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16


class DeconvParam:
    """
    class of DeconvTilingParam
    """

    def __init__(self):
        self.para_map = {"DATA_AMOUNT_CUB": 0, "FUSION_TYPE": FUSION_NONE}

    def update_para_map(self, key, value):
        """
        updata para map with key and value
        """
        self.para_map[key] = value

    def get_para_map(self, key):
        """
        get value by key
        """
        return self.para_map.get(key, None)


class Conv2dDxOptiSchedule:
    """
    class of Conv2dDxOptiSchedule
    """
    def __init__(self):
        self.dx_para = DeconvParam()

    @staticmethod
    def _get_all_tensors(res):
        """
        get all tensor
        :param res: tensor
        :return: list
        """

        all_tensor = {}
        leaf_tensor = {}

        def get(tensor):
            """
            find all tensor
            :param tensor: c_gm
            :return: all tensor
            """
            tensor_list = tensor.op.input_tensors
            for one_tensor in tensor_list:
                if not one_tensor.op.input_tensors:
                    leaf_tensor[one_tensor.op.name] = tensor
                # check which tensor has not been checked
                if one_tensor.op.name not in all_tensor:
                    all_tensor[one_tensor.op.name] = one_tensor
                    if one_tensor.op.tag == "conv2d_backprop_input_opti":
                        continue
                    get(one_tensor)

        get(res)
        return all_tensor, leaf_tensor

    @staticmethod
    def _raise_dx_opti_err(msg):
        """
        In op Conv2DBackpropInput_opti, [%s] % (msg)
        msg for discribe the error info
        the error info only for Conv2DBackpropInput_opti's developers
        """
        args_dict = {"errCode": "E60108", "reason": msg}
        msg = error_manager_util.get_error_message(args_dict)
        raise RuntimeError(args_dict, msg)

    @staticmethod
    def _print_debug(*params):
        """
        print log if debug
        :param params: infos
        :return: None
        """
        if DEBUG_MODE:
            print(params)

    @staticmethod
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
        if DEBUG_MODE and "debug" in process:
            start = process + " IR start"
            end = process + " IR end\n"
            sch = sch.normalize()
            print(start)
            bounds = tvm.schedule.InferBound(sch)
            stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
            print(stmt)
            print(end)

    @staticmethod
    def _calc_double_op_num(fusion_type):
        if fusion_type in (FUSION_DX_DEQUANT_QUANT, FUSION_DX_DEQUANT, FUSION_DX_REQUANT):
            double_op_num = 0
            if fusion_type == FUSION_DX_DEQUANT_QUANT:
                double_op_num += 4
            elif fusion_type == FUSION_DX_DEQUANT:
                for ub_tensor in TENSOR_MAP.get("elewise_tensor"):
                    if "dequant2" in ub_tensor.op.name or "dequant_relu" in ub_tensor.op.name:
                        double_op_num = 1
                for ub_tensor in TENSOR_MAP.get("elewise_tensor"):
                    if len(ub_tensor.op.input_tensors) > 1 and "dequant2" not in ub_tensor.op.name:
                        double_op_num += 1
                double_op_num = min(2, double_op_num)
            double_op_num += 0.125
            FUSION_TYPE_2_OPERAND_NUM[fusion_type] = double_op_num
        if fusion_type == FUSION_DX_ELEWISE:
            double_op_num = 0
            for ub_tensor in TENSOR_MAP.get("ub_list"):
                if len(ub_tensor.op.input_tensors) > 1:
                    double_op_num += 1
            double_op_num = min(2, double_op_num)
            if "bias_add_vector" in TENSOR_MAP:
                double_op_num += 0.125
            FUSION_TYPE_2_OPERAND_NUM[fusion_type] = double_op_num


    def _get_data_amount_l1(self, l1_shape, isdouble, l0c_multi_group_flag):
        """
        using tilling parameter calculate data amount in l1

        Parameters:
        ---------------------------------------------------
        :param l1_shape:  'AL1_shape' or 'BL1_shape'
        :param isdouble:  True or False
        :return:  data amount in l1_shape
        ---------------------------------------------------
        """
        if TILING.get(l1_shape) is None:
            self._raise_dx_opti_err("{} can not be None".format(l1_shape))
        if TILING.get(l1_shape) != [] and len(TILING.get(l1_shape)) != CONST_L1_SHAPE_DIM:
            self._raise_dx_opti_err("{} should be {}".format(l1_shape, CONST_L1_SHAPE_DIM))

        if TILING.get(l1_shape) == [] and not l0c_multi_group_flag:
            if l1_shape == "AL1_shape":
                data_amount_l1 = (
                    reduce(lambda x, y: x * y, DIM_MAP.get("A_matrix_dim")[2:])
                    // TILING.get("block_dim")[2]
                ) * DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend)
            if l1_shape == "BL1_shape":
                data_amount_l1 = (
                    reduce(lambda x, y: x * y, DIM_MAP.get("B_matrix_dim")[1:])
                    // TILING["block_dim"][1]
                )
        else:
            block_m, block_k, block_n = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("b_l1").dtype)["mac"]
            # if l0c_multi_group_flag and TILING.get(l1_shape) is [], l1 comput at l0c
            full_k = DIM_MAP.get("B_matrix_dim")[1] * block_k
            l1_k = TILING.get(l1_shape)[0] if TILING.get(l1_shape) else full_k
            l1_mn = TILING.get(l1_shape)[1] if TILING.get(l1_shape) else 1
            l1_g = TILING.get(l1_shape)[3] if TILING.get(l1_shape) else 1
            if l1_k == 0 or l1_mn == 0:
                self._raise_dx_opti_err("l1_k or l1_mn can not be zero")
            if l1_k % block_k != 0:
                self._raise_dx_opti_err("l1_k can not be divided by {}".format(block_k))
            if l1_shape == "AL1_shape":
                data_amount_l1 = (
                    l1_k
                    * l1_mn
                    * TILING.get("CL0_matrix")[1]
                    * l1_g
                    * block_m
                    * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("a_l1").dtype)
                )
            else:
                data_amount_l1 = (
                    l1_k
                    * l1_mn
                    * TILING.get("CL0_matrix")[0]
                    * l1_g
                    * block_n
                    * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("b_l1").dtype)
                )
            if isdouble == 2:
                data_amount_l1 = data_amount_l1 * 2
        self._print_debug("{} data_amount_l1:{}".format(l1_shape, int(data_amount_l1) / 1024))
        return data_amount_l1


    def _check_tilling_l0(self, l0_shape, l0_space, isdouble):
        """
        check tilling parameter in L0 buffer

        Parameter:
        --------------------------------------------------
        :param l0_shape: 'AL0_matrix' or 'BL0_matrix'
        :param l0_space: LO buffer size
        :param isdouble: True or False
        :return: None
        ---------------------------------------------------
        """
        row = TILING.get(l0_shape)[0]
        col = TILING.get(l0_shape)[1]
        group = TILING.get(l0_shape)[5]
        if row == 0 or col == 0:
            self._raise_dx_opti_err("k, m, n, group in L0A/B can not be zero")
        data_amount_l0 = (
            row
            * col
            * TILING.get(l0_shape)[2]
            * TILING.get(l0_shape)[3]
            * group
            * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("b_l0b").dtype)
            * isdouble
        )
        self._print_debug("data_amount_l0A/B[KB]:", tvm.div(data_amount_l0, 1024))
        if isinstance(data_amount_l0, int) and data_amount_l0 > l0_space:
            self._raise_dx_opti_err("tilling size exceed L0A/B Buffer")


    def _check_tilling_l0c(self, l0c_shape, l0c_space, isdouble):
        """
        check tilling parameter in L0c

        Parameter:
        -----------------------------------------------------
        :param l0c_shape:'CL0_matrix'
        :param l0c_space: LOC buffer size
        :param isdouble: True or False
        :return: None
        -----------------------------------------------------
        """
        cl0_m, cl0_n = TILING.get(l0c_shape)[1], TILING.get(l0c_shape)[0]
        cl0_group = TILING.get(l0c_shape)[5]
        if TILING.get("BL0_matrix") != []:
            bl0_n = TILING.get("BL0_matrix")[1]
            if cl0_m == 0 or cl0_n == 0 or cl0_group == 0:
                self._raise_dx_opti_err("cl0_m, cl0_n can not be zero")
            if cl0_n != bl0_n:
                self._raise_dx_opti_err(
                    "axis n in tilling BL0 " "is not equal to axis n in tilling CL0"
                )
        data_amount_cl0 = (
            cl0_m
            * cl0_n
            * TILING.get(l0c_shape)[2]
            * TILING.get(l0c_shape)[3]
            * cl0_group
            * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("c_l0c").dtype)
            * isdouble
        )
        self._print_debug("data_amount_l0C[KB]:", tvm.div(data_amount_cl0, 1024))
        if isinstance(data_amount_cl0, int) and data_amount_cl0 > l0c_space:
            self._raise_dx_opti_err("tilling size exceed L0C Buffer")


    def _check_tilling_cub(self, default_tiling_flag, strideh, stridew, cub_space, isdouble, is_conv1d_bool):
        """
        check tilling parameter in cub

        Parameter:
        ------------------------------------------------------
        :param cub_matrix: 'CUB_matrix'
        :param strideh: stride_h
        :param stridew: stride_w
        :param cub_space: UB buffer size
        :param isdouble: True or False
        :return: None
        -------------------------------------------------------
        """

        def _get_dilate_cub_size():
            block_m, _, _ = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("c_ub").dtype).get("mac")
            if is_conv1d_bool:
                dilate_cub_size = (
                    (1 + stridew)
                    * nc_factor
                    * cl0_m_extent
                    * TILING.get("CUB_matrix")[3]
                    * group_cub
                    * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("c_ub").dtype)
                    * isdouble
                )
            else:
                if cl0_m_extent < DIM_MAP.get("img_shape")[3]:
                    self._raise_dx_opti_err("mc of CL0_matrix " "smaller than weight of Image")
                if DIM_MAP.get("img_shape")[3] > block_m:
                    check_ifmc_falg = bool(
                        (cl0_m_extent // DIM_MAP.get("img_shape")[3])
                        * DIM_MAP.get("img_shape")[3]
                        * strideh
                        * stridew
                        <= CUB_BUFFER_LIMIT
                    )
                    if (
                        cl0_m_extent % DIM_MAP.get("img_shape")[3] == 0
                        and check_ifmc_falg
                        and DIM_MAP.get("img_shape")[2]
                        % (cl0_m_extent // DIM_MAP.get("img_shape")[3])
                        == 0
                    ):
                        n_is_hfactor = cl0_m_extent // DIM_MAP.get("img_shape")[3]
                    else:
                        n_is_hfactor = (cl0_m_extent - block_m) // DIM_MAP.get("img_shape")[3]
                else:
                    check_ifmc_falg_s = False
                    if cl0_m_extent % DIM_MAP.get("img_shape")[3] == 0:
                        n_is_hfactor = cl0_m_extent // DIM_MAP.get("img_shape")[3]
                        while DIM_MAP.get("img_shape")[2] % n_is_hfactor != 0:
                            n_is_hfactor = n_is_hfactor - 1
                        check_ifmc_falg_s = bool(
                            n_is_hfactor
                            * DIM_MAP.get("img_shape")[3]
                            * DIM_MAP.get("dilate_dim")[0]
                            * DIM_MAP.get("dilate_dim")[1]
                            > CUB_BUFFER_LIMIT
                        )
                    if cl0_m_extent % DIM_MAP.get("img_shape")[3] != 0 or check_ifmc_falg_s:
                        n_is_hfactor = max((cl0_m_extent - block_m), block_m) // DIM_MAP.get("img_shape")[3]
                        while DIM_MAP.get("img_shape")[2] % n_is_hfactor != 0:
                            n_is_hfactor = n_is_hfactor - 1
                dy_w = DIM_MAP.get("img_shape")[3]
                tilling_ub_m0 = TILING.get("CUB_matrix")[3]
                real_m = n_is_hfactor * dy_w
                if ((dy_w % tilling_ub_m0 != 0) and default_tiling_flag == 1
                    and TENSOR_MAP.get("c_ub").dtype != "float16"):
                    # add tiling_ub_m0 is needed by buffer_tile of ub
                    real_m = n_is_hfactor * align(dy_w + tilling_ub_m0, tilling_ub_m0)

                dy_h = DIM_MAP.get("img_shape")[2]
                if dy_w == 1 and dy_h == 1:
                    expansion = 1
                elif dy_w == 1:
                    expansion = strideh
                elif dy_h == 1:
                    expansion = stridew
                else:
                    expansion = stridew * strideh

                dilate_cub_size = (
                    (1 + expansion)
                    * nc_factor
                    * real_m
                    * tilling_ub_m0
                    * group_cub
                    * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("c_ub").dtype)
                    * isdouble
                )
            return dilate_cub_size

        nc_factor, mc_factor = TILING.get("CUB_matrix")[0], TILING.get("CUB_matrix")[1]
        group_cub = TILING.get("CUB_matrix")[5]
        if mc_factor != TILING.get("CL0_matrix")[1]:
            self._raise_dx_opti_err("mc_factor is not equal to mc")
        if TILING.get("CL0_matrix")[0] % nc_factor != 0:
            self._raise_dx_opti_err("nc_factor is not factor of nc")
        cl0_m_extent = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]
        if strideh > 1 or stridew > 1:
            data_amount_cub = _get_dilate_cub_size()
        else:
            data_amount_cub = (
                nc_factor
                * mc_factor
                * TILING.get("CUB_matrix")[2]
                * TILING.get("CUB_matrix")[3]
                * group_cub
                * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("c_ub").dtype)
                * isdouble
            )
        self.dx_para.update_para_map("DATA_AMOUNT_CUB", data_amount_cub)
        self._print_debug(
            "DATA_AMOUNT_CUB[KB]:", self.dx_para.get_para_map("DATA_AMOUNT_CUB") / 1024
        )

        if self.dx_para.get_para_map("DATA_AMOUNT_CUB") > cub_space:
            self._raise_dx_opti_err(
                "tilling ub size:{} exceed CUB Buffer:{}".format(
                    self.dx_para.get_para_map("DATA_AMOUNT_CUB"), cub_space
                )
            )


    def _get_tiling_l0a_l0b(self, cl0_matrix, l0_matrix, instr):
        """ get l0a and l0b matrix according to l0b and l0a matrix"""
        k_dim = DIM_MAP.get("A_matrix_dim")[-3]
        batch_l0 = 1
        g_l0 = 1
        if instr == "A":
            block_m, block_k, block_n = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("a_l0a").dtype)[
                "mac"
            ]
            # l0_matrix is bl0_matrix:[kb, nb, n0, k0]
            if l0_matrix != []:
                full_ab = [cl0_matrix[1], l0_matrix[0], block_m, block_k, batch_l0, g_l0]
            else:
                full_ab = [cl0_matrix[1], k_dim, block_m, block_k, batch_l0, g_l0]
        elif instr == "B":
            block_m, block_k, block_n = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("b_l0b").dtype)[
                "mac"
            ]
            # l0_matrix is al0_matrix:[ma, ka, m0, k0]
            if l0_matrix != []:
                full_ab = [l0_matrix[1], cl0_matrix[0], block_n, block_k, batch_l0, g_l0]
            else:
                full_ab = [k_dim, cl0_matrix[0], block_n, block_k, batch_l0, g_l0]
        else:
            self._raise_dx_opti_err("instr should be A or B")

        return full_ab


    def _check_tilinng_k_l1(self):
        _, block_k, _ = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("b_l1").dtype)["mac"]
        k_al1 = TILING.get("AL1_shape")[0]
        k_bl1 = TILING.get("BL1_shape")[0]
        if k_al1 % k_bl1 != 0 and k_bl1 % k_al1 != 0:
            self._raise_dx_opti_err(
                "kal1 should be divisible by kbl1 or kbl1" "should be divisible by kal1 "
            )
        if k_al1 % (TILING.get("AL0_matrix")[1] * block_k) != 0:
            self._raise_dx_opti_err("ka should be divisible by kal1")
        if (
            TILING.get("BL0_matrix")
            and k_bl1 % (TILING.get("BL0_matrix")[0] * block_k) != 0
        ):
            self._raise_dx_opti_err("kb should be divisible by kbl1")


    def _check_tiling_bl0_matrix(self, manual_pingpong_buffer, data_amount_l1b, l0c_multi_group_flag):
        if TILING.get("BL0_matrix") is None:
            self._raise_dx_opti_err("tiling[BL0_matrix] can not be None")
        if TILING.get("BL0_matrix") == []:
            if TILING.get("BL1_shape") == []:
                data_amount_l0b = data_amount_l1b
            else:
                data_amount_l0b = data_amount_l1b // TILING.get("BL1_shape")[3]
            if data_amount_l0b > tbe_platform_info.get_soc_spec("L0B_SIZE"):
                self._raise_dx_opti_err("tiling size exceed L0B Buffer")
        else:
            self._check_tilling_l0(
                "BL0_matrix",
                tbe_platform_info.get_soc_spec("L0B_SIZE"),
                manual_pingpong_buffer.get("BL0_pbuffer")
            )
            full_k1 = DIM_MAP.get("dy_c1_extend")
            if TILING.get("BL0_matrix")[0] != full_k1 and TILING.get("BL0_matrix")[5] > 1:
                self._raise_dx_opti_err("If axis k in tiling BL0 is not full load, group_BL0 must be 1.")

            if TILING.get("AL0_matrix")[1] != TILING.get("BL0_matrix")[0] and not l0c_multi_group_flag:
                self._raise_dx_opti_err(
                    "axis k in tilling AL0 is not equal to axis k in tilling BL0"
                )

    @staticmethod
    def _check_and_set_default_tiling(tiling, atype, btype, l0c_multi_group_flag):
        """
        check and set default tiling
        :param tiling:
        :param atype:
        :param btype:
        :return: default tiling
        """

        # checkout tiling["AL0_matrix"][2] flag
        def _check_tiling(tiling):
            if tiling["AL0_matrix"][2] == 32:
                return False
            return True

        if not _check_tiling(tiling):
            tiling = {}
            bit_dir = {
                "float32": 8,
                "int32": 16,
                "float16": 16,
                "int8": 32,
                "bfloat16": 16
            }
            k_al1 = bit_dir.get(atype, 32)
            k_al0 = bit_dir.get(atype, 32)
            k_bl1 = bit_dir.get(btype, 32)
            k_bl0 = bit_dir.get(btype, 32)

            m_al0 = m_cl0 = 1
            if TENSOR_MAP.get("dilate_ub") is not None:
                # when mmad, the min unit of M is a fmp's w
                dy_w = DIM_MAP.get("img_shape")[3]
                block_m = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("c_l0c").dtype).get("mac")[0]
                # mc's calculation rule refor to auto tiling
                if dy_w % 16 == 0:
                    m_al0 = m_cl0 = dy_w // block_m
                else:
                    # add one is needed by buffer_tile of ub
                    m_al0 = m_cl0 = int_ceil_div(dy_w, block_m) + 1
            if l0c_multi_group_flag:
                n_min = DIM_MAP.get("dx_c1_extend")
                group_cl0 = 2
            else:
                n_min = 1
                group_cl0 = 1
            ka_factor = 2 if atype == "float32" else 1
            kb_factor = 2 if btype == "float32" else 1
            tiling["AUB_shape"] = None
            tiling["BUB_shape"] = None
            tiling["AL1_shape"] = [k_al1 * ka_factor, 1, 1, 1]
            tiling["BL1_shape"] = [k_bl1 * kb_factor, 1, 1, 1]
            tiling["AL0_matrix"] = [m_al0, ka_factor, 16, k_al0, 1, 1]
            tiling["BL0_matrix"] = [kb_factor, n_min, 16, k_bl0, 1, 1]
            tiling["CL0_matrix"] = [n_min, m_cl0, 16, 16, 1, group_cl0]
            tiling["CUB_matrix"] = [n_min, m_cl0, 16, 16, 1, group_cl0]
            tiling["block_dim"] = [1, 1, 1, 1]
            tiling["n_bef_batch_flag"] = 0
            tiling["n_bef_group_flag"] = 0
            tiling["batch_bef_group_fla"] = 0
            tiling["A_overhead_opt_flag"] = 0
            tiling["B_overhead_opt_flag"] = 0
            tiling["AUB_channel_wise_flag"] = None
            tiling["BUB_channel_wise_flag"] = None
            tiling["CUB_channel_wise_flag"] = None
            tiling["manual_pingpong_buffer"] = {
                "AUB_pbuffer": 1,
                "BUB_pbuffer": 1,
                "AL1_pbuffer": 1,
                "BL1_pbuffer": 1,
                "AL0_pbuffer": 1,
                "BL0_pbuffer": 1,
                "CL0_pbuffer": 1,
                "CUB_pbuffer": 1,
                "UBG_pbuffer": 1
            }
        return tiling


    def _get_tiling(
        self,
        tensor,
        fusion_type,
        kernel_name,
        is_conv1d_bool,
        tiling_case=None,
        var_map=None,
        l0c_multi_group_flag=False
    ):
        """
        get tilling parameter from get_tilling and check all parameter
        """

        def _handle_quant_tiling():
            if fusion_type in (FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
                TILING["CUB_matrix"][3] = 32
                TILING["CL0_matrix"][3] = 32
                if not l0c_multi_group_flag:
                    # correct c1_dim due to c1_l0c/c1_ddr is 2
                    TILING["CL0_matrix"][0] //= 2
                    if TILING["CL0_matrix"][0] == 0:
                        TILING["CL0_matrix"][0] = 1

                    TILING["CUB_matrix"][0] //= 2
                    if TILING["CUB_matrix"][0] == 0:
                        TILING["CUB_matrix"][0] = 1
                else:
                    # in l0c_multi_group_flag scenes, correct g_dim
                    TILING["CL0_matrix"][5] //= 2
                    TILING["CUB_matrix"][5] //= 2

            def _group_quant_illegal_quant():
                if l0c_multi_group_flag:
                    # n axis is full load, block_dim_n is 1
                    n_dim_rule = (TILING["block_dim"][1] == 1 and TILING["CL0_matrix"][0] == filter_shape_g[1]
                                and TILING["CUB_matrix"][0] == filter_shape_g[1])
                    if TILING["BL1_shape"]:
                        n_dim_rule = n_dim_rule and TILING["BL1_shape"][1] == 1
                    if not n_dim_rule:
                        self._raise_dx_opti_err("Illegal tiling in dequant + quant or requant fusion scene.")

            _group_quant_illegal_quant()

        def _check_tiling_l1():
            data_amount_l1b = self._get_data_amount_l1(
                "BL1_shape", manual_pingpong_buffer.get("BL1_pbuffer"), l0c_multi_group_flag
            )
            if "dedy_h" not in var_map and "dedy_w" not in var_map:
                data_amount_l1a = self._get_data_amount_l1(
                    "AL1_shape", manual_pingpong_buffer.get("AL1_pbuffer"), l0c_multi_group_flag
                )
                if self.dx_para.get_para_map("l1_fusion_type") != -1:
                    data_amount_l1a = 0
                if (int(data_amount_l1a) + int(data_amount_l1b)) > tbe_platform_info.get_soc_spec("L1_SIZE"):
                    self._raise_dx_opti_err("tiling size exceed L1 Buffer")

            if TILING.get("BL1_shape") and TILING.get("AL1_shape"):
                self._check_tilinng_k_l1()
            return data_amount_l1b

        no_need_use_ub_flag = self.dx_para.get_para_map("no_need_use_ub_flag")
        _, block_k, block_n = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("b_l1").dtype).get("mac")
        # if filter dtype is int8, than channel block_size is 32
        if tensor.dtype == "int32" or fusion_type in (
            FUSION_DX_DEQUANT,
            FUSION_DX_DEQUANT_QUANT,
            FUSION_DX_REQUANT
        ):
            # DIM_MAP["filter_shape"] : co_dim, ci_dim, _, _
            filter_shape_g = [
                DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend) * block_k,
                DIM_MAP.get(cube_util.GroupDictKeys.dx_c1_extend),
                1,
                1,
                block_n]
        elif TENSOR_MAP.get("filter_placehold").dtype == "float32":
            filter_shape_g = [
                DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend) * block_k * 2,
                (DIM_MAP.get(cube_util.GroupDictKeys.dx_c1_extend) + 1) // 2,
                1,
                1,
                block_n
            ]
        else:
            # DIM_MAP["filter_shape"] : ci_dim, co_dim, _, _
            filter_shape_g = [
                DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend) * block_k,
                DIM_MAP.get(cube_util.GroupDictKeys.dx_c1_extend),
                1,
                1,
                block_n]

        if TENSOR_MAP.get("dilate_ub") is None:
            strideh, stridew = 1, 1
        else:
            strideh, stridew = DIM_MAP.get("dilate_dim")
        # times of the dx ub space
        self._calc_double_op_num(fusion_type)

        if fusion_type in (FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
            if not l0c_multi_group_flag:
                filter_shape_g[1] = (filter_shape_g[1] + 1) // 2 * 2

        bias_flag = self._get_bias_flag()

        global TILING
        if not TILING:
            if not var_map:
                in_fm_memory_type = self.dx_para.get_para_map("input_memory_type")
                out_fm_memory_type = self.dx_para.get_para_map("output_memory_type")
                info_dict = {
                    "op_type": "conv2d_backprop_input",
                    "A_shape": list(DIM_MAP.get("dy_6GD_shape")[1:]),
                    "B_shape": list(filter_shape_g),
                    "C_shape": None,
                    "A_dtype": str(TENSOR_MAP.get("img_placehold").dtype),
                    "B_dtype": str(TENSOR_MAP.get("filter_placehold").dtype),
                    "C_dtype": str(tensor.dtype),
                    "mad_dtype": str(TENSOR_MAP.get("c_l0c").dtype),
                    "padl": 0,
                    "padr": 0,
                    "padu": 0,
                    "padd": 0,
                    "strideH": 1,
                    "strideW": 1,
                    "strideH_expand": strideh,
                    "strideW_expand": stridew,
                    "dilationH": 1,
                    "dilationW": 1,
                    "group": DIM_MAP.get(cube_util.GroupDictKeys.g_extend),
                    "bias_flag": bias_flag,
                    "fused_double_operand_num": FUSION_TYPE_2_OPERAND_NUM.get(fusion_type),
                    "kernel_name": kernel_name.value,
                    "in_fm_memory_type": in_fm_memory_type,
                    "out_fm_memory_type": out_fm_memory_type,
                    "l1_fusion_type": self.dx_para.get_para_map("l1_fusion_type"),
                    "fusion_type": self.dx_para.get_para_map("fusion_type_num")
                }
                TILING = get_tiling(info_dict)
            else:
                TILING = deepcopy(tiling_case)

        if no_need_use_ub_flag:
            TILING["CUB_matrix"] = TILING.get("CL0_matrix")

        if "dedy_w" in var_map and not var_map.get("dedy_w")[1]:
            mc = int_ceil_div(DIM_MAP.get("img_shape")[3], TILING.get("AL0_matrix")[2]) + 1
            TILING["AL0_matrix"][0] = TILING["CL0_matrix"][1] = TILING["CUB_matrix"][1] = mc
        compile_param = TILING.get("tbe_compile_para")
        # when compile_param is None, tbe_compile_para is None
        tbe_compile_para, tbe_sch_control_para = parse_tbe_compile_para(compile_param)
        self.dx_para.update_para_map("tbe_compile_para", tbe_compile_para)
        self.dx_para.update_para_map('preload_c_l0c', tbe_sch_control_para.get("preload"))
        self.dx_para.update_para_map('preload_a_l1', tbe_sch_control_para.get("preload_l1"))
        out_of_order = False
        if tbe_compile_para is not None:
            out_of_order = tbe_compile_para.get("out_of_order")
        self.dx_para.update_para_map("out_of_order", out_of_order)
        default_tiling_flag = 0
        if TILING["AL0_matrix"][2] == 32:
            default_tiling_flag = 1
        TILING = self._check_and_set_default_tiling(
            TILING, TENSOR_MAP.get("img_placehold").dtype, TENSOR_MAP.get("filter_placehold").dtype,
            l0c_multi_group_flag
        )

        self._print_debug(
            "opti dx shape, kernel_name:",
            kernel_name,
            "filter:",
            filter_shape_g,
            "dy:",
            DIM_MAP.get("img_shape"),
            "dx:",
            DIM_MAP.get("out_img_shape")
        )
        self._print_debug("tiling:", TILING)

        if TILING.get("AL0_matrix") == []:
            TILING["AL0_matrix"] = self._get_tiling_l0a_l0b(
                TILING.get("CL0_matrix"), TILING.get("BL0_matrix"), "A"
            )

        if TILING.get("BL0_matrix") == []:
            TILING["BL0_matrix"] = self._get_tiling_l0a_l0b(
                TILING.get("CL0_matrix"), TILING.get("AL0_matrix"), "B"
            )

        manual_pingpong_buffer = TILING.get("manual_pingpong_buffer")

        data_amount_l1b = _check_tiling_l1()

        # check tilling in AL0 BL0
        if TILING.get("AL0_matrix") is None or TILING.get("AL0_matrix") == []:
            self._raise_dx_opti_err("tiling[AL0_matrix] can not be None or []")
        self._check_tilling_l0(
            "AL0_matrix",
            tbe_platform_info.get_soc_spec("L0A_SIZE"),
            manual_pingpong_buffer.get("AL0_pbuffer")
        )

        self._check_tiling_bl0_matrix(manual_pingpong_buffer, data_amount_l1b, l0c_multi_group_flag)

        # check tilling in CL0
        self._check_tilling_l0c(
            "CL0_matrix",
            tbe_platform_info.get_soc_spec("L0C_SIZE"),
            manual_pingpong_buffer.get("CL0_pbuffer")
        )

        # check tilling in CUB  attention:light when stride get  #########
        if "dedy_h" not in var_map and "dedy_w" not in var_map and not no_need_use_ub_flag:
            self._check_tilling_cub(
                default_tiling_flag,
                strideh,
                stridew,
                tbe_platform_info.get_soc_spec("UB_SIZE"),
                manual_pingpong_buffer.get("CUB_pbuffer"),
                is_conv1d_bool
            )

        _handle_quant_tiling()

    @staticmethod
    def _get_bias_flag():
        if (
            TENSOR_MAP.get("bias_add_vector") is not None
            or TENSOR_MAP.get("c_add_bias") is not None
        ):
            bias_flag = 1
        else:
            bias_flag = 0
        return bias_flag

    @staticmethod
    def _get_src_tensor(tensor, index):
        """
        get input tensor according to the specified index
        :param tensor: Tensor for getting input tensor
        :param index: specified index
        :return: specified input tensor
        """
        if tensor is not None and tensor.op.input_tensors:
            return tensor.op.input_tensors[index]
        return None


    def _quant_tensor_info(self, all_tensor, leaf_tensor, dx_res, quant_para):
        """
        check dequant + quant
        """

        self.dx_para.update_para_map("FUSION_TYPE", FUSION_DX_DEQUANT_QUANT)
        quant_para["q_round"] = dx_res.op.attrs["round_mode"].value

        elewise_cache_list = []
        input_cache_buffer = []

        for key, value in all_tensor.items():
            if "reform" in key:
                TENSOR_MAP["reform_op"] = value
            elif key == "input_ub":
                TENSOR_MAP[key] = value
                if value.op.attrs["c_out"].value % 2:
                    quant_para["q_padding"] = True
            elif key == "cast_i8_ub":
                TENSOR_MAP[key] = value
            elif key in ("dequant", "dequant1"):
                TENSOR_MAP["deq"] = self._get_src_tensor(value, 1)
                TENSOR_MAP["c_ub"] = value
                if "vector" in value.op.tag:
                    quant_para["deq_vector"] = True
            elif not value.op.input_tensors:
                if "dequant" not in leaf_tensor.get(key).op.name:
                    input_cache_buffer.append([value, leaf_tensor.get(key)])
            elif value.op.tag != "conv2d_backprop_input_opti":
                elewise_cache_list.append(value)

        TENSOR_MAP["elewise_tensor"] = elewise_cache_list
        TENSOR_MAP["input_tensor"] = input_cache_buffer
        return quant_para


    def _requant_tensor_info(self, dx_res, quant_para):
        """
        check requant
        """
        self.dx_para.update_para_map("FUSION_TYPE", FUSION_DX_REQUANT)
        TENSOR_MAP["data_transfer"] = self._get_src_tensor(dx_res, 0)
        TENSOR_MAP["c_ub"] = self._get_src_tensor(TENSOR_MAP.get("data_transfer"), 0)
        if "vector" in TENSOR_MAP.get("c_ub").op.tag:
            quant_para["req_vector"] = True
        TENSOR_MAP["deq"] = self._get_src_tensor(TENSOR_MAP.get("c_ub"), 1)
        return quant_para


    def _dequant_elewise_info(self, all_tensor, leaf_tensor, quant_para):
        """
        check dequant + elewise
        """
        elewise_cache_list = []
        input_cache_buffer = []

        for key, value in all_tensor.items():
            if key in ("dequant", "dequant1"):
                self.dx_para.update_para_map("FUSION_TYPE", FUSION_DX_DEQUANT)
                TENSOR_MAP["deq"] = self._get_src_tensor(value, 1)
                TENSOR_MAP["c_ub"] = value
                if "vector" in value.op.tag:
                    quant_para["deq_vector"] = True
            elif not value.op.input_tensors:
                if "dequant" not in leaf_tensor.get(key).op.name:
                    input_cache_buffer.append([value, leaf_tensor.get(key)])
            elif value.op.tag != "conv2d_backprop_input_opti":
                elewise_cache_list.append(value)

        if self.dx_para.get_para_map("FUSION_TYPE") == FUSION_DX_DEQUANT:
            TENSOR_MAP["elewise_tensor"] = elewise_cache_list
            TENSOR_MAP["input_tensor"] = input_cache_buffer
        return quant_para


    def _check_quant_fusion(self, dx_res):
        """
        check the quant fusion

        :return the quant para
        """

        quant_para = {
            "deq_vector": False,
            "req_vector": False,
            "q_round": None,
            "q_padding": False
        }
        all_tensor, leaf_tensor = self._get_all_tensors(dx_res)

        if dx_res.op.tag == "quant":
            quant_para = self._quant_tensor_info(all_tensor, leaf_tensor, dx_res, quant_para)
        elif "requant_remove_pad" in dx_res.op.tag:
            quant_para = self._requant_tensor_info(dx_res, quant_para)
        else:
            quant_para = self._dequant_elewise_info(all_tensor, leaf_tensor, quant_para)

        return quant_para


    def _set_data_layout(self, res, dex_res, sch, var_range):
        """
        get DIM_MAP which contains all ops

        Parameter:
        ----------------------------------------------------------
        :param res: op
        :param dex_res: op
        :param sch: schedule
        :param var_range: var_range for dynamic shape
        :return: None
        ----------------------------------------------------------
        """

        def _get_tensor_dx_gm(tensor_add_res):
            """
            get dx_gm tensor by add_res tensor
            :param sch:
            :param tensor_add_res: add_res tensor
            :return: dx_gm tensor
            """
            global TENSOR_MAP
            tensor_add_left = tensor_add_res.op.input_tensors[0]
            tensor_add_right = tensor_add_res.op.input_tensors[1]
            if tensor_add_left.op.tag == "conv2d_backprop_input_opti":
                tensor_dx_gm = tensor_add_left
                tensor_add_input_gm = tensor_add_right
            else:
                tensor_dx_gm = tensor_add_right
                tensor_add_input_gm = tensor_add_left

            tensor_add_input_ub = sch.cache_read(
                tensor_add_input_gm, tbe_platform_info.scope_ubuf, [tensor_add_res]
            )
            return tensor_dx_gm, tensor_add_input_ub

        def _check_dx_fusion_type(res, fusion_tensor_map):
            """
            check fusion type and set buffer
            """

            def _handle_elewise_tensor():
                deq_ub_read = [fusion_tensor_map.get("c_ub")]
                for elewise_tensor_mem in fusion_tensor_map.get("elewise_tensor", []):
                    if elewise_tensor_mem.op.tag != res.op.tag:
                        sch[elewise_tensor_mem].set_scope(tbe_platform_info.scope_ubuf)
                    if "dequant2" in elewise_tensor_mem.op.name:
                        deq_ub_read.append(elewise_tensor_mem)

                deq_scale = fusion_tensor_map.get("deq")
                deq_ub = sch.cache_read(deq_scale, tbe_platform_info.scope_ubuf, deq_ub_read)
                fusion_tensor_map["deq"] = deq_ub

                input_list = []
                for input_tensor_mem in fusion_tensor_map.get("input_tensor", []):
                    if input_tensor_mem[1].op.tag != res.op.tag:
                        input_tensor_des = input_tensor_mem[1]
                    else:
                        input_tensor_des = c_ub_res
                    input_tensor_ub = sch.cache_read(input_tensor_mem[0], tbe_platform_info.scope_ubuf,
                                                     input_tensor_des)
                    input_list.append(input_tensor_ub)
                fusion_tensor_map["input_tensor"] = input_list

            def _handle_elewise_fusion():
                self.dx_para.update_para_map("FUSION_TYPE", FUSION_DX_ELEWISE)
                input_tensor_list = []
                ub_list = []
                c_ub_res = sch.cache_write(res, tbe_platform_info.scope_ubuf)
                all_tensor, leaf_tensor = self._get_all_tensors(res)
                for key, value in all_tensor.items():
                    if value.op.tag == "conv2d_backprop_input_opti":
                        tensor_dx_gm = value
                        sch[value].compute_inline()
                    elif value.op.input_tensors:
                        sch[value].set_scope(tbe_platform_info.scope_ubuf)
                        ub_list.append(value)
                    else:
                        if leaf_tensor.get(key).op.tag == res.op.tag:
                            input_tensor_des = c_ub_res
                        else:
                            input_tensor_des = leaf_tensor.get(value.op.name)
                        input_tensor_ub = sch.cache_read(value, tbe_platform_info.scope_ubuf, input_tensor_des)
                        input_tensor_list.append(input_tensor_ub)
                ub_list.append(c_ub_res)
                fusion_tensor_map["input_tensor_list"] = input_tensor_list
                fusion_tensor_map["ub_list"] = ub_list
                fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm

            def _handle_dx_add_drelu_fusion():
                tensor_dx_gm = None
                tensor_add_input_ub = None
                tensor_add_input_1_ub = None
                inter_add_compute_tensor = None
                self.dx_para.update_para_map("FUSION_TYPE", FUSION_DX_ADD_DRELU)
                tensor_add_res = drelu_gm.op.input_tensors[1]
                sch[tensor_add_res].set_scope(tbe_platform_info.scope_ubuf)
                fusion_tensor_map["add_res_ub"] = tensor_add_res
                all_tensor, _ = self._get_all_tensors(tensor_add_res)
                if len(all_tensor) == 2:
                    tensor_dx_gm, tensor_add_input_ub = _get_tensor_dx_gm(tensor_add_res)
                if len(all_tensor) == 4:
                    placeholder_list = []
                    for one_tensor in all_tensor.values():
                        if isinstance(one_tensor.op, tvm.tensor.PlaceholderOp):
                            placeholder_list.append(one_tensor)
                        elif one_tensor.op.tag == "conv2d_backprop_input_opti":
                            tensor_dx_gm = one_tensor
                        elif one_tensor.op.tag == "elewise_binary_add":
                            inter_add_compute_tensor = one_tensor
                    tensor_add_input_gm = placeholder_list[1]
                    tensor_add_input_1_gm = placeholder_list[0]
                    sch[inter_add_compute_tensor].set_scope(tbe_platform_info.scope_ubuf)
                    tensor_add_input_1_ub = sch.cache_read(tensor_add_input_gm, tbe_platform_info.scope_ubuf,
                                                               [tensor_add_res])
                    tensor_add_input_ub = sch.cache_read(tensor_add_input_1_gm, tbe_platform_info.scope_ubuf,
                                                             [inter_add_compute_tensor])
                fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm
                fusion_tensor_map["add_input_ub"] = tensor_add_input_ub
                fusion_tensor_map["add_input_1_ub"] = tensor_add_input_1_ub
                fusion_tensor_map["inter_add_compute_tensor"] = inter_add_compute_tensor

            if self.dx_para.get_para_map("FUSION_TYPE") in (
                FUSION_DX_DEQUANT,
                FUSION_DX_DEQUANT_QUANT,
                FUSION_DX_REQUANT
            ):
                if res.op.tag != "dequant_remove_pad" and self.dx_para.get_para_map("FUSION_TYPE") == FUSION_DX_DEQUANT:
                    c_ub_res = sch.cache_write(res, tbe_platform_info.scope_ubuf)
                    fusion_tensor_map["elewise_tensor"].append(c_ub_res)

                for tensor in fusion_tensor_map:
                    if tensor not in ("elewise_tensor", "input_tensor", "deq"):
                        sch[fusion_tensor_map[tensor]].set_scope(tbe_platform_info.scope_ubuf)

                _handle_elewise_tensor()
                tensor_dx_gm = fusion_tensor_map["c_ub"].op.input_tensors[0]
            elif (res.op.tag == "emit_insn_elewise_multiple_sel|bool"
                  or (var_range and res.op.tag == "elewise_multiple_sel")):
                drelu_gm = res
                # dx+add+drelu
                if "elewise_binary_add" in drelu_gm.op.input_tensors[1].op.tag:
                    _handle_dx_add_drelu_fusion()
                    tensor_dx_gm = fusion_tensor_map.get("fusion_dx_gm")
                # dx+drelu
                else:
                    self.dx_para.update_para_map("FUSION_TYPE", FUSION_DX_DRELU)
                    tensor_dx_gm = drelu_gm.op.input_tensors[1]

                tensor_bitmask_gm = drelu_gm.op.input_tensors[0]
                sch[tensor_dx_gm].set_scope(tbe_platform_info.scope_ubuf)
                tensor_bitmask = sch.cache_read(
                    tensor_bitmask_gm, tbe_platform_info.scope_ubuf, [drelu_gm]
                )
                tensor_drelu = sch.cache_write(drelu_gm, tbe_platform_info.scope_ubuf)

                fusion_tensor_map["bitmask_ub"] = tensor_bitmask
                fusion_tensor_map["drelu_ub"] = tensor_drelu
                fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm  # inter_gm
            # dx+add
            elif "elewise" in res.op.tag:
                _handle_elewise_fusion()
                tensor_dx_gm = fusion_tensor_map.get("fusion_dx_gm")
            elif res.op.tag == "conv2d_backprop_input_opti":
                self.dx_para.update_para_map("FUSION_TYPE", FUSION_NONE)
                tensor_dx_gm = res
            elif "5HD_trans_NHWC" in res.op.tag:
                tensor_dx_gm = res.op.input_tensors[0]
                self.dx_para.update_para_map("5HD_TRANS_NHWC", True)
                sch[tensor_dx_gm].compute_inline()
            else:
                self._raise_dx_opti_err(DX_SUPPORT_TAG_LOG_PREFIX + " unsupported data flow")
            return tensor_dx_gm, fusion_tensor_map

        def _get_ub_tensor(fusion_type):
            if tensor_dx_gm.op.input_tensors[0].op.name == "bias_add_vector":
                bias_add_vector = tensor_dx_gm.op.input_tensors[0]
                tensor_dilate_ub = bias_add_vector.op.input_tensors[0]
                tensor_bias = bias_add_vector.op.input_tensors[1]
                sch[bias_add_vector].set_scope(tbe_platform_info.scope_ubuf)
                bias_ub = sch.cache_read(
                    tensor_bias, tbe_platform_info.scope_ubuf, [bias_add_vector]
                )
                TENSOR_MAP["bias_add_vector"] = bias_add_vector
                TENSOR_MAP["bias_ub"] = bias_ub
            else:
                tensor_dilate_ub = tensor_dx_gm.op.input_tensors[0]

            if (
                tensor_dilate_ub is not None
                and tensor_dilate_ub.op.tag == "conv2d_backprop_input_opti"
            ):
                if var_map:
                    TENSOR_MAP["tensor_vn"] = tensor_dilate_ub
                    sch[tensor_dilate_ub].set_scope(tbe_platform_info.scope_ubuf)
                    tensor_fillling_zero = tensor_dilate_ub.op.input_tensors[0]
                    TENSOR_MAP["tensor_fillling_zero"] = tensor_fillling_zero
                    sch[tensor_fillling_zero].set_scope(tbe_platform_info.scope_ubuf)
                    tensor_dilate = tensor_dilate_ub.op.input_tensors[1]
                    TENSOR_MAP["dilate_ub"] = tensor_dilate
                    sch[tensor_dilate].set_scope(tbe_platform_info.scope_ubuf)
                    tensor_cub = tensor_dilate.op.input_tensors[0]
                else:
                    TENSOR_MAP["dilate_ub"] = tensor_dilate_ub
                    sch[tensor_dilate_ub].set_scope(tbe_platform_info.scope_ubuf)
                    tensor_cub = tensor_dilate_ub.op.input_tensors[0]
                    tensor_fillling_zero = tensor_dilate_ub.op.input_tensors[1]
                    TENSOR_MAP["tensor_fillling_zero"] = tensor_fillling_zero
                    sch[tensor_fillling_zero].set_scope(tbe_platform_info.scope_ubuf)
            else:
                if tensor_dx_gm.op.input_tensors[0].op.name == "bias_add_vector":
                    tensor_cub = tensor_dilate_ub
                elif (tensor_dx_gm.op.input_tensors[0].op.name == "C" or
                      tensor_dx_gm.op.input_tensors[0].op.name == "c_add_bt"):
                    tensor_cub = None
                else:
                    tensor_cub = tensor_dx_gm.op.input_tensors[0]
            if fusion_type in (
                FUSION_DX_DEQUANT,
                FUSION_DX_DEQUANT_QUANT,
                FUSION_DX_REQUANT
            ):
                tensor_cub = TENSOR_MAP.get("c_ub")

            return tensor_cub

        def _get_l1_fusion_para():
            fusion_para = DeConvKernelSize1Pattern.fusion_para
            self.dx_para.update_para_map("input_memory_type", [fusion_para.get("input_memory_type")])
            self.dx_para.update_para_map("l1_fusion_type", fusion_para.get("l1_fusion_type"))
            self.dx_para.update_para_map("fmap_l1_addr_flag", fusion_para.get("fmap_l1_addr_flag"))
            self.dx_para.update_para_map("fmap_l1_valid_size", fusion_para.get("fmap_l1_valid_size"))

            load3d_flag = bool(fusion_para.get("l1_fusion_type") != -1)
            self.dx_para.update_para_map("load3d_flag", load3d_flag)
            out_mem = fusion_para.get("output_memory_type")
            if out_mem == "fuse_flag":
                if dex_res.op.tag == "conv_virtual_res":
                    for out_member in DOUBLE_TENSOR_OUT:
                        out_member_addr = out_member
                        if out_member.dtype == "float16":
                            out_member_addr = out_member.op.input_tensors[0]
                        res_addr_type = 0
                        if "addr_type" in out_member_addr.op.attrs:
                            res_addr_type = out_member_addr.op.attrs["addr_type"].value
                        output_memory_type = [res_addr_type]
                        if res_addr_type == 1:
                            sch[out_member].set_scope(tbe_platform_info.scope_cbuf_fusion)
                else:
                    if "addr_type" in dex_res.op.attrs:
                        res_addr_type = dex_res.op.attrs["addr_type"].value
                    else:
                        res_addr_type = 0
                    output_memory_type = [res_addr_type]
                    if res_addr_type == 1:
                        sch[dex_res].set_scope(tbe_platform_info.scope_cbuf_fusion)
            else:
                if out_mem == 1:
                    sch[dex_res].set_scope(tbe_platform_info.scope_cbuf_fusion)
                output_memory_type = [out_mem]
            self.dx_para.update_para_map("output_memory_type", output_memory_type)

        def _al1_fusion_handle():
            if (self.dx_para.get_para_map("load3d_flag") and self.dx_para.get_para_map("input_memory_type")[0] == 1):
                a_l0a_before = a_l0a.op.input_tensors[0]
                dedy = a_l0a_before.op.input_tensors[0]
                dedy_col = sch.cache_read(
                    dedy, tbe_platform_info.scope_cbuf_fusion, [a_l0a_before]
                )
                sch[dedy].set_scope(tbe_platform_info.scope_cbuf_fusion)
                TENSOR_MAP["a_l0a_before"] = a_l0a_before
                al1_shape = dedy_col.shape
                sch[dedy_col].buffer_align(
                    (1, 1),
                    (1, 1),
                    (al1_shape[2], al1_shape[2]),
                    (al1_shape[3], al1_shape[3]),
                    (1, 1)
                )
            elif (self.dx_para.get_para_map("load3d_flag") and self.dx_para.get_para_map("input_memory_type")[0] == 0):
                a_l0a_before = a_l0a.op.input_tensors[0]
                dedy_col = a_l0a_before.op.input_tensors[0]
                dedy = dedy_col.op.input_tensors[0]
                sch[dedy_col].set_scope(tbe_platform_info.scope_cbuf_fusion)
                TENSOR_MAP["a_l0a_before"] = a_l0a_before
                if self.dx_para.get_para_map("l1_fusion_type") == 1:
                    al1_shape = dedy_col.shape
                    sch[dedy_col].buffer_align(
                        (1, 1),
                        (1, 1),
                        (al1_shape[2], al1_shape[2]),
                        (al1_shape[3], al1_shape[3]),
                        (1, 1)
                    )
            else:
                dedy_col_ori = a_l0a.op.input_tensors[0]
                dedy_col = dedy_col_ori.op.input_tensors[0]
                if dedy_col.op.input_tensors:
                    dedy = dedy_col.op.input_tensors[0]
                    self.dx_para.update_para_map("FM_NHWC_TRANS_5HD", True)
                    sch[dedy_col_ori].compute_inline()
                    sch[dedy_col].set_scope(tbe_platform_info.scope_cbuf)
                else:
                    dedy = dedy_col
                    dedy_col = dedy_col_ori
                    sch[dedy_col].set_scope(tbe_platform_info.scope_cbuf)
                a_l0a_before = None
            return dedy_col, dedy, a_l0a_before

        def _al1_buffer_align():
            storage_align_size = 256
            if dedy_col.dtype == "int8":
                storage_align_size = 512

            if self.dx_para.get_para_map("load3d_flag"):
                sch[a_l0a_before].set_scope(tbe_platform_info.scope_cbuf)
                dx_w = DIM_MAP.get("img_shape")[3]
                sch[a_l0a_before].buffer_align(
                    (1, 1),
                    (dx_w, dx_w),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, tbe_platform.CUBE_MKN.get(a_l0a_before.dtype)["mac"][1])
                )
            else:
                sch[dedy_col].storage_align(sch[dedy_col].op.axis[1], storage_align_size, 0)

        def _get_var_map(var_range):
            """
            get var map from var_range
            """
            var_names = ["batch_n", "dedy_h", "dedy_w", "dx_h", "dx_w"]
            var_map = {}
            if not var_range:
                return var_map
            for name in var_names:
                if name in var_range:
                    var_map[name] = var_range[name]
            return var_map

        global TENSOR_MAP
        global DIM_MAP
        # L1 fusion write select

        self._print_debug("dx fusion tag:", res.op.tag)
        tensor_dx_gm, TENSOR_MAP = _check_dx_fusion_type(res, TENSOR_MAP)
        _get_l1_fusion_para()
        fusion_type = self.dx_para.get_para_map("FUSION_TYPE")
        var_map = _get_var_map(var_range)

        # get tensor of ub by fusion_type
        tensor_cub = _get_ub_tensor(fusion_type)
        no_need_use_ub_flag = (tensor_cub is None)
        self.dx_para.update_para_map("no_need_use_ub_flag", no_need_use_ub_flag)

        if fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
            c_ub_img = tensor_cub.op.input_tensors[0]
            sch[c_ub_img].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
            sch[c_ub_img].compute_inline()
            c_ub = c_ub_img.op.input_tensors[0]
            sch[c_ub].compute_inline()
            tensor_cub = c_ub

        if tensor_cub is not None:
            if tensor_cub.op.input_tensors[0].name == "c_add_bias":
                c_add_bias = tensor_cub.op.input_tensors[0]
                bias_l0c = c_add_bias.op.input_tensors[0]
                tensor_mmad = c_add_bias.op.input_tensors[1]
                bias_ub_brc = bias_l0c.op.input_tensors[0]
                tensor_bias = bias_ub_brc.op.input_tensors[0]
                bias_ub = sch.cache_read(tensor_bias, tbe_platform_info.scope_ubuf, [bias_ub_brc])
                TENSOR_MAP["c_add_bias"] = c_add_bias
                TENSOR_MAP["bias_l0c"] = bias_l0c
                TENSOR_MAP["bias_ub_brc"] = bias_ub_brc
                TENSOR_MAP["bias_ub"] = bias_ub
                TENSOR_MAP["tensor_bias"] = tensor_bias
            else:
                tensor_mmad = tensor_cub.op.input_tensors[0]
        else:
            if tensor_dx_gm.op.input_tensors[0].name == 'c_add_bt':
                c_add_bias = tensor_dx_gm.op.input_tensors[0]
                bias_bt = c_add_bias.op.input_tensors[2]
                bias_ddr = bias_bt.op.input_tensors[0]
                bias_l1 = sch.cache_read(bias_ddr, tbe_platform_info.scope_cbuf, [bias_bt])
                TENSOR_MAP["c_add_bt"] = c_add_bias
                TENSOR_MAP["bias_bt"] = bias_bt
                TENSOR_MAP["bias_l1"] = bias_l1
            if self.dx_para.get_para_map("no_need_use_ub_flag"):
                tensor_mmad = tensor_dx_gm.op.input_tensors[0]

        a_l0a = tensor_mmad.op.input_tensors[0]
        dedy_col, dedy, a_l0a_before = _al1_fusion_handle()
        weight_l0 = tensor_mmad.op.input_tensors[1]
        weight_l1_ori = weight_l0.op.input_tensors[0]
        weight_l1 = weight_l1_ori.op.input_tensors[0]
        input_fp32_flag = (a_l0a.dtype == "float32" and weight_l0.dtype == "float32")
        self.dx_para.update_para_map("input_fp32_flag", input_fp32_flag)

        if weight_l1.op.input_tensors:
            if "NHWC_trans_FZ" in weight_l1.op.tag:
                self.dx_para.update_para_map("WEIGHT_NHWC_TRANS_FZ", True)
                sch[weight_l1_ori].compute_inline()
        else:
            weight_l1 = weight_l1_ori

        # set scope
        if fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
            tensor_cub = TENSOR_MAP.get("c_ub")
        if tensor_cub is not None:
            sch[tensor_cub].set_scope(tbe_platform_info.scope_ubuf)
            TENSOR_MAP["c_ub"] = tensor_cub

        # TO DO
        TENSOR_MAP["img_placehold"] = dedy
        DIM_MAP["img_shape"] = cube_util.shape_to_list(TENSOR_MAP["img_placehold"].shape)

        _al1_buffer_align()

        TENSOR_MAP["a_l1"] = dedy_col
        sch[a_l0a].set_scope(tbe_platform_info.scope_ca)

        l0a_m0, l0a_k0, _ = tbe_platform.CUBE_MKN.get(a_l0a.dtype)["mac"]
        sch[a_l0a].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, l0a_m0), (1, l0a_k0))
        TENSOR_MAP["a_l0a"] = a_l0a
        if TENSOR_MAP.get("c_add_bias") is not None:
            sch[c_add_bias].set_scope(tbe_platform_info.scope_cc)
            sch[bias_l0c].set_scope(tbe_platform_info.scope_cc)
            sch[bias_ub_brc].set_scope(tbe_platform_info.scope_ubuf)
        if TENSOR_MAP.get("c_add_bt") is not None:
            sch[c_add_bias].set_scope(tbe_platform_info.scope_cc)
            sch[bias_l1].set_scope(tbe_platform_info.scope_cbuf)
            sch[bias_bt].set_scope("local.BT")
        sch[weight_l1].set_scope(tbe_platform_info.scope_cbuf)
        TENSOR_MAP["b_l1"] = weight_l1
        sch[weight_l0].set_scope(tbe_platform_info.scope_cb)
        TENSOR_MAP["b_l0b"] = weight_l0

        sch[tensor_mmad].set_scope(tbe_platform_info.scope_cc)
        TENSOR_MAP["c_l0c"] = tensor_mmad
        TENSOR_MAP["c_gm"] = dex_res
        TENSOR_MAP["filter_placehold"] = weight_l1.op.input_tensors[0]

        # fill in dimmap
        DIM_MAP["group_dict"] =  tensor_dx_gm.op.attrs["group_dict"]
        group_dict_map = DIM_MAP.get("group_dict")
        DIM_MAP[cube_util.GroupDictKeys.g_extend] = group_dict_map[cube_util.GroupDictKeys.g_extend].value
        DIM_MAP[cube_util.GroupDictKeys.dy_c1_extend] = group_dict_map[cube_util.GroupDictKeys.dy_c1_extend].value
        DIM_MAP[cube_util.GroupDictKeys.dx_c1_extend] = group_dict_map[cube_util.GroupDictKeys.dx_c1_extend].value
        if self.dx_para.get_para_map("5HD_TRANS_NHWC"):
            res_ori = res.op.input_tensors[0]
            DIM_MAP["out_img_shape"] = cube_util.shape_to_list(res_ori.shape)
        else:
            DIM_MAP["out_img_shape"] = cube_util.shape_to_list(res.shape)
        if self.dx_para.get_para_map("FM_NHWC_TRANS_5HD"):
            DIM_MAP["img_shape"] = cube_util.shape_to_list(TENSOR_MAP.get("a_l1").shape)
        else:
            DIM_MAP["img_shape"] = cube_util.shape_to_list(TENSOR_MAP.get("img_placehold").shape)
        DIM_MAP["A_matrix_dim"] = cube_util.shape_to_list(dedy_col.shape)
        DIM_MAP["B_matrix_dim"] = cube_util.shape_to_list(weight_l0.shape)
        DIM_MAP["filter_shape"] = cube_util.shape_to_list(weight_l1.op.input_tensors[0].shape)
        DIM_MAP["dx_5D_shape"] = cube_util.shape_to_list(tensor_dx_gm.op.attrs["dx_5D_shape"])
        DIM_MAP["dx_6GD_shape"] = [DIM_MAP.get(cube_util.GroupDictKeys.g_extend),
                                   DIM_MAP.get("dx_5D_shape")[0],
                                   DIM_MAP.get(cube_util.GroupDictKeys.dx_c1_extend)] + DIM_MAP.get("dx_5D_shape")[2:]
        DIM_MAP["dy_6GD_shape"] = [DIM_MAP.get(cube_util.GroupDictKeys.g_extend),
                                   DIM_MAP.get("img_shape")[0],
                                   DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend)] + DIM_MAP.get("img_shape")[2:]
        if weight_l0.dtype == "float32":
            DIM_MAP.get("dy_6GD_shape")[2] *= 2

        if TENSOR_MAP.get("dilate_ub") is not None:
            DIM_MAP["dilate_dim"] = cube_util.shape_to_list(TENSOR_MAP.get("dilate_ub").op.attrs["dilate"])
            DIM_MAP["out_hwdim"] = cube_util.shape_to_list(TENSOR_MAP.get("dilate_ub").op.attrs["out_hw"])

        if "batch_n" in var_map:
            sch.set_var_range(DIM_MAP.get("img_shape")[0], *var_range.get("batch_n"))
            sch.set_var_range(cube_util.shape_to_list(tensor_dx_gm.op.attrs["dx_5D_shape"])[0],
                              *var_range.get("batch_n"))
        if "dedy_h" in var_map:
            sch.set_var_range(DIM_MAP.get("img_shape")[2], *var_range.get("dedy_h"))
            sch.set_var_range(cube_util.shape_to_list(tensor_dx_gm.op.attrs["dx_5D_shape"])[2], *var_range.get("dx_h"))
        if "dedy_w" in var_map:
            sch.set_var_range(DIM_MAP.get("img_shape")[3], *var_range.get("dedy_w"))
            sch.set_var_range(cube_util.shape_to_list(tensor_dx_gm.op.attrs["dx_5D_shape"])[3], *var_range.get("dx_w"))

        return tensor_dx_gm, var_map


    def _get_aicore_tiling_factor(self, is_conv1d_bool, sch, var_map, var_range, l0c_multi_group_flag):
        """
        using tilling parameter calculate factor

        :return: tilling factor from ub to ddr
            tilling factor from l0c to ub
            tilling factor from ddr to AL1
            tilling factor from ddr to Bl1
        """

        def _get_undilate_loc_m(l0c_tiling_factor):

            if l0c_tiling_factor[1] < DIM_MAP.get("img_shape")[-2]:
                self._raise_dx_opti_err("mc of CL0_matrix small than weight of Image")
            if DIM_MAP.get("img_shape")[3] > block_m:
                check_ifmc_falg = bool(
                    (mc_from_tiling // DIM_MAP.get("img_shape")[3])
                    * DIM_MAP.get("img_shape")[3]
                    * DIM_MAP.get("dilate_dim")[0]
                    * DIM_MAP.get("dilate_dim")[1]
                    <= CUB_BUFFER_LIMIT
                )
                if (
                    mc_from_tiling % DIM_MAP.get("img_shape")[3] == 0
                    and check_ifmc_falg
                    and DIM_MAP.get("img_shape")[2]
                    % (mc_from_tiling // DIM_MAP.get("img_shape")[3])
                    == 0
                ):
                    n_is_hfactor = (mc_from_tiling) // DIM_MAP.get("img_shape")[3]
                else:
                    n_is_hfactor = (mc_from_tiling - block_m) // DIM_MAP.get("img_shape")[3]
            else:
                check_ifmc_falg_s = False
                if mc_from_tiling % DIM_MAP.get("img_shape")[3] == 0:
                    n_is_hfactor = mc_from_tiling // DIM_MAP.get("img_shape")[3]
                    while DIM_MAP.get("img_shape")[2] % n_is_hfactor != 0:
                        n_is_hfactor = n_is_hfactor - 1
                    check_ifmc_falg_s = bool(
                        n_is_hfactor
                        * DIM_MAP.get("img_shape")[3]
                        * DIM_MAP.get("dilate_dim")[0]
                        * DIM_MAP.get("dilate_dim")[1]
                        > CUB_BUFFER_LIMIT
                    )
                if mc_from_tiling % DIM_MAP.get("img_shape")[3] != 0 or check_ifmc_falg_s:
                    n_is_hfactor = max((mc_from_tiling - block_m), block_m) // DIM_MAP.get("img_shape")[3]
                    while DIM_MAP.get("img_shape")[2] % n_is_hfactor != 0:
                        n_is_hfactor = n_is_hfactor - 1

            l0c_tiling_factor[1] = (
                DIM_MAP.get("out_hwdim")[1] * n_is_hfactor * DIM_MAP.get("dilate_dim")[0]
            )
            if l0c_tiling_factor[1] == 0:
                self._raise_dx_opti_err("nw can not be zero")
            undilate_l0c_m = n_is_hfactor * DIM_MAP.get("img_shape")[3]
            return undilate_l0c_m

        def _get_undilate_loc_m_dynamic(l0c_tiling_factor, sch, var_range):
            n_is_hfactor = tvm.var("n_is_hfactor")
            dedy_h, dedy_w = DIM_MAP.get("img_shape")[2], DIM_MAP.get("img_shape")[3]
            nc_factor, mc_factor, m0, n0 = TILING.get("CUB_matrix")[:4]
            cub_db_flag = TILING.get("manual_pingpong_buffer").get("CUB_pbuffer")
            cub_dtype_bit = DTYPE_BYTE_MAP.get("float16")
            ub_buffer_size = tbe_platform_info.get_soc_spec("UB_SIZE")
            stride_h, stride_w = DIM_MAP.get("dilate_dim")
            _, dx_w = DIM_MAP.get("out_hwdim")

            if "dedy_w" in var_map and not var_map.get("dedy_w")[1]:
                sch.set_var_value(n_is_hfactor, (mc_from_tiling - block_m) // DIM_MAP.get("img_shape")[3])
            else:
                if "dedy_h" not in var_map:
                    dedy_h = tvm.var("dedy_h")
                    sch.set_var_value(dedy_h, DIM_MAP.get("img_shape")[2])
                if "dedy_w" not in var_map:
                    dedy_w = tvm.var("dedy_w")
                    sch.set_var_value(dedy_w, DIM_MAP.get("img_shape")[3])
                check_ifmc_flag = (mc_from_tiling // dedy_w * dedy_w * stride_h * stride_w <= CUB_BUFFER_LIMIT)

                max_n_is_hfactor_bound = tvm.floordiv(
                    (ub_buffer_size // cub_db_flag // cub_dtype_bit - nc_factor * mc_factor * m0 * n0)
                    // (nc_factor * n0 * stride_h), dx_w)
                max_n_is_hfactor_offset = tvm.floordiv((ub_buffer_size // cub_dtype_bit // m0 -
                    (stride_w + dx_w * stride_w)), (dx_w * stride_w))
                max_n_is_hfactor = tvm.min(max_n_is_hfactor_bound, max_n_is_hfactor_offset)

                if ("dedy_w" in var_map and mc_from_tiling >= var_range["dedy_w"][0]
                    or "dedy_w" not in var_map and mc_from_tiling >= DIM_MAP.get("img_shape")[3]):
                    sch.set_var_value(n_is_hfactor,
                                    tvm.max(tvm.min(tvm.select(
                                        tvm.all(mc_from_tiling // dedy_w * dedy_w == mc_from_tiling,
                                                check_ifmc_flag,
                                                dedy_h % (mc_from_tiling // dedy_w) == 0),
                                                mc_from_tiling // DIM_MAP.get("img_shape")[3],
                                                (mc_from_tiling - block_m) // DIM_MAP.get("img_shape")[3]),
                                        max_n_is_hfactor), 1))
                else:
                    sch.set_var_value(n_is_hfactor,
                                      tvm.max((mc_from_tiling - block_m) // DIM_MAP.get("img_shape")[3], 1))
                sch.set_var_range(n_is_hfactor, 1, mc_from_tiling)
            l0c_tiling_factor[1] = dx_w * n_is_hfactor * stride_h
            undilate_l0c_m = n_is_hfactor * DIM_MAP.get("img_shape")[3]
            return undilate_l0c_m

        def _get_al1_and_bl1_parts(l0c_parts):
            al1_parts = [1, 1, 1]
            if TILING["AL1_shape"]:  # AL1_shape = [C1,H*W,16,16],batch=1
                # parts of k-axis from DDR to L1---need div by H*W
                factor = 2 if TENSOR_MAP.get("a_l1").dtype == "float32" else 1
                al1_parts = [
                    int_ceil_div(
                        DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend) * factor,
                        int_ceil_div(TILING["AL1_shape"][0], block_k)
                    ),
                    int_ceil_div(l0c_parts[1], TILING["AL1_shape"][1]),
                    # factor of group
                    TILING["AL1_shape"][3]
                ]

            bl1_parts = [1, 1, 1]
            if TILING["BL1_shape"]:
                if (l0c_parts[0] % TILING["BL1_shape"][1]) != 0:
                    self._raise_dx_opti_err(
                        "second value of BL1_shape should be factor of n block num"
                    )
                bl1_parts = [
                    int_ceil_div(
                        DIM_MAP.get("B_matrix_dim")[1],
                        int_ceil_div(TILING["BL1_shape"][0], block_k)
                    ),
                    int_ceil_div(l0c_parts[0], TILING["BL1_shape"][1]),
                    # factor of group
                    TILING["BL1_shape"][3]
                ]
            return al1_parts, bl1_parts

        block_m = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("c_l0c").dtype)["mac"][0]
        block_k = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("b_l1").dtype)["mac"][1]
        # get factor from l0c, ub to ddr
        mc_from_tiling = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]
        l0c_tiling_factor = [TILING["CL0_matrix"][0], mc_from_tiling]
        if TENSOR_MAP.get("c_gm").dtype == "float32" and TENSOR_MAP.get("b_l1").dtype == "float32":
            l0c_tiling_factor[0] *= 2
        undilate_l0c_m = (mc_from_tiling // DIM_MAP.get("img_shape")[3]) * DIM_MAP.get("img_shape")[3]

        need_buffer_tile = False
        if DIM_MAP.get("dilate_dim") is not None:
            # get fh*w(fh is factor of H),
            # and update l0c_tiling_factor[1] to dilate*fh*w
            if is_conv1d_bool:
                l0c_tiling_factor[1] = mc_from_tiling * DIM_MAP.get("dilate_dim")[1]
                undilate_l0c_m = mc_from_tiling
                need_buffer_tile = True
            else:
                if "dedy_h" in var_map or "dedy_w" in var_map:
                    undilate_l0c_m = _get_undilate_loc_m_dynamic(l0c_tiling_factor, sch, var_range)
                else:
                    undilate_l0c_m = _get_undilate_loc_m(l0c_tiling_factor)
                if undilate_l0c_m % block_m != 0:
                    need_buffer_tile = True

        mask_ub_need_bind_buffer = False
        if TENSOR_MAP.get("drelu_ub") is not None:
            if (
                l0c_tiling_factor[1] % block_m != 0
                or (
                    DIM_MAP.get("out_img_shape")[2] -
                    DIM_MAP.get("out_img_shape")[2] // l0c_tiling_factor[1] * l0c_tiling_factor[1]
                ) % block_m != 0
            ):
                self._print_debug("Tiling[CUB_matrix][0] reset to 1, mask_ub_need_bind_buffer")
                mask_ub_need_bind_buffer = True
                TILING["CUB_matrix"][0] = 1

        # From LOC to GM [NumOfparts for N axis, NumOfparts for M axis ]
        l0c_parts = [
            int_ceil_div(
                DIM_MAP.get("dx_6GD_shape")[2] // TILING["block_dim"][1],
                l0c_tiling_factor[0]
            ),
            int_ceil_div(
                DIM_MAP.get("out_img_shape")[2] // TILING["block_dim"][2],
                l0c_tiling_factor[1]
            )
        ]

        if l0c_multi_group_flag:
            # used to calculate bl1_parts
            l0c_parts[0] = int_ceil_div(DIM_MAP.get("out_img_shape")[1] // TILING["block_dim"][3], l0c_tiling_factor[0])
        if "dedy_h" in var_map or "dedy_w" in var_map:
            l0c_parts[1] = int_ceil_div(
                int_ceil_div(DIM_MAP.get("out_img_shape")[2], TILING["block_dim"][2]),
                l0c_tiling_factor[1]
            )

        # C_UB_factor is size of each part from l0c to ub, second item is 1
        # From L0C to UB,[NumOfparts for N axis, NumOfparts for M axis]
        l0c_ub_parts = [
            int_ceil_div(l0c_tiling_factor[0], TILING["CUB_matrix"][0]),
            int_ceil_div(
                l0c_tiling_factor[1], TILING["CUB_matrix"][1] * TILING["CUB_matrix"][2]
            )
        ]
        self._print_debug(
            "l0c_ub_tiling_factor:", TILING["CUB_matrix"], "l0c_ub_parts:", l0c_ub_parts
        )

        al1_parts, bl1_parts = _get_al1_and_bl1_parts(l0c_parts)

        return (
            l0c_tiling_factor,
            l0c_ub_parts,
            al1_parts,
            bl1_parts,
            undilate_l0c_m,
            need_buffer_tile,
            mask_ub_need_bind_buffer
        )


    @staticmethod
    def _get_mmad_factor():
        """
        get tilling factor in mmad

        :return:tilling factor for al0
                tilling factor for bl0
                tilling factor for reduce axis
        """
        al0_factor = [TILING.get("AL0_matrix")[0], TILING.get("AL0_matrix")[1]]
        bl0_factor = [TILING.get("BL0_matrix")[0], TILING.get("BL0_matrix")[1], TILING.get("BL0_matrix")[5]]
        reduce_factor = TILING.get("AL0_matrix")[1]
        return al0_factor, bl0_factor, reduce_factor


    def _bind_multi_core(
        self,
        sch,
        c_gm,
        g_dim,
        l1_n_outer_outer,
        l1_n_out_inner,
        l1_m_outer_outer,
        l1_m_outer_inner,
        var_map
    ):
        if "block_dim" in TILING and not self.dx_para.get_para_map("load3d_flag"):
            block_dim = TILING["block_dim"]
        else:
            block_dim = [1, 1, 1, 1]
        blockidx_list = []
        # split batch axis
        if "batch_n" in var_map:
            batch_dim_factor = int_ceil_div(DIM_MAP.get("out_img_shape")[0], block_dim[0])
            batch_dim_factor = tvm.max(1, batch_dim_factor)
            batch_out, batch_in = sch[c_gm].split(c_gm.op.axis[0], batch_dim_factor)
        else:
            batch_out, batch_in = sch[c_gm].split(c_gm.op.axis[0], nparts=block_dim[0])
        g_outer, g_inner = sch[c_gm].split(g_dim, nparts=block_dim[3])
        l1_n_out_inner_out, l1_n_out_inner_in = sch[c_gm].split(
            l1_n_out_inner, nparts=block_dim[1]
        )
        l1_m_outer_inner_out, l1_m_outer_inner_in = sch[c_gm].split(
            l1_m_outer_inner, nparts=block_dim[2]
        )

        # reorder
        sch[c_gm].reorder(
            g_outer,
            batch_out,
            l1_n_out_inner_out,
            l1_m_outer_inner_out,
            g_inner,
            batch_in,
            l1_n_outer_outer,
            l1_n_out_inner_in,
            l1_m_outer_outer,
            l1_m_outer_inner_in
        )

        blocks = block_dim[0] * block_dim[1] * block_dim[2] * block_dim[3]
        if blocks != 1:
            out_fused = sch[c_gm].fuse(g_outer, batch_out, l1_n_out_inner_out, l1_m_outer_inner_out)
            out_fused_out, _ = sch[c_gm].split(out_fused, nparts=blocks)
            bind_out, bind_in = sch[c_gm].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_gm].bind(bind_out, blockidx)
            if blocks == block_dim[0]:
                sch[c_gm].pragma(bind_in, "json_info_batchBindOnly")
        blockidx_list = [g_outer, batch_out, l1_n_out_inner_out, l1_m_outer_inner_out]
        return [batch_in, g_inner, l1_m_outer_inner_in, l1_n_out_inner_out, l1_n_out_inner_in, blockidx_list]


    def _get_l0c_and_l1_axis(
        self, sch, c_gm, l0c_factor, al1_parts, bl1_parts, num_batch, dx_c1_extend, var_map, l0c_multi_group_flag
    ):
        """
        get l0c and l1 axis

        Parameter:
        ------------------------------------------------------------------
        :param sch: schedule
        :param c_gm: op
        :param l0c_factor: tilling factor for l0c
        :param al1_parts: tilling factor for al1
        :param bl1_parts: tilling factor for bl1
        :param batch_in_axis: tilling factor for batch
        -------------------------------------------------------------------
        """

        def _get_reorder_flag(al1_parts, bl1_parts):
            reorder_flag = False
            if (
                TILING["AL1_shape"]
                and al1_parts[0] != 1
                and TILING["BL1_shape"]
                and bl1_parts[0] != 1
            ):
                if bl1_parts[1] >= al1_parts[1]:
                    reorder_flag = True
            if (
                TILING["AL1_shape"]
                and al1_parts[0] == 1
                and TILING["BL1_shape"]
                and bl1_parts[0] == 1
            ):
                if bl1_parts[1] >= al1_parts[1]:
                    reorder_flag = True
            if (
                TILING["BL1_shape"]
                and bl1_parts[0] != 1
                and TILING["AL1_shape"]
                and al1_parts[0] == 1
            ):
                reorder_flag = True
            self._print_debug("reorder_flag:", reorder_flag)
            return reorder_flag

        if self.dx_para.get_para_map("5HD_TRANS_NHWC"): # (batch, howo_axis, co_axis)
            c_gm_howo_axis_idx = 1
            c_gm_co_axis_idx = 2
        else:                                           # (batch, co_aixs//16, howo_axis, 16)
            c_gm_howo_axis_idx = 2
            c_gm_co_axis_idx = 1

        # split c_gm according to factor of loc and out_shape
        if not l0c_multi_group_flag:
            g_dim, c_gm_inner = sch[c_gm].split(c_gm.op.axis[c_gm_co_axis_idx], dx_c1_extend)
        else:
            g_dim, c_gm_inner = sch[c_gm].split(c_gm.op.axis[c_gm_co_axis_idx], l0c_factor[0])

        if self.dx_para.get_para_map("5HD_TRANS_NHWC"):
            l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm_inner, l0c_factor[0] * 16)
        else:
            l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm_inner, l0c_factor[0])

        l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[c_gm_howo_axis_idx], l0c_factor[1])
        sch[c_gm].reorder(g_dim, c_gm.op.axis[0], l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)

        # split c_gm according to factor of a_l1 and b_l1
        l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(
            l0c_m_outer, nparts=al1_parts[1]
        )
        l1_n_outer_outer, l1_n_out_inner = sch[c_gm].split(l0c_n_outer, nparts=bl1_parts[1])
        self._print_ir_conv("split gm by loc_factor and l1_parts", sch)
        [batch_in, g_inner, l1_m_outer_inner_in, l1_n_out_inner_out,
        l1_n_out_inner_in, blockidx_list] = self._bind_multi_core(sch, c_gm, g_dim, l1_n_outer_outer, l1_n_out_inner,
                                                                  l1_m_outer_outer, l1_m_outer_inner, var_map)
        self._print_ir_conv("bind multi core", sch)
        # reorder al1 and bl1 axis according to double buffer
        batch_in_out_axis, batch_in_inner_axis = sch[c_gm].split(batch_in, factor=1)

        # m or n reorder flag, if m_outer is smaller, reorder is true
        reorder_flag = False
        if not var_map:
            reorder_flag = _get_reorder_flag(al1_parts, bl1_parts)
        self._print_ir_conv("before reorder", sch)
        if reorder_flag:
            sch[c_gm].reorder(l1_m_outer_outer, batch_in_inner_axis, l1_n_outer_outer)
            overload_axis = l1_m_outer_outer
            overload_flag_gm = False
        else:
            sch[c_gm].reorder(l1_n_outer_outer, l1_m_outer_outer, batch_in_inner_axis)
            overload_axis = l1_n_outer_outer
            overload_flag_gm = True
        self._print_ir_conv("after reorder", sch)

        return (batch_in_out_axis, l1_n_outer_outer, batch_in_inner_axis, l1_n_out_inner_out, l1_m_outer_inner_in,
                l0c_n_inner, l0c_m_inner, l1_m_outer_outer, l1_n_out_inner_in, blockidx_list, g_inner, overload_axis,
                overload_flag_gm, l0c_m_outer, reorder_flag)


    @staticmethod
    def _get_l0a_and_l0b_axis(
        sch, c_l0c, new_c_col_axis, al0_axis_factor, bl0_axis_factor, reduce_axis_factor
    ):
        """
        get l0a and l0b axis
        Parameter:
        ---------------------------------------------------------------
        :param sch: schedule
        :param c_l0c: op
        :param new_c_col_axis:
        :param al0_axis_factor:
        :param bl0_axis_factor:
        :param reduce_axis_factor:
        :return:
        ---------------------------------------------------------------
        """
        # split and get axis of reduce, al0_at_axis, bl0_at_axis
        reduce_out, reduce_inner = sch[c_l0c].op.reduce_axis
        al0_m_out, al0_m_inner = sch[c_l0c].split(
            new_c_col_axis[2],
            al0_axis_factor[0] * tbe_platform.CUBE_MKN.get(c_l0c.dtype)["mac"][0]
        )
        bl0_n_outer, bl0_n_inner = sch[c_l0c].split(new_c_col_axis[1], bl0_axis_factor[1])
        # for reduce axis, al0 and b_l0b should be the same
        k_outer_outer, k_outer_inner = sch[c_l0c].split(reduce_out, reduce_axis_factor)
        _, batch_l0c_inner = sch[c_l0c].split(c_l0c.op.axis[1], 1)
        sch[c_l0c].reorder(
            k_outer_outer,
            bl0_n_outer,
            al0_m_out,
            batch_l0c_inner,
            bl0_n_inner,
            al0_m_inner,
            new_c_col_axis[3],
            k_outer_inner,
            reduce_inner
        )

        return al0_m_out, bl0_n_outer, k_outer_outer, batch_l0c_inner


    def _get_al1_and_bl1_axis(self, sch, c_l0c, al1_parts, bl1_parts, k_outer_outer):
        """
        get al1 and bli axis
        Parameter:
        ---------------------------------------------------------------
        :param sch: schedule
        :param c_l0c: op
        :param al1_parts:
        :param bl1_parts:
        :param k_outer_outer:
        :return:
        ---------------------------------------------------------------
        """
        #  ============ a_l1 and b_l1 slice can be different with CUB & CL0 =====
        outer_factor = max(int(al1_parts[0]), int(bl1_parts[0]))
        inner_factor = min(int(al1_parts[0]), int(bl1_parts[0]))
        if outer_factor % inner_factor != 0:
            self._raise_dx_opti_err("illegal value of AL1_shape & BL1_shape")

        if int(al1_parts[0]) > int(bl1_parts[0]):
            k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer, nparts=al1_parts[0]
            )
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer_outer, nparts=(bl1_parts[0])
            )
            al1_at_l0c_axis = k_outer_outer_outer_inner
            bl1_at_l0c_axis = k_outer_outer_outer_outer
            overload_flag_l0c = True
        else:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer, nparts=bl1_parts[0]
            )
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer_outer, nparts=(al1_parts[0])
            )
            al1_at_l0c_axis = k_outer_outer_outer_outer
            bl1_at_l0c_axis = k_outer_outer_outer_inner
            overload_flag_l0c = False
        reduce_axis_serial = [
            k_outer_outer_outer_outer,
            k_outer_outer_outer_inner,
            k_outer_outer_inner
        ]
        return al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial, overload_flag_l0c


    @staticmethod
    def _dilate_schedule(sch, dilate_ub, out_w, dilate_w, dilate_h, var_map=None):
        """
        :param sch:
        :param dilate_ub:
        :param out_h:
        :param out_w:
        :param dilate_h:
        :param dilate_w:
        :return:
        """
        h_axis, w_axis = sch[dilate_ub].split(dilate_ub.op.axis[2], out_w)
        wo_axis, wi_axis = sch[dilate_ub].split(w_axis, dilate_w)
        ho_axis, hi_axis = sch[dilate_ub].split(h_axis, dilate_h)

        sch[dilate_ub].reorder(
            wi_axis, dilate_ub.op.axis[0], dilate_ub.op.axis[1], ho_axis, wo_axis
        )
        if not var_map:
            sch[dilate_ub].unroll(hi_axis)
            sch[dilate_ub].unroll(wi_axis)
        else:
            sch[dilate_ub].set_store_predicate(tvm.all(wi_axis.var == 0, hi_axis.var == 0))
        return wo_axis


    def _check_quant_fusion_legal(self, fusion_type):
        """
        In the two fusion scenes, l0b, ub and l0c has the minimum data load,
        Any buffer hyperspace, the fusion backs to single operator
        :param fusion_type:  FUSION_DX_DEQUANT_QUANT or FUSION_DX_REQUANT

        """
        block_m, block_k, block_n = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("b_l1").dtype)["mac"]
        min_n = DIM_MAP.get("dx_c1_extend")
        min_l0b = min_n * block_k * block_n * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("b_l0b").dtype)
        if min_l0b > tbe_platform_info.get_soc_spec("L0B_SIZE"):
            self._raise_dx_opti_err("The minimum data in l0b exceed buffer space, the fusion backs to single operator.")
        min_m = 1
        if TENSOR_MAP.get("dilate_ub") is not None:
            # when mmad, the min unit of M is a fmap's w
            dy_w = DIM_MAP.get("img_shape")[3]
            # mc's calculation rule refor to auto tiling
            if dy_w % 16 == 0:
                min_m = dy_w // block_m
            else:
                # add one is needed by buffer_tile of ub
                min_m = int_ceil_div(dy_w, block_m) + 1
        group_l0c = 2
        min_l0c = min_m * min_n * block_n * block_m * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("c_l0c").dtype) * group_l0c
        if min_l0c > tbe_platform_info.get_soc_spec("L0C_SIZE"):
            self._raise_dx_opti_err("The minimum data in l0c exceed buffer space, the fusion backs to single operator.")
        self._calc_double_op_num(fusion_type)
        min_ub = min_m * min_n * block_n * block_m * DTYPE_BYTE_MAP.get(TENSOR_MAP.get("c_gm").dtype) \
                * group_l0c * FUSION_TYPE_2_OPERAND_NUM.get(fusion_type)
        if min_ub > tbe_platform_info.get_soc_spec("UB_SIZE"):
            self._raise_dx_opti_err("The minimum data in ub exceed buffer space, the fusion backs to single operator.")


    def opti_schedule(self, tensor, sch_list, tiling_case=None, var_range=None):
        """
        the schedule of conv2d_backprop_opti_input

        Parameter:
        -------------------------------------------------------------------
        :param tensor: a tensor
        :param sch_list: a schedule list
        :param tiling_case: fix tiling for dynamic shape
        :param var_range: var_range for dynamic shape
        :return: schedule
        -------------------------------------------------------------------
        """

        def _do_buffer_tile():
            def _get_cub_buffertile_m_min():
                # cub buffertile hw axis
                block_m = tbe_platform.CUBE_MKN.get(TENSOR_MAP.get("c_ub").dtype)["mac"][0]
                mm_coefficient_factor = undilate_l0c_m
                moo_coefficient_unzero = int_ceil_div(
                    int_ceil_div(DIM_MAP.get("out_img_shape")[2], l0c_factor[1]), al1_parts[1]
                )

                moo_coefficient = 0 if al1_parts[1] == 1 else moo_coefficient_unzero
                moio_coefficient = (
                    0
                    if TILING["block_dim"][2] == 1
                    else int_ceil_div(moo_coefficient_unzero, TILING["block_dim"][2])
                )
                moii_coefficient = (
                    0
                    if int_ceil_div(moo_coefficient_unzero, TILING["block_dim"][2]) == 1
                    else 1
                )
                cub_buffertile_m_min = (
                    (
                        moo_coefficient * tile_axis.var
                        + moio_coefficient * moio_axis
                        + moii_coefficient * c_slice_axis
                    )
                    * mm_coefficient_factor
                    // block_m
                    * block_m
                )
                return cub_buffertile_m_min

            def _get_cub_buffertile_n_min():
                # cub buffertile cout axis
                no_coefficient = l0c_factor[0]
                noo_coefficient_unzero = int_ceil_div(
                    int_ceil_div(DIM_MAP.get("dx_6GD_shape")[2], l0c_factor[0]), bl1_parts[1]
                )
                noo_coefficient = 0 if bl1_parts[1] == 1 else noo_coefficient_unzero
                noio_coefficient = (
                    0
                    if TILING["block_dim"][1] == 1
                    else int_ceil_div(noo_coefficient_unzero, TILING["block_dim"][1])
                )
                go_coefficient = (
                    0
                    if TILING["block_dim"][3] == 1
                    else int_ceil_div(DIM_MAP.get("dx_6GD_shape")[0], TILING.get("block_dim")[3])
                )
                gi_coefficient = (
                    0
                    if DIM_MAP.get("dx_6GD_shape")[0] == 1
                    else 1
                )
                noii_coefficient = (
                    0
                    if int_ceil_div(noo_coefficient_unzero, TILING["block_dim"][1]) == 1
                    else 1
                )
                nio_coefficient = (
                    0
                    if l0c_ub_parts[0] == 1
                    else int_ceil_div(l0c_factor[0], l0c_ub_parts[0])
                )
                cub_buffertile_n_min = (
                    go_coefficient * group_axis
                    + gi_coefficient * g_inner) * DIM_MAP.get("dx_6GD_shape")[2] + (
                    bl1_at_c_axis.var * noo_coefficient
                    + noio_coefficient * noio_axis
                    + noii_coefficient * noii_axis.var
                ) * no_coefficient + nio_coefficient * l0c_n_inner_outer.var
                return cub_buffertile_n_min

            l0c_factor_tile = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]

            # multi core and one core
            group_axis, batcho_axis, noio_axis, moio_axis = blockidx_list
            # cub buffertile batch axis
            batch_factor = int_ceil_div(DIM_MAP.get("img_shape")[0], TILING.get("block_dim")[0])
            batcho_coefficient = 0 if TILING["block_dim"][0] == 1 else batch_factor
            batchio_coefficient = 0 if batch_factor == 1 else 1
            batch_dim = [
                batcho_axis * batcho_coefficient
                + batch_in_out_axis.var * batchio_coefficient,
                1
            ]
            cub_buffertile_m_min = _get_cub_buffertile_m_min()
            cub_buffertile_m_extend = l0c_factor_tile

            cub_buffertile_n_min = _get_cub_buffertile_n_min()
            cub_buffertile_n_extend = TILING["CUB_matrix"][0]

            sch[c_ub].buffer_tile(
                (batch_dim[0], batch_dim[1]),
                (cub_buffertile_n_min, cub_buffertile_n_extend),
                (cub_buffertile_m_min, cub_buffertile_m_extend),
                (0, 16)
            )

        def _attach_ub(fusion_type):
            def _attach_ub_quant():
                ub_attach_list = [
                    "input_ub",
                    "deq",
                    "reform_op",
                    "cast_i8_ub",
                    "data_transfer"
                ]
                for tensor in TENSOR_MAP:
                    if tensor == "elewise_tensor":
                        for elewise_tensor in TENSOR_MAP.get(tensor):
                            if "broadcast" in elewise_tensor.op.tag or elewise_tensor.op.tag == "dequant_remove_pad":
                                sch[elewise_tensor].compute_inline()
                            else:
                                sch[elewise_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
                    elif tensor == "input_tensor":
                        for input_tensor in TENSOR_MAP.get(tensor):
                            sch[input_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
                    elif tensor == "input_ub" and not quan_para.get("q_padding"):
                        sch[TENSOR_MAP.get(tensor)].compute_inline()
                    elif tensor in ub_attach_list:
                        sch[TENSOR_MAP.get(tensor)].compute_at(sch[c_gm], l0c_m_inner_outer)

            def _attach_ub_bias():
                split_bias_flag = TILING.get("CUB_channel_wise_flag")
                if bias_add_vector_ub is not None:
                    if split_bias_flag:
                        sch[bias_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                    else:
                        sch[bias_ub].compute_at(sch[c_gm], batch_in_out_axis)
                    sch[bias_add_vector_ub].compute_at(sch[c_gm], l0c_m_inner_outer)

                if bias_ub_brc is not None:
                    if split_bias_flag:
                        sch[bias_ub].compute_at(sch[c_gm], c_slice_axis)
                    else:
                        sch[bias_ub].compute_at(sch[c_gm], batch_in_out_axis)

            def _do_fusion_dx_drelu_compute_at():
                sch[drelu_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                if self.dx_para.get_para_map("out_of_order"):
                    # if cub_m%16!=0, when copy bitmask to ub, for every n0,
                    # the buffer should align to 32B*strideh*stridew
                    self._print_debug("bitmask_ub compute_at c_slice_axis")
                    if mask_ub_need_bind_buffer:
                        if TENSOR_MAP.get("dilate_ub") is None:
                            strideh, stridew = 1, 1
                        else:
                            strideh, stridew = DIM_MAP.get("dilate_dim")
                        align_buffer = reduce(lambda x, y: x * y, TILING["CUB_matrix"][1:4]) * strideh * stridew
                        self._print_debug("mask_ub_need_bind_buffer, align_buffer:", align_buffer)
                        sch[bitmask_ub].bind_buffer(bitmask_ub.op.axis[1], align_buffer, 0)
                    sch[bitmask_ub].compute_at(sch[c_gm], c_slice_axis)
                else:
                    sch[bitmask_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[fusion_dx_gm].compute_at(sch[c_gm], l0c_m_inner_outer)

            def _do_fusion_dx_add_drelu_compute_at():
                sch[add_res_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[add_input_ub].compute_at(sch[c_gm], add_input_at)
                if add_input_1_ub is not None:
                    sch[add_input_1_ub].compute_at(sch[c_gm], add_input_at)
                    sch[inter_add_compute_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[drelu_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[bitmask_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[fusion_dx_gm].compute_at(sch[c_gm], l0c_m_inner_outer)

            if fusion_type in [FUSION_DX_DRELU]:
                _do_fusion_dx_drelu_compute_at()
            elif fusion_type in [FUSION_DX_ADD_DRELU]:
                _do_fusion_dx_add_drelu_compute_at()
            elif fusion_type == FUSION_DX_ELEWISE:
                for input_tensor in TENSOR_MAP.get("input_tensor_list"):
                    sch[input_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
                for ub_tensor in TENSOR_MAP.get("ub_list"):
                    if "broadcast" in ub_tensor.op.tag:
                        sch[ub_tensor].compute_inline()
                    else:
                        sch[ub_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
            elif fusion_type in (
                FUSION_DX_DEQUANT,
                FUSION_DX_DEQUANT_QUANT,
                FUSION_DX_REQUANT
            ):
                _attach_ub_quant()
                for double_out_tensor_mem in DOUBLE_TENSOR_OUT:
                    sch[double_out_tensor_mem].compute_at(sch[c_gm], l0c_m_inner_outer)
            _attach_ub_bias()

            if dilate_ub is not None:
                filling_zero_ub = TENSOR_MAP.get("tensor_fillling_zero")
                sch[dilate_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[filling_zero_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                if var_map:
                    tensor_vn = TENSOR_MAP.get("tensor_vn")
                    sch[tensor_vn].compute_at(sch[c_gm], l0c_m_inner_outer)
            if not self.dx_para.get_para_map("no_need_use_ub_flag"):
                sch[c_ub].compute_at(sch[c_gm], l0c_m_inner_outer)

            if "data_transfer" in TENSOR_MAP:
                sch[c_ub].compute_inline()
                sch[TENSOR_MAP.get("data_transfer")].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, tbe_platform.CUBE_MKN.get("int8")["mac"][0]),
                    (1, tbe_platform.CUBE_MKN.get("int8")["mac"][0])
                )

        def _attach_al1_bl1():
            # attach tensor of al1 and bl1 to c_l0c
            def _al1_attach():
                al1_attach_at_cl0 = al1_parts[0] != 1 or (l0c_multi_group_flag and al1_parts[2] == 1)
                if al1_attach_at_cl0:
                    al1_compute_axis = al1_at_l0c_axis
                    if TILING["A_overhead_opt_flag"]:
                        sch[a_l1].allocate_at(sch[c_l0c], al1_compute_axis)
                        al1_compute_axis = al0_m_out
                    sch[a_l1].compute_at(sch[c_l0c], al1_compute_axis)
                    if self.dx_para.get_para_map("load3d_flag"):
                        sch[a_l0a_before].compute_at(sch[c_l0c], al1_compute_axis)
                elif TILING["AL1_shape"]:
                    al1_compute_scope = c_gm
                    al1_compute_axis = al1_at_c_axis
                    if TILING["A_overhead_opt_flag"]:
                        run_once_axis = [bl1_at_c_axis, noii_axis] if reorder_flag else []
                        sch[a_l1].allocate_at(sch[c_gm], al1_compute_axis, run_once_axes=run_once_axis)
                        al1_compute_axis = al0_m_out
                        al1_compute_scope = c_l0c
                    sch[a_l1].compute_at(sch[al1_compute_scope], al1_compute_axis)
                    if self.dx_para.get_para_map("load3d_flag"):
                        sch[a_l0a_before].compute_at(sch[al1_compute_scope], al1_compute_axis)
                else:
                    al1_compute_axis = batch_in_out_axis
                    al1_compute_scope = c_gm
                    if TILING["A_overhead_opt_flag"]:
                        sch[a_l1].allocate_at(sch[c_gm], al1_compute_axis,
                                            run_once_axes=[bl1_at_c_axis, noii_axis])
                        al1_compute_scope = c_l0c
                        al1_compute_axis = al0_m_out
                    sch[a_l1].compute_at(sch[al1_compute_scope], al1_compute_axis)
                    if self.dx_para.get_para_map("load3d_flag"):
                        sch[a_l0a_before].compute_at(sch[al1_compute_scope], al1_compute_axis)

            def _bl1_attach():
                bl1_attach_at_cl0 = bl1_parts[0] != 1 or (l0c_multi_group_flag and bl1_parts[2] == 1)
                if bl1_attach_at_cl0:
                    bl1_compute_axis = bl1_at_l0c_axis
                    if TILING["B_overhead_opt_flag"]:
                        sch[b_l1].allocate_at(sch[c_l0c], bl1_compute_axis)
                        bl1_compute_axis = bl0_n_outer
                    sch[b_l1].compute_at(sch[c_l0c], bl1_compute_axis)
                elif TILING["BL1_shape"]: # bl1_parts[0] == 1
                    bl1_compute_axis = bl1_at_c_axis
                    bl1_compute_scope = c_gm
                    if TILING["B_overhead_opt_flag"]:
                        run_once_axis = [al1_at_c_axis] if reorder_flag else [al1_at_c_axis, tile_axis]
                        sch[b_l1].allocate_at(sch[c_gm], bl1_compute_axis, run_once_axes=run_once_axis)
                        bl1_compute_axis = bl0_compute_axis
                        bl1_compute_scope = bl0_compute_scope
                    sch[b_l1].compute_at(sch[bl1_compute_scope], bl1_compute_axis)
                else:  # TILING["BL1_shape"]=[]
                    bl1_compute_axis = batch_in_out_axis
                    bl1_compute_scope = c_gm
                    if TILING["B_overhead_opt_flag"]:
                        sch[b_l1].allocate_at(sch[c_gm], bl1_compute_axis, run_once_axes=[al1_at_c_axis, tile_axis])
                        bl1_compute_axis = bl0_compute_axis
                        bl1_compute_scope = bl0_compute_scope
                    sch[b_l1].compute_at(sch[bl1_compute_scope], bl1_compute_axis)
            _al1_attach()
            _bl1_attach()

        def _do_double_buffer(fusion_type):
            # a_l1 b_l1
            if TILING.get("manual_pingpong_buffer")["AL1_pbuffer"] == 2 and (
                TILING["AL1_shape"] != []
            ):
                sch[a_l1].double_buffer()
            if TILING.get("manual_pingpong_buffer")["BL1_pbuffer"] == 2 and (
                TILING["BL1_shape"] != []
            ):
                sch[b_l1].double_buffer()

            # L0A L0B
            if TILING.get("manual_pingpong_buffer")["AL0_pbuffer"] == 2:
                sch[a_l0a].double_buffer()
            if TILING.get("manual_pingpong_buffer")["BL0_pbuffer"] == 2:
                sch[b_l0b].double_buffer()

            # c_l0c
            _double_buffer_l0c()

            # C_UB
            if not self.dx_para.get_para_map("no_need_use_ub_flag"):
                _double_buffer_cub(fusion_type)

        def _double_buffer_l0c():
            if TILING.get("manual_pingpong_buffer")["CL0_pbuffer"] == 2:
                sch[c_l0c].double_buffer()
                if bias_l0c is not None:
                    sch[bias_l0c].double_buffer()
                    sch[c_add_bias].double_buffer()
                    sch[bias_l0c].preload()
                    sch[bias_ub_brc].double_buffer()
                    sch[bias_ub_brc].preload()

        def _double_buffer_cub(fusion_type):
            if TILING.get("manual_pingpong_buffer")["CUB_pbuffer"] == 2:
                sch[c_ub].double_buffer()
                if bias_add_vector_ub is not None:
                    sch[bias_add_vector_ub].double_buffer()
                if dilate_ub is not None:
                    filling_zero_ub = TENSOR_MAP.get("tensor_fillling_zero")
                    sch[dilate_ub].double_buffer()
                    sch[filling_zero_ub].double_buffer()
                    if var_map:
                        tensor_vn = TENSOR_MAP.get("tensor_vn")
                        sch[tensor_vn].double_buffer()
                if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
                    sch[fusion_dx_gm].double_buffer()
                    sch[drelu_ub].double_buffer()
                    if fusion_type == FUSION_DX_ADD_DRELU:
                        sch[add_res_ub].double_buffer()
                        if dilate_ub is not None:
                            sch[add_input_ub].double_buffer()
                            sch[add_input_ub].preload()
                            if add_input_1_ub is not None:
                                sch[inter_add_compute_tensor].double_buffer()
                                sch[add_input_1_ub].double_buffer()
                                sch[add_input_1_ub].preload()
                        elif inter_add_compute_tensor is not None:
                            sch[inter_add_compute_tensor].double_buffer()
                elif fusion_type == FUSION_DX_ELEWISE:
                    for ub_tensor in TENSOR_MAP.get("ub_list"):
                        sch[ub_tensor].double_buffer()
                    for input_tensor in TENSOR_MAP.get("input_tensor_list"):
                        sch[input_tensor].double_buffer()
                        sch[input_tensor].preload()

        def _do_reused_by(fusion_type):
            if dilate_ub is not None:
                dx_output_ub = dilate_ub
            else:
                dx_output_ub = c_ub
            if fusion_type == FUSION_DX_ADD_DRELU:
                if dilate_ub is not None:
                    filling_zero_ub = TENSOR_MAP.get("tensor_fillling_zero")
                    sch[filling_zero_ub].reused_by(add_input_ub)
                    if add_input_1_ub is not None:
                        sch[dx_output_ub].reused_by(fusion_dx_gm, add_res_ub, inter_add_compute_tensor)
                    else:
                        sch[dx_output_ub].reused_by(fusion_dx_gm, add_res_ub)
                    if var_map:
                        sch[dx_output_ub].reused_by(drelu_ub)
                elif inter_add_compute_tensor is not None:
                    sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub, add_res_ub, inter_add_compute_tensor)
                else:
                    sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub, add_res_ub)
            elif fusion_type == FUSION_DX_DRELU:
                sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub)
            elif fusion_type == FUSION_DX_ELEWISE:
                iv_c_gm = cube_util.calc_info_of_iter_vars(sch[c_gm])
                len_axis = iv_c_gm[-2][1].extent
                if FUSION_TYPE_2_OPERAND_NUM.get(fusion_type) < 1:
                    for ub_tensor in TENSOR_MAP.get("ub_list"):
                        len_align = (
                            tvm.min(
                                len_axis, dx_output_ub.shape[2] - l0c_m_outer.var * len_axis
                            )
                            * ub_tensor.op.axis[3].dom.extent
                        )

                        sch[ub_tensor].bind_buffer(ub_tensor.op.axis[1], len_align, 0)
                        sch[dx_output_ub].reused_by(ub_tensor)

        def _fusion_intrin_mapping(fusion_type):
            def _add_res_ub_insn():
                if dilate_ub is None or add_input_1_ub is not None:
                    sch[add_res_ub].emit_insn(add_res_ub.op.axis[0], "vector_add")
                else:
                    sch[add_res_ub].emit_insn(add_res_ub.op.axis[0], "phony_insn")

            def _quant_vector_insn():
                for tensor_name in TENSOR_MAP:
                    if tensor_name == "reform_op":
                        reform_ub = TENSOR_MAP.get(tensor_name)
                        ndim = len(sch[reform_ub].op.axis)
                        coo, _ = sch[reform_ub].split(
                            sch[reform_ub].op.axis[ndim - 1],
                            tbe_platform.CUBE_MKN.get("float16")["mac"][1]
                        )
                        axis_list = sch[reform_ub].op.axis[0:ndim - 1]
                        sch[reform_ub].reorder(coo, *axis_list)
                        sch[reform_ub].emit_insn(sch[reform_ub].op.axis[2], "vector_auto")
                    elif tensor_name == "elewise_tensor":
                        for elewise_tensor in TENSOR_MAP.get(tensor_name):
                            sch[elewise_tensor].emit_insn(
                                elewise_tensor.op.axis[0], "vector_auto"
                            )

            def _quant_copy_insn():
                for tensor_name in TENSOR_MAP:
                    if tensor_name == "deq":
                        sch[TENSOR_MAP.get(tensor_name)].emit_insn(
                            TENSOR_MAP.get(tensor_name).op.axis[0], "dma_copy"
                        )
                    elif tensor_name == "input_tensor":
                        for input_tensor in TENSOR_MAP.get(tensor_name):
                            sch[input_tensor].emit_insn(
                                input_tensor.op.axis[0], "dma_copy"
                            )
                    elif tensor_name == "data_transfer":
                        c_ub_reform = TENSOR_MAP.get(tensor_name)
                        reform_outer, reform_inner = sch[c_ub_reform].split(
                            c_ub_reform.op.axis[3], nparts=2
                        )
                        sch[c_ub_reform].reorder(
                            reform_outer,
                            c_ub_reform.op.axis[0],
                            c_ub_reform.op.axis[1],
                            c_ub_reform.op.axis[2],
                            reform_inner
                        )
                        sch[c_ub_reform].emit_insn(c_ub_reform.op.axis[2], "dma_copy")

            def _quant_ub_insn():
                _quant_vector_insn()
                _quant_copy_insn()
                for tensor_name in TENSOR_MAP:
                    if tensor_name == "c_ub" and fusion_type != FUSION_DX_REQUANT:
                        axis_index = 2 if quan_para.get("deq_vector") else 0
                        if cube_util.is_v200_version_new():
                            sch[TENSOR_MAP.get(tensor_name)].emit_insn(TENSOR_MAP.get(tensor_name).op.axis[axis_index],
                                                                "dma_copy")
                        else:
                            emit = "vector" if axis_index == 2 else "scalar"
                            sch[TENSOR_MAP.get(tensor_name)].pragma(TENSOR_MAP.get(tensor_name).op.axis[axis_index],
                                                                "deq_scale", emit)
                    elif tensor_name == "input_ub" and quan_para.get("q_padding"):
                        sch[TENSOR_MAP.get(tensor_name)].emit_insn(TENSOR_MAP.get(tensor_name).op.axis[0],
                                                                   "dma_padding")
                    elif tensor_name == "cast_i8_ub":
                        cast_i8_ub = TENSOR_MAP.get(tensor_name)
                        round_mode = quan_para.get("q_round").lower()
                        if round_mode != 'round':
                            error_manager_cube.raise_err_message_cube(
                                f'Round mode should be Round only, {round_mode} is not supported')
                        round_mode_emit_insn = "vector_conv"
                        sch[cast_i8_ub].emit_insn(cast_i8_ub.op.axis[0], round_mode_emit_insn)

            if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
                sch[bitmask_ub].emit_insn(bitmask_ub.op.axis[0], "dma_copy")
                sch[drelu_ub].emit_insn(drelu_ub.op.axis[0], "vector_selects_bool")
                sch[fusion_dx_gm].emit_insn(fusion_dx_gm.op.axis[0], "phony_insn")
                if fusion_type == FUSION_DX_ADD_DRELU:
                    sch[add_input_ub].emit_insn(add_input_ub.op.axis[0], "dma_copy")
                    if add_input_1_ub is not None:
                        sch[add_input_1_ub].emit_insn(add_input_1_ub.op.axis[0], "dma_copy")
                        if dilate_ub is not None:
                            sch[inter_add_compute_tensor].emit_insn(inter_add_compute_tensor.op.axis[0], "phony_insn")
                        else:
                            sch[inter_add_compute_tensor].emit_insn(inter_add_compute_tensor.op.axis[0], "vector_add")
                    _add_res_ub_insn()
            elif fusion_type == FUSION_DX_ELEWISE:
                for input_tensor in TENSOR_MAP.get("input_tensor_list"):
                    sch[input_tensor].emit_insn(input_tensor.op.axis[0], "dma_copy")
                for ub_tensor in TENSOR_MAP.get("ub_list"):
                    sch[ub_tensor].emit_insn(ub_tensor.op.axis[0], "vector_auto")
            elif fusion_type in (
                FUSION_DX_DEQUANT,
                FUSION_DX_DEQUANT_QUANT,
                FUSION_DX_REQUANT
            ):
                _quant_ub_insn()

        def _intrin_mapping(fusion_type):
            def _l1fusion_intrin():
                if TILING["AL1_shape"] is not None:
                    if self.dx_para.get_para_map("input_memory_type")[0] == 1:
                        sch[a_l1].emit_insn(a_l1.op.axis[0], "phony_insn")
                    else:
                        if self.dx_para.get_para_map("FM_NHWC_TRANS_5HD"):
                            sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy", {"layout_transform": "nd2nz"})
                        else:
                            sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy")
                        if self.dx_para.get_para_map("l1_fusion_type") != -1:
                            sch[a_l1].pragma(a_l1.op.axis[0], "jump_data", 1)
                if self.dx_para.get_para_map("load3d_flag"):
                    setfmatrix_dict = {
                        "conv_kernel_h": 1,
                        "conv_kernel_w": 1,
                        "conv_padding_top": 0,
                        "conv_padding_bottom": 0,
                        "conv_padding_left": 0,
                        "conv_padding_right": 0,
                        "conv_stride_h": 1,
                        "conv_stride_w": 1,
                        "conv_fm_c": a_l1.shape[1] * a_l1.shape[4],
                        "conv_fm_h": a_l1.shape[2],
                        "conv_fm_w": a_l1.shape[3]
                    }
                    sch[a_l0a_before].emit_insn(
                        a_l0a_before.op.axis[0], "set_fmatrix", setfmatrix_dict
                    )
                    sch[a_l0a].emit_insn(a_l0a.op.axis[1], "im2col")
                else:
                    sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy")

            _l1fusion_intrin()
            if self.dx_para.get_para_map("WEIGHT_NHWC_TRANS_FZ"):
                sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy", {"layout_transform": "nd2nz"})
            else:
                sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy")
            if b_l0b.dtype == "float32":
                sch[b_l0b].split(b_l0b.op.axis[-2], factor=8)
                sch[b_l0b].emit_insn(b_l0b.op.axis[-4], "dma_copy", {'img2col': 1})
            else:
                sch[b_l0b].emit_insn(b_l0b.op.axis[0], "dma_copy")

            if fusion_type not in (
                FUSION_DX_DEQUANT,
                FUSION_DX_DEQUANT_QUANT,
                FUSION_DX_REQUANT
            ):
                if c_ub is not None:
                    if tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2ub"):
                        if c_ub.dtype == "float32" and a_l1.dtype == "float32":
                            _, split_axis = sch[c_ub].split(c_ub.op.axis[-3], factor=2)
                            sch[c_ub].emit_insn(split_axis, "fixpipe_op")
                        else:
                            sch[c_ub].emit_insn(c_ub.op.axis[0], "fixpipe_op")
                    else:
                        sch[c_ub].emit_insn(c_ub.op.axis[0], "dma_copy")

            def _emit_single_out_insn():
                emit_str = get_fixpipe_emit_str()
                if self.dx_para.get_para_map("5HD_TRANS_NHWC"):
                    sch[c_gm].emit_insn(l0c_m_inner_inner, emit_str, {"layout_transform": "nz2nd"})
                elif c_gm.dtype == "float32" and a_l1.dtype == "float32" and c_ub is None:
                    _, split_axis = sch[c_gm].split(l0c_n_inner_inner, factor=2)
                    sch[c_gm].emit_insn(split_axis, emit_str, {"layout_transform": "channel_split"})
                else:
                    sch[c_gm].emit_insn(l0c_n_inner_inner, emit_str)

            if DOUBLE_TENSOR_OUT:
                sch[DOUBLE_TENSOR_OUT[0]].emit_insn(DOUBLE_TENSOR_OUT[0].op.axis[0], "dma_copy")
                sch[DOUBLE_TENSOR_OUT[1]].emit_insn(DOUBLE_TENSOR_OUT[1].op.axis[0], "dma_copy")
                sch[c_gm].emit_insn(l0c_n_inner_inner, "phony_insn")
            else:
                _emit_single_out_insn()

            if dilate_ub is not None:
                filling_zero_ub = TENSOR_MAP.get("tensor_fillling_zero")
                if var_map:
                    tensor_vn = TENSOR_MAP.get("tensor_vn")
                    sch[dilate_ub].reused_by(tensor_vn)
                    sch[tensor_vn].emit_insn(sch[tensor_vn].op.axis[0], "phony_insn")
                if bias_add_vector_ub is not None:
                    if var_map:
                        sch[filling_zero_ub].reused_by(dilate_ub, bias_add_vector_ub)
                    else:
                        sch[dilate_ub].reused_by(filling_zero_ub, bias_add_vector_ub)
                else:
                    if var_map:
                        sch[filling_zero_ub].reused_by(dilate_ub)
                    else:
                        sch[dilate_ub].reused_by(filling_zero_ub)
                if fusion_type in (FUSION_DX_ADD_DRELU,):
                    sch[filling_zero_ub].emit_insn(
                        sch[filling_zero_ub].op.axis[0], "phony_insn"
                    )
                else:
                    sch[filling_zero_ub].emit_insn(
                        sch[filling_zero_ub].op.axis[0], "vector_dup"
                    )
                vadd_at_axis = self._dilate_schedule(
                    sch,
                    dilate_ub,
                    DIM_MAP.get("out_hwdim")[1],
                    DIM_MAP.get("dilate_dim")[1],
                    DIM_MAP.get("dilate_dim")[0],
                    var_map
                )
                sch[dilate_ub].emit_insn(vadd_at_axis, "vector_add")
            elif bias_add_vector_ub is not None:
                sch[c_ub].reused_by(bias_add_vector_ub)

            if bias_add_vector_ub is not None:
                sch[bias_ub].emit_insn(sch[bias_ub].op.axis[0], "dma_copy")
                sch[bias_add_vector_ub].emit_insn(
                    sch[bias_add_vector_ub].op.axis[0],
                    "vector_auto"
                )

            if fusion_type != FUSION_NONE:
                _fusion_intrin_mapping(fusion_type)
            mad_dict = {
                "mad_pattern": 2,
                "k_outer": [
                    reduce_axis_serial[0],
                    reduce_axis_serial[1],
                    reduce_axis_serial[2]
                ]
            }

            if bias_ub_brc is not None:
                sch[bias_l0c].reused_by(c_add_bias, c_l0c)
                sch[c_add_bias].emit_insn(c_add_bias.op.axis[0], 'phony_insn')
                sch[bias_l0c].split(bias_l0c.op.axis[3], BRC_STANDARD_BLOCK_SIZE)
                sch[bias_l0c].emit_insn(bias_l0c.op.axis[2], 'dma_copy')
                sch[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
                sch[bias_ub_brc].emit_insn(bias_ub_brc.op.axis[0], 'vector_auto')
                mad_dict["init_bias"] = 1

            if bias_bt is not None:
                sch[bias_bt].emit_insn(bias_bt.op.axis[0], 'dma_copy', {'mem_align': 1})
                sch[bias_l1].emit_insn(bias_l1.op.axis[0], 'dma_copy', {'mem_align': 1})

            if (self.dx_para.get_para_map("no_need_use_ub_flag") and
                "impl_mode" in c_l0c.op.attrs and
                c_l0c.op.attrs["impl_mode"] == "high_performance"):
                mad_dict["hf32"] = 1
            sch[c_l0c].emit_insn(batch_l0c_inner, "mad", mad_dict)
            self._print_ir_conv("intrin mapping", sch)

        def _get_al1_bound():
            dedy_shape_nc1hwc0 = DIM_MAP.get("img_shape")
            tiling_m0 = TILING["CL0_matrix"][2]
            if len(TILING["AL1_shape"]) != 0:
                k_al1, multi_m_al1 = TILING["AL1_shape"][:2]
                al1_m = multi_m_al1 * TILING["CL0_matrix"][1] * tiling_m0
                al1_bound = al1_m * k_al1
            else:
                al1_bound = (
                    DIM_MAP.get(cube_util.GroupDictKeys.dy_c1_extend)
                    * align(dedy_shape_nc1hwc0[2] * dedy_shape_nc1hwc0[3], tiling_m0)
                    * dedy_shape_nc1hwc0[4]
                )
            return al1_bound

        def _get_dilate_ub_bound():
            nc_factor, mc_factor, tiling_m0, tiling_n0 = TILING["CUB_matrix"][:4]
            cub_bound = nc_factor * mc_factor * tiling_m0 * tiling_n0
            sch[c_ub].set_buffer_size(cub_bound)
            l0c_bound = (
                TILING["CL0_matrix"][0] * TILING["CL0_matrix"][1] * tiling_m0 * tiling_n0
            )
            sch[c_l0c].set_buffer_size(l0c_bound)
            dilate_bound = l0c_factor[1] * nc_factor * tiling_n0
            sch[dilate_ub].set_buffer_size(dilate_bound)
            filling_zero_ub = TENSOR_MAP.get("tensor_fillling_zero")
            sch[filling_zero_ub].set_buffer_size(dilate_bound)
            if fusion_type == FUSION_DX_ADD_DRELU:
                sch[add_input_ub].set_buffer_size(dilate_bound)
                sch[add_res_ub].set_buffer_size(dilate_bound)
                sch[fusion_dx_gm].set_buffer_size(dilate_bound)

        def _is_conv1d():
            return (
                DIM_MAP.get("dx_5D_shape")[2] == 1
                and DIM_MAP.get("img_shape")[2] == 1
                and (TENSOR_MAP.get("dilate_ub") is None or DIM_MAP.get("dilate_dim")[0] == 1)
            )

        def _res_select_write(res):
            # selet write
            if res.op.tag == "conv_virtual_res":
                DOUBLE_TENSOR_OUT.append(res.op.input_tensors[0])
                DOUBLE_TENSOR_OUT.append(res.op.input_tensors[1])
                res_before_write_select = res.op.input_tensors[0]
            else:
                res_before_write_select = res

            return res_before_write_select

        def _get_fusion_type(fusion_type):
            # get the fusion num
            fusion_type_num = FUSION_TYPE_2_NUM.get(fusion_type)
            if isinstance(fusion_type_num, tuple):
                if dx_res.dtype == "float16":
                    fusion_type_num = 1
                else:
                    fusion_type_num = 2
            self.dx_para.update_para_map("fusion_type_num", fusion_type_num)

        def _handle_dynamic_shape():
            # set storage bound
            if var_map:
                al1_bound = _get_al1_bound()
                sch[a_l1].set_buffer_size(al1_bound)
            if ("dedy_h" in var_map or "dedy_w" in var_map) and dilate_ub is not None:
                _get_dilate_ub_bound()
            # sequential_malloc
            sch.sequential_malloc(tbe_platform_info.scope_cbuf)
            sch.sequential_malloc(tbe_platform_info.scope_ca)
            sch.sequential_malloc(tbe_platform_info.scope_cb)
            sch.sequential_malloc(tbe_platform_info.scope_cc)
            sch.sequential_malloc(tbe_platform_info.scope_ubuf)
            # mem_unique
            sch[a_l1].mem_unique()
            sch[a_l0a].mem_unique()
            sch[b_l1].mem_unique()
            sch[b_l0b].mem_unique()
            sch[c_l0c].mem_unique()
            # in some specific fusion mode cub_tensor need to be reused_by
            if (fusion_type not in (FUSION_DX_DRELU, FUSION_DX_ADD_DRELU) and
                not (bias_add_vector_ub is not None and dilate_ub is None)) and \
                not self.dx_para.get_para_map("no_need_use_ub_flag"):
                sch[c_ub].mem_unique()

        def _check_overload_dy(overload_flag_gm, overload_flag_l0c):
            """
            check whether dy is overload
            Use the following conditions to judge:
            1. if multi core in n axis, dy will overload
            2. overload_flag_gm or overload_flag_l0c is True, al1 and
                bl1 not full load, will overload
            Returns
            -------
            true for overload, false for not overload
            """
            overload_flag = False
            if TILING.get("block_dim")[1] > 1:
                overload_flag = True
            elif (
                TILING["AL1_shape"]
                and TILING["BL1_shape"]
                and (overload_flag_gm or overload_flag_l0c)
            ):
                overload_flag = True
            return overload_flag

        def _set_overload_flag(param, overload_flag, overload_axis):
            """
            set flag on the first axis
            """
            cache_read_mode = 0 if overload_flag else 1
            param.pragma(overload_axis, "json_info_cache_read_mode", cache_read_mode)

        TILING.clear()
        dx_res = tensor
        sch = sch_list[0]
        self._print_ir_conv("schedule", sch)

        dx_res_write = _res_select_write(dx_res)
        quan_para = self._check_quant_fusion(dx_res_write)

        # set scope for all tensor
        tensor_dx_gm, var_map = self._set_data_layout(dx_res_write, dx_res, sch, var_range)
        kernel_name = tensor_dx_gm.op.attrs["kernel_name"]
        fusion_type = self.dx_para.get_para_map("FUSION_TYPE")
        _get_fusion_type(fusion_type)
        is_conv1d_bool = _is_conv1d()
        self._print_debug("IS_CONV1D:", is_conv1d_bool)

        self._print_ir_conv("set scope", sch)

        # get tensor
        a_l1, b_l1, a_l0a, b_l0b, c_ub, dilate_ub, c_l0c, c_gm = (
            TENSOR_MAP.get("a_l1"),
            TENSOR_MAP.get("b_l1"),
            TENSOR_MAP.get("a_l0a"),
            TENSOR_MAP.get("b_l0b"),
            TENSOR_MAP.get("c_ub"),
            TENSOR_MAP.get("dilate_ub"),
            TENSOR_MAP.get("c_l0c"),
            TENSOR_MAP.get("c_gm")
        )
        l0c_multi_group_flag = False
        if dx_res_write.dtype == "int8":
            # In quant or requant scenes, co of ddr is 32, c1_ddr is c1_loc//2
            DIM_MAP.get("dx_6GD_shape")[2] = (DIM_MAP.get("dx_6GD_shape")[2] + 1) // 2
            DIM_MAP.get("dx_6GD_shape")[5] = DIM_MAP.get("dx_6GD_shape")[5] * 2
            # In quant scenes, if C1 % 2 == 1 and G>1, the min group_l0c is 2
            if DIM_MAP.get("dx_c1_extend") % 2 == 1 and DIM_MAP.get("g_extend") > 1:
                l0c_multi_group_flag = True
                self._check_quant_fusion_legal(fusion_type)

        if dx_res_write.dtype == "float32" and a_l1.dtype == "float32":
            # cin1_g was calculated with c0=8
            DIM_MAP.get("dx_6GD_shape")[5] = DIM_MAP.get("dx_6GD_shape")[5] // 2
        if self.dx_para.get_para_map("load3d_flag"):
            a_l0a_before = TENSOR_MAP.get("a_l0a_before")
        drelu_ub, bitmask_ub, add_res_ub, add_input_ub, add_input_1_ub, inter_add_compute_tensor, fusion_dx_gm = (
            TENSOR_MAP.get("drelu_ub"),
            TENSOR_MAP.get("bitmask_ub"),
            TENSOR_MAP.get("add_res_ub"),
            TENSOR_MAP.get("add_input_ub"),
            TENSOR_MAP.get("add_input_1_ub"),
            TENSOR_MAP.get("inter_add_compute_tensor"),
            TENSOR_MAP.get("fusion_dx_gm")
        )
        bias_add_vector_ub, bias_ub = TENSOR_MAP.get("bias_add_vector"), TENSOR_MAP.get(
            "bias_ub"
        )
        bias_ub_brc, bias_l0c, c_add_bias = (
            TENSOR_MAP.get("bias_ub_brc"), TENSOR_MAP.get("bias_l0c"), TENSOR_MAP.get("c_add_bias")
        )

        bias_l1, bias_bt = (
            TENSOR_MAP.get("bias_l1"), TENSOR_MAP.get("bias_bt")
        )
        self._get_tiling(
            dx_res_write, fusion_type, kernel_name, is_conv1d_bool, tiling_case, var_map, l0c_multi_group_flag
        )

        # get factor and parts from tiling
        (
            l0c_factor,
            l0c_ub_parts,
            al1_parts,
            bl1_parts,
            undilate_l0c_m,
            need_buffer_tile,
            mask_ub_need_bind_buffer
        ) = self._get_aicore_tiling_factor(is_conv1d_bool, sch, var_map, var_range, l0c_multi_group_flag)
        al0_axis_factor, bl0_axis_factor, reduce_axis_factor = self._get_mmad_factor()
        num_batch = DIM_MAP.get("img_shape")[0]
        self._print_ir_conv("before split", sch)
        dx_c1_extend = DIM_MAP.get(cube_util.GroupDictKeys.dx_c1_extend)
        # split and get axis of l0c, al1, bl1
        (
            batch_in_out_axis,
            l1_n_outer_outer,
            batch_in_inner_axis,
            l1_n_out_inner_out,
            l1_m_outer_inner_in,
            l0c_n_inner,
            l0c_m_inner,
            tile_axis,
            noii_axis,
            blockidx_list,
            g_inner,
            overload_axis,
            overload_flag_gm,
            l0c_m_outer,
            reorder_flag
        ) = self._get_l0c_and_l1_axis(
            sch, c_gm, l0c_factor, al1_parts, bl1_parts, num_batch, dx_c1_extend, var_map, l0c_multi_group_flag
        )
        al1_at_c_axis = batch_in_inner_axis
        bl1_at_c_axis = l1_n_outer_outer
        c_slice_axis = l1_m_outer_inner_in
        self._print_ir_conv("split with al1 and bl1 factor", sch)

        # attach tensor of CUB
        support_fixpipe = (tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out") or
                            tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2ub"))
        if self.dx_para.get_para_map("input_fp32_flag") and support_fixpipe:
            l0c_n_inner_outer, l0c_n_inner_inner = sch[c_gm].split(
                l0c_n_inner, nparts=max(l0c_ub_parts[0] // 2, 1)
            )
        elif self.dx_para.get_para_map("5HD_TRANS_NHWC"):
            l0c_n_inner_outer, l0c_n_inner_inner = sch[c_gm].split(
                l0c_n_inner, factor=16
            )
        else:
            l0c_n_inner_outer, l0c_n_inner_inner = sch[c_gm].split(
                l0c_n_inner, nparts=l0c_ub_parts[0]
            )
        l0c_m_inner_outer, l0c_m_inner_inner = sch[c_gm].split(l0c_m_inner, nparts=1)
        add_input_at, l0c_m_inner_outer = sch[c_gm].split(l0c_m_inner_outer, nparts=1)

        if self.dx_para.get_para_map("5HD_TRANS_NHWC"):
            sch[c_gm].reorder(
                l0c_n_inner_outer,
                add_input_at,
                l0c_m_inner_outer,
                l0c_m_inner_inner,
                l0c_n_inner_inner
            )
        else:
            sch[c_gm].reorder(
                l0c_n_inner_outer,
                add_input_at,
                l0c_m_inner_outer,
                l0c_n_inner_inner,
                l0c_m_inner_inner
            )
        self._print_ir_conv("reorder loc", sch)

        _attach_ub(fusion_type)
        self._print_ir_conv("attach CUB", sch)

        # attach tensor of l0c
        new_c_col_axis = [
            sch[c_l0c].op.axis[1],
            sch[c_l0c].op.axis[2],
            sch[c_l0c].op.axis[3],
            sch[c_l0c].op.axis[4],
            sch[c_l0c].op.axis[0]
        ]
        sch[c_l0c].compute_at(sch[c_gm], c_slice_axis)

        def _do_buffer_align():
            sch[c_l0c].buffer_align(
                (1, 1),
                (1, 1),
                (1, 1),
                (1, tbe_platform.CUBE_MKN.get(c_l0c.dtype).get("mac")[0]),
                (1, tbe_platform.CUBE_MKN.get(c_l0c.dtype).get("mac")[2]),
                (1, 1),
                (1, tbe_platform.CUBE_MKN.get(a_l0a.dtype).get("mac")[1])
            )
        _do_buffer_align()
        if self.dx_para.get_para_map("no_need_use_ub_flag") and not var_map:
            sch[c_l0c].storage_align(sch[c_l0c].op.axis[2], CUBE_MUL_SHAPE, 0)
        if bias_l0c is not None:
            sch[bias_l0c].compute_at(sch[c_gm], c_slice_axis)
            sch[c_add_bias].compute_at(sch[c_gm], c_slice_axis)
            sch[bias_ub_brc].compute_at(sch[c_gm], c_slice_axis)
        if bias_bt is not None:
            sch[bias_l1].compute_at(sch[c_gm], c_slice_axis)
            sch[bias_bt].compute_at(sch[c_gm], c_slice_axis)

        self._print_ir_conv("attach l0c", sch)

        # split and get axis of reduce, al0_at_axis, bl0_at_axis
        al0_m_out, bl0_n_outer, k_outer_outer, batch_l0c_inner = self._get_l0a_and_l0b_axis(
            sch, c_l0c, new_c_col_axis, al0_axis_factor, bl0_axis_factor, reduce_axis_factor
        )
        self._print_ir_conv("split with al0/bl0/reduce factor", sch)

        # attach tensor of a_l0a
        sch[a_l0a].compute_at(sch[c_l0c], al0_m_out)

        def _bl0_attach():
            """
            if l0c_multi_group_flag and bl0_factor in group axis is 2,
            comput axis equal with l0c
            """
            if l0c_multi_group_flag and bl0_axis_factor[2] > 1:
                bl0_compute_axis = c_slice_axis
                bl0_compute_scope = c_gm
                sch[b_l0b].compute_at(sch[c_gm], c_slice_axis)
            else:
                bl0_compute_axis = bl0_n_outer
                bl0_compute_scope = c_l0c
                sch[b_l0b].compute_at(sch[c_l0c], bl0_n_outer)
            return bl0_compute_axis, bl0_compute_scope
        bl0_compute_axis, bl0_compute_scope = _bl0_attach()
        self._print_ir_conv("attach l0a/l0b", sch)

        # split and get axis of al1_at_l0c_axis, bl1_at_l0c_axis
        (
            al1_at_l0c_axis,
            bl1_at_l0c_axis,
            reduce_axis_serial,
            overload_flag_l0c
        ) = self._get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer)

        _attach_al1_bl1()
        self._print_ir_conv("attach al1/bl1", sch)

        def _buffer_tile_l0c_c1():
            no_coefficient = (l0c_factor[0] * 2 if dx_res_write.dtype == "int8" else l0c_factor[0])
            noo_coefficient_unzero = int_ceil_div(
                    int_ceil_div(DIM_MAP.get("dx_6GD_shape")[2], l0c_factor[0]), bl1_parts[1]
                )
            noo_coefficient = 0 if bl1_parts[1] == 1 else noo_coefficient_unzero
            noio_coefficient = (
                0
                if TILING["block_dim"][1] == 1
                else int_ceil_div(noo_coefficient_unzero, TILING["block_dim"][1])
            )
            noii_coefficient = (
                0 if int_ceil_div(noo_coefficient_unzero, TILING["block_dim"][1]) == 1
                else 1
            )
            cub_buffertile_n_min = (
                bl1_at_c_axis.var * noo_coefficient
                + noio_coefficient * l1_n_out_inner_out
                + noii_coefficient * noii_axis.var
            ) * no_coefficient
            sch[c_l0c].buffer_tile(
                (None, 1),
                (None, None),
                (cub_buffertile_n_min, no_coefficient),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
            )
            if c_add_bias is not None:
                sch[c_add_bias].buffer_tile(
                (None, 1),
                (None, None),
                (cub_buffertile_n_min, no_coefficient),
                (None, None),
                (None, None),
                )
        if not l0c_multi_group_flag and not self.dx_para.get_para_map("no_need_use_ub_flag"):
            _buffer_tile_l0c_c1()
        # do buffer_tile or buffer_align for cub
        if need_buffer_tile and not self.dx_para.get_para_map("input_fp32_flag"):
            _do_buffer_tile()
            self._print_ir_conv("after_tile", sch)
        else:
            if c_ub is not None:
                if self.dx_para.get_para_map("input_fp32_flag"):
                    sch[c_ub].buffer_align(
                        (1, 1),
                        (1, 1),
                        (1, tbe_platform.CUBE_MKN.get("float32")["mac"][0]),
                        (1, tbe_platform.CUBE_MKN.get("float32")["mac"][1])
                    )
                else:
                    sch[c_ub].buffer_align(
                        (1, 1),
                        (1, 1),
                        (1, tbe_platform.CUBE_MKN.get("float16")["mac"][0]),
                        (1, tbe_platform.CUBE_MKN.get("float16")["mac"][0])
                    )
            if bias_add_vector_ub is not None and dilate_ub is None:
                sch[bias_add_vector_ub].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, tbe_platform.CUBE_MKN.get("float16")["mac"][0]),
                    (1, tbe_platform.CUBE_MKN.get("float16")["mac"][0])
                )

        # double buffer
        _do_double_buffer(fusion_type)
        self._print_ir_conv("enable double buffer", sch)

        _do_reused_by(fusion_type)
        self._print_ir_conv("reused_by", sch)

        # preload
        if not var_map and self.dx_para.get_para_map("DATA_AMOUNT_CUB") * (
            1 + 2 * FUSION_TYPE_2_OPERAND_NUM.get(fusion_type)
        ) <= tbe_platform_info.get_soc_spec("UB_SIZE"):
            self._print_debug("dx opti ub preload enable.")
            if fusion_type == FUSION_DX_DRELU:
                sch[bitmask_ub].double_buffer()
            elif fusion_type == FUSION_DX_ADD_DRELU:
                sch[bitmask_ub].double_buffer()
                sch[bitmask_ub].preload()
                if dilate_ub is None:
                    sch[add_input_ub].double_buffer()
                    sch[add_input_ub].preload()
                    if add_input_1_ub is not None:
                        sch[add_input_1_ub].double_buffer()
                        sch[add_input_1_ub].preload()

            self._print_ir_conv("preload", sch)
        # intrin mapping
        _intrin_mapping(fusion_type)

        overload_flag = _check_overload_dy(overload_flag_gm, overload_flag_l0c)
        _set_overload_flag(sch[c_gm], overload_flag, overload_axis)

        def _handle_workspace():
            l1_tensor_map = {}
            if not self.dx_para.get_para_map("fmap_l1_addr_flag"):
                l1_tensor_map = None
            else:
                fmap = DeConvKernelSize1Pattern.dedy
                if (
                    self.dx_para.get_para_map("l1_fusion_type") != -1
                    and self.dx_para.get_para_map("input_memory_type")[0] == 0
                ):
                    sch[a_l1].set_buffer_size(
                        self.dx_para.get_para_map("fmap_l1_valid_size")
                    )
                    l1_tensor_map[fmap] = a_l1
                else:
                    l1_tensor_map = None
            L1CommonParam.l1_fusion_tensors_map = l1_tensor_map

        _handle_workspace()

        # clear global cache
        if var_map:
            _handle_dynamic_shape()

        sch.tbe_compile_para = self.dx_para.get_para_map("tbe_compile_para")
        if self.dx_para.get_para_map("preload_c_l0c") and TILING.get("manual_pingpong_buffer")["CL0_pbuffer"] == 2:
            sch[c_l0c].preload()
        support_preload_a_l1 = (self.dx_para.get_para_map("preload_a_l1")
                                and TILING.get("manual_pingpong_buffer")["AL1_pbuffer"] == 2)
        # allocate_at conflicts preload
        if TILING["A_overhead_opt_flag"] == 0 and support_preload_a_l1:
            sch[a_l1].preload()

        TILING.clear()
        DIM_MAP.clear()
        TENSOR_MAP.clear()
        DOUBLE_TENSOR_OUT.clear()
        return sch


def opti_schedule(tensor, sch_list, tiling_case=None, var_range=None):
    """
    the schedule of conv2d_backprop_opti_input
    Parameter:
    -------------------------------------------------------------------
    :param tensor: a tensor
    :param sch_list: a schedule list
    :param tiling_case: fix tiling for dynamic shape
    :param var_range: var_range for dynamic shape
    :return: schedule
    -------------------------------------------------------------------
    """
    dx_sch = Conv2dDxOptiSchedule()
    sch = dx_sch.opti_schedule(tensor, sch_list, tiling_case, var_range)
    return []
