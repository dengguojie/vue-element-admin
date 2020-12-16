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
from functools import reduce  # pylint: disable=C0302

from te import tvm
from te.domain.tiling.get_tiling import get_tiling
from te.lang.cce.te_compute.conv2d_backprop_input_opti_compute import (
    DeConvKernelSize1Pattern
)
from te.lang.cce.te_compute.cube_util import calc_info_of_iter_vars
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.lang.cce.te_compute.cube_util import GroupDictKeys
from te.lang.cce.te_compute.util import int_ceil_div
from te.lang.cce.te_schedule.util import L1CommonParam
from te.platform import cce_conf
from te.platform import cce_params
from te.utils.error_manager import error_manager_util

# Don't modify,used in log_util
DX_SUPPORT_TAG_LOG_PREFIX = "#Conv2DBackpropInput only support#"
# default False
DEBUG_MODE = 0
CONST_L1_SHAPE_DIM = 4
DTYPE_BYTE_MAP = {"float16": 2, "float32": 4, "int8": 1, "int32": 4}
CUB_BUFFER_LIMIT = 4096
TENSOR_MAP = {}
TILING = {}
DIM_MAP = {}

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
        pass

    para_map = {"DATA_AMOUNT_CUB": 0, "FUSION_TYPE": FUSION_NONE}

    @staticmethod
    def update_para_map(key, value):
        """
        updata para map with key and value
        """
        DeconvParam.para_map[key] = value

    @staticmethod
    def get_para_map(key):
        """
        get value by key
        """
        return DeconvParam.para_map[key]


def _raise_dx_opti_err(msg):
    """
    In op Conv2DBackpropInput_opti, [%s] % (msg)
    msg for discribe the error info
    the error info only for Conv2DBackpropInput_opti's developers
    """
    args_dict = {"errCode": "E60108", "reason": msg}
    msg = error_manager_util.get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def _print_debug(*params):
    """
    print log if debug
    :param params: infos
    :return: None
    """
    if DEBUG_MODE:
        print(params)


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


def _calc_double_op_num(fusion_type):
    if fusion_type in (FUSION_DX_DEQUANT_QUANT, FUSION_DX_DEQUANT, FUSION_DX_REQUANT):
        double_op_num = 0.125
        if fusion_type == FUSION_DX_DEQUANT_QUANT:
            double_op_num += 4
        elif fusion_type == FUSION_DX_DEQUANT:
            if "dequant_relu" in TENSOR_MAP or "dequant_sqrt" in TENSOR_MAP:
                double_op_num += 1
            elif TENSOR_MAP.get("input_tensor"):
                double_op_num += 1
        else:
            pass
        FUSION_TYPE_2_OPERAND_NUM[fusion_type] = double_op_num
    if fusion_type == FUSION_DX_ELEWISE:
        double_op_num = 0
        if "bias_add_vector" in TENSOR_MAP:
            double_op_num += 0.125
        if TENSOR_MAP.get("input_tensor_list"):
            double_op_num += 1
        FUSION_TYPE_2_OPERAND_NUM[fusion_type] = double_op_num


def _get_data_amount_l1(l1_shape, isdouble):
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
        _raise_dx_opti_err("{} can not be None".format(l1_shape))
    if TILING.get(l1_shape) != [] and len(TILING.get(l1_shape)) != CONST_L1_SHAPE_DIM:
        _raise_dx_opti_err("{} should be {}".format(l1_shape, CONST_L1_SHAPE_DIM))

    if TILING.get(l1_shape) == []:
        if l1_shape == "AL1_shape":
            data_amount_l1 = (
                reduce(lambda x, y: x * y, DIM_MAP["A_matrix_dim"][2:])
                // TILING["block_dim"][2]
            ) * DIM_MAP[GroupDictKeys.dy_c1_extend]
        if l1_shape == "BL1_shape":
            data_amount_l1 = (
                reduce(lambda x, y: x * y, DIM_MAP["B_matrix_dim"][1:])
                // TILING["block_dim"][1]
            )
    else:
        block_m, block_k, block_n = cce_params.CUBE_MKN[TENSOR_MAP.get("b_l1").dtype][
            "mac"
        ]
        l1_k = TILING.get(l1_shape)[0]
        l1_mn = TILING.get(l1_shape)[1]
        if l1_k == 0 or l1_mn == 0:
            _raise_dx_opti_err("l1_k or l1_mn can not be zero")
        if l1_k % block_k != 0:
            _raise_dx_opti_err("l1_k can not be divided by {}".format(block_k))
        if l1_shape == "AL1_shape":
            data_amount_l1 = (
                l1_k
                * l1_mn
                * TILING.get("CL0_matrix")[1]
                * block_m
                * DTYPE_BYTE_MAP[TENSOR_MAP.get("a_l1").dtype]
            )
        else:
            data_amount_l1 = (
                l1_k
                * l1_mn
                * TILING.get("CL0_matrix")[0]
                * block_n
                * DTYPE_BYTE_MAP[TENSOR_MAP.get("b_l1").dtype]
            )
        if isdouble == 2:
            data_amount_l1 = data_amount_l1 * 2
    _print_debug("{} data_amount_l1:{}".format(l1_shape, int(data_amount_l1) / 1024))
    return data_amount_l1


def _check_tilling_l0(l0_shape, l0_space, isdouble):
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
    if row == 0 or col == 0:
        _raise_dx_opti_err("k, m, n in L0A/B can not be zero")
    data_amount_l0 = (
        row
        * col
        * TILING.get(l0_shape)[2]
        * TILING.get(l0_shape)[3]
        * DTYPE_BYTE_MAP[TENSOR_MAP.get("b_l0b").dtype]
        * isdouble
    )
    _print_debug("data_amount_l0A/B[KB]:", data_amount_l0 / 1024)
    if data_amount_l0 > l0_space:
        _raise_dx_opti_err("tilling size exceed L0A/B Buffer")


def _check_tilling_l0c(l0c_shape, l0c_space, isdouble):
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
    if TILING.get("BL0_matrix") != []:
        bl0_n = TILING.get("BL0_matrix")[1]
        if cl0_m == 0 or cl0_n == 0:
            _raise_dx_opti_err("cl0_m, cl0_n can not be zero")
        if cl0_n != bl0_n:
            _raise_dx_opti_err(
                "axis n in tilling BL0 " "is not equal to axis n in tilling CL0"
            )
    data_amount_cl0 = (
        cl0_m
        * cl0_n
        * TILING.get(l0c_shape)[2]
        * TILING.get(l0c_shape)[3]
        * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_l0c").dtype]
        * isdouble
    )
    _print_debug("data_amount_l0C[KB]:", data_amount_cl0 / 1024)
    if data_amount_cl0 > l0c_space:
        _raise_dx_opti_err("tilling size exceed L0C Buffer")


def _check_tilling_cub(strideh, stridew, cub_space, isdouble, is_conv1d_bool):
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
        block_m, _, _ = cce_params.CUBE_MKN[TENSOR_MAP.get("c_ub").dtype]["mac"]
        if is_conv1d_bool:
            dilate_cub_size = (
                (1 + stridew)
                * nc_factor
                * cl0_m_extent
                * TILING.get("CUB_matrix")[3]
                * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_ub").dtype]
                * isdouble
            )
        else:
            if cl0_m_extent < DIM_MAP["img_shape"][3]:
                _raise_dx_opti_err("mc of CL0_matrix " "smaller than weight of Image")
            if DIM_MAP["img_shape"][3] > block_m:
                check_ifmc_falg = bool(
                    (cl0_m_extent // DIM_MAP["img_shape"][3])
                    * DIM_MAP["img_shape"][3]
                    * strideh
                    * stridew
                    <= CUB_BUFFER_LIMIT
                )
                if (
                    cl0_m_extent % DIM_MAP["img_shape"][3] == 0
                    and check_ifmc_falg
                    and DIM_MAP["img_shape"][2]
                    % (cl0_m_extent // DIM_MAP["img_shape"][3])
                    == 0
                ):
                    n_is_hfactor = cl0_m_extent // DIM_MAP["img_shape"][3]
                else:
                    n_is_hfactor = (cl0_m_extent - block_m) // DIM_MAP["img_shape"][3]
            else:
                check_ifmc_falg_s = False
                if cl0_m_extent % DIM_MAP["img_shape"][3] == 0:
                    n_is_hfactor = cl0_m_extent // DIM_MAP["img_shape"][3]
                    while DIM_MAP["img_shape"][2] % n_is_hfactor != 0:
                        n_is_hfactor = n_is_hfactor - 1
                    check_ifmc_falg_s = bool(
                        n_is_hfactor
                        * DIM_MAP["img_shape"][3]
                        * DIM_MAP["dilate_dim"][0]
                        * DIM_MAP["dilate_dim"][1]
                        > CUB_BUFFER_LIMIT
                    )
                if cl0_m_extent % DIM_MAP["img_shape"][3] != 0 or check_ifmc_falg_s:
                    n_is_hfactor = max((cl0_m_extent - block_m), block_m) // DIM_MAP["img_shape"][3]
                    while DIM_MAP["img_shape"][2] % n_is_hfactor != 0:
                        n_is_hfactor = n_is_hfactor - 1
            real_m = n_is_hfactor * DIM_MAP["img_shape"][3]
            dilate_cub_size = (
                (1 + strideh * stridew)
                * nc_factor
                * real_m
                * TILING.get("CUB_matrix")[3]
                * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_ub").dtype]
                * isdouble
            )
        return dilate_cub_size

    nc_factor, mc_factor = TILING.get("CUB_matrix")[0], TILING.get("CUB_matrix")[1]
    if mc_factor != TILING.get("CL0_matrix")[1]:
        _raise_dx_opti_err("mc_factor is not equal to mc")
    if TILING.get("CL0_matrix")[0] % nc_factor != 0:
        _raise_dx_opti_err("nc_factor is not factor of nc")
    cl0_m_extent = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]
    if strideh > 1 or stridew > 1:
        data_amount_cub = _get_dilate_cub_size()
    else:
        data_amount_cub = (
            nc_factor
            * mc_factor
            * TILING.get("CUB_matrix")[2]
            * TILING.get("CUB_matrix")[3]
            * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_ub").dtype]
            * isdouble
        )
    DeconvParam.update_para_map("DATA_AMOUNT_CUB", data_amount_cub)
    _print_debug(
        "DATA_AMOUNT_CUB[KB]:", DeconvParam.get_para_map("DATA_AMOUNT_CUB") / 1024
    )

    if DeconvParam.get_para_map("DATA_AMOUNT_CUB") > cub_space:
        _raise_dx_opti_err(
            "tilling ub size:{} exceed CUB Buffer:{}".format(
                DeconvParam.get_para_map("DATA_AMOUNT_CUB"), cub_space
            )
        )


def _get_tiling_l0a_l0b(cl0_matrix, l0_matrix, instr):
    """ get l0a and l0b matrix """
    k_dim = DIM_MAP.get("A_matrix_dim")[-3]
    if instr == "A":
        block_m, block_k, block_n = cce_params.CUBE_MKN[TENSOR_MAP.get("a_l0a").dtype][
            "mac"
        ]
        # l0_matrix is bl0_matrix:[kb, nb, n0, k0]
        if l0_matrix != []:
            full_ab = [cl0_matrix[1], l0_matrix[0], block_m, block_k, 1]
        else:
            full_ab = [cl0_matrix[1], k_dim, block_m, block_k, 1]
    elif instr == "B":
        block_m, block_k, block_n = cce_params.CUBE_MKN[TENSOR_MAP.get("b_l0b").dtype][
            "mac"
        ]
        # l0_matrix is al0_matrix:[ma, ka, m0, k0]
        if l0_matrix != []:
            full_ab = [l0_matrix[1], cl0_matrix[0], block_n, block_k, 1]
        else:
            full_ab = [k_dim, cl0_matrix[0], block_n, block_k, 1]
    else:
        _raise_dx_opti_err("instr should be A or B")

    return full_ab


def _check_tilinng_k_l1():
    _, block_k, _ = cce_params.CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"]
    k_al1 = TILING.get("AL1_shape")[0]
    k_bl1 = TILING.get("BL1_shape")[0]
    if k_al1 % k_bl1 != 0 and k_bl1 % k_al1 != 0:
        _raise_dx_opti_err(
            "kal1 should be divisible by kbl1 or kbl1" "should be divisible by kal1 "
        )
    if k_al1 % (TILING.get("AL0_matrix")[1] * block_k) != 0:
        _raise_dx_opti_err("ka should be divisible by kal1")
    if (
        TILING.get("BL0_matrix")
        and k_bl1 % (TILING.get("BL0_matrix")[0] * block_k) != 0
    ):
        _raise_dx_opti_err("kb should be divisible by kbl1")


def _check_tiling_bl0_matrix(manual_pingpong_buffer, data_amount_l1b):
    if TILING.get("BL0_matrix") is None:
        _raise_dx_opti_err("tiling[BL0_matrix] can not be None")
    if TILING.get("BL0_matrix") == []:
        data_amount_l0b = data_amount_l1b
        if data_amount_l0b > cce_conf.get_soc_spec("L0B_SIZE"):
            _raise_dx_opti_err("tiling size exceed L0B Buffer")
    else:
        _check_tilling_l0(
            "BL0_matrix",
            cce_conf.get_soc_spec("L0B_SIZE"),
            manual_pingpong_buffer.get("BL0_pbuffer")
        )
        if TILING.get("AL0_matrix")[1] != TILING.get("BL0_matrix")[0]:
            _raise_dx_opti_err(
                "axis k in tilling AL0 is not " "equal to axis k in tilling BL0"
            )


# check tiling and set default tiling
def _check_and_set_default_tiling(tiling, atype, btype):
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
            "float32": 16,
            "int32": 16,
            "float16": 16,
            "int8": 32
        }
        if atype in bit_dir.keys():
            k_al1 = bit_dir[atype]
            k_al0 = bit_dir[atype]
        else:
            # defaut value 32
            k_al1 = 32
            k_al0 = 32

        if btype in bit_dir.keys():
            k_bl1 = bit_dir[atype]
            k_bl0 = bit_dir[atype]
        else:
            # defaut value 32
            k_bl1 = 32
            k_bl0 = 32
        m_al0 = m_cl0 = 1
        if TENSOR_MAP.get("dilate_ub") is not None:
            # when mmad, the min unit of M is a fmp's w
            dy_w = DIM_MAP["img_shape"][3]
            block_m = cce_params.CUBE_MKN[TENSOR_MAP.get("c_l0c").dtype]["mac"][0]
            # mc's calculation rule refor to auto tiling
            if dy_w % 16 == 0:
                m_al0 = m_cl0 = dy_w // block_m
            else:
                # add one is needed by buffer_tile of ub
                m_al0 = m_cl0 = int_ceil_div(dy_w, block_m) + 1

        tiling["AUB_shape"] = None
        tiling["BUB_shape"] = None
        tiling["AL1_shape"] = [k_al1, 1, 1, 1]
        tiling["BL1_shape"] = [k_bl1, 1, 1, 1]
        tiling["AL0_matrix"] = [m_al0, 1, 16, k_al0, 1, 1]
        tiling["BL0_matrix"] = [1, 1, 16, k_bl0, 1, 1]
        tiling["CL0_matrix"] = [1, m_cl0, 16, 16, 1, 1]
        tiling["CUB_matrix"] = [1, m_cl0, 16, 16, 1, 1]
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


def _get_tiling(  # pylint: disable=R0913,R0914,R0915
    tensor,
    fusion_type,
    kernel_name,
    is_conv1d_bool,
    dynamic_para,
    tiling_case=None
):
    """
    get tilling parameter from get_tilling and check all parameter
    """

    def handle_quant_tiling():
        if fusion_type in (FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
            TILING["CL0_matrix"][0] //= 2
            if TILING["CL0_matrix"][0] == 0:
                TILING["CL0_matrix"][0] = 1
            TILING["CL0_matrix"][3] = 32

            TILING["CUB_matrix"][0] //= 2
            if TILING["CUB_matrix"][0] == 0:
                TILING["CUB_matrix"][0] = 1
            TILING["CUB_matrix"][3] = 32

    def _check_tiling_l1():
        data_amount_l1b = _get_data_amount_l1(
            "BL1_shape", manual_pingpong_buffer.get("BL1_pbuffer")
        )
        if dynamic_para != "dynamic_hw":
            data_amount_l1a = _get_data_amount_l1(
                "AL1_shape", manual_pingpong_buffer.get("AL1_pbuffer")
            )
            if DeconvParam.get_para_map("l1_fusion_type") != -1:
                data_amount_l1a = 0
            if (int(data_amount_l1a) + int(data_amount_l1b)) > cce_conf.get_soc_spec(
                "L1_SIZE"
            ):
                _raise_dx_opti_err("tiling size exceed L1 Buffer")

        if TILING.get("BL1_shape") and TILING.get("AL1_shape"):
            _check_tilinng_k_l1()
        return data_amount_l1b

    _, block_k, block_n = cce_params.CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"]
    # if filter dtype is int8, than channel block_size is 32
    if tensor.dtype == "int32" or fusion_type in (
        FUSION_DX_DEQUANT,
        FUSION_DX_DEQUANT_QUANT,
        FUSION_DX_REQUANT
    ):
        # DIM_MAP["filter_shape"] : co_dim, ci_dim, _, _
        filter_shape_g = [
            DIM_MAP[GroupDictKeys.dy_c1_extend] * block_k,
            DIM_MAP[GroupDictKeys.dx_c1_extend],
            1,
            1,
            block_n]
    else:
        # DIM_MAP["filter_shape"] : ci_dim, co_dim, _, _
        filter_shape_g = [
            DIM_MAP[GroupDictKeys.dy_c1_extend] * block_k,
            DIM_MAP[GroupDictKeys.dx_c1_extend],
            1,
            1,
            block_n]

    if TENSOR_MAP.get("dilate_ub") is None:
        strideh, stridew = 1, 1
    else:
        strideh, stridew = DIM_MAP["dilate_dim"]
    # times of the dx ub space
    _calc_double_op_num(fusion_type)

    if fusion_type in (FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
        filter_shape_g[1] = (filter_shape_g[1] + 1) // 2 * 2

    bias_flag = _get_bias_flag()

    global TILING  # pylint:disable=W0603
    if not TILING:
        if dynamic_para is None:
            in_fm_memory_type = DeconvParam.get_para_map("input_memory_type")
            out_fm_memory_type = DeconvParam.get_para_map("output_memory_type")
            info_dict = {
                "op_type": "conv2d_backprop_input",
                "A_shape": list(DIM_MAP["dy_6GD_shape"][1:]),
                "B_shape": list(filter_shape_g),
                "C_shape": None,
                "A_dtype": str(TENSOR_MAP["img_placehold"].dtype),
                "B_dtype": str(TENSOR_MAP["filter_placehold"].dtype),
                "C_dtype": str(tensor.dtype),
                "mad_dtype": str(TENSOR_MAP["c_l0c"].dtype),
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
                "group": DIM_MAP[GroupDictKeys.g_extend],
                "bias_flag": bias_flag,
                "fused_double_operand_num": FUSION_TYPE_2_OPERAND_NUM.get(fusion_type),
                "kernel_name": kernel_name.value,
                "in_fm_memory_type": in_fm_memory_type,
                "out_fm_memory_type": out_fm_memory_type,
                "l1_fusion_type": DeconvParam.get_para_map("l1_fusion_type"),
                "fusion_type": DeconvParam.get_para_map("fusion_type_num")
            }
            TILING = get_tiling(info_dict)
        else:
            TILING = deepcopy(tiling_case)
    TILING = _check_and_set_default_tiling(
        TILING, TENSOR_MAP["img_placehold"].dtype, TENSOR_MAP["filter_placehold"].dtype
    )

    _print_debug(
        "opti dx shape, kernel_name:",
        kernel_name,
        "filter:",
        filter_shape_g,
        "dy:",
        DIM_MAP["img_shape"],
        "dx:",
        DIM_MAP["out_img_shape"]
    )
    _print_debug("tiling:", TILING)

    if TILING.get("AL0_matrix") == []:
        TILING["AL0_matrix"] = _get_tiling_l0a_l0b(
            TILING.get("CL0_matrix"), TILING.get("BL0_matrix"), "A"
        )

    if TILING.get("BL0_matrix") == []:
        TILING["BL0_matrix"] = _get_tiling_l0a_l0b(
            TILING.get("CL0_matrix"), TILING.get("AL0_matrix"), "B"
        )

    manual_pingpong_buffer = TILING.get("manual_pingpong_buffer")

    data_amount_l1b = _check_tiling_l1()

    # check tilling in AL0 BL0
    if TILING.get("AL0_matrix") is None or TILING.get("AL0_matrix") == []:
        _raise_dx_opti_err("tiling[AL0_matrix] can not be None or []")
    _check_tilling_l0(
        "AL0_matrix",
        cce_conf.get_soc_spec("L0A_SIZE"),
        manual_pingpong_buffer.get("AL0_pbuffer")
    )

    _check_tiling_bl0_matrix(manual_pingpong_buffer, data_amount_l1b)

    # check tilling in CL0
    _check_tilling_l0c(
        "CL0_matrix",
        cce_conf.get_soc_spec("L0C_SIZE"),
        manual_pingpong_buffer.get("CL0_pbuffer")
    )

    # check tilling in CUB  attention:light when stride get  #########
    cube_vector_split_flag = cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT")
    if (dynamic_para != "dynamic_hw") and (not cube_vector_split_flag):
        _check_tilling_cub(
            strideh,
            stridew,
            cce_conf.get_soc_spec("UB_SIZE"),
            manual_pingpong_buffer.get("CUB_pbuffer"),
            is_conv1d_bool
        )

    handle_quant_tiling()


def _get_bias_flag():
    if (
        TENSOR_MAP.get("bias_add_vector") is not None
        or TENSOR_MAP.get("c_add_bias") is not None
    ):
        bias_flag = 1
    else:
        bias_flag = 0
    return bias_flag


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


def _quant_tensor_info(dx_res, quant_para):
    """
    check dequant + quant
    """

    def input_tensor_info(tensor_info):
        input_tensor = _get_src_tensor(tensor_info, 0)
        if not input_tensor.op.input_tensors:
            input_cache_buffer.append([input_tensor, tensor_info])
            tensor_info = _get_src_tensor(tensor_info, 1)
        else:
            input_tensor = _get_src_tensor(tensor_info, 1)
            input_cache_buffer.append([input_tensor, tensor_info])
            tensor_info = _get_src_tensor(tensor_info, 0)
        return tensor_info

    DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_DEQUANT_QUANT)
    quant_para["q_round"] = dx_res.op.attrs["round_mode"].value
    temp_tensor = dx_res
    elewise_cache_list = []
    input_cache_buffer = []

    while temp_tensor.op.input_tensors:
        if temp_tensor.op.name == "cast_i8_ub":
            TENSOR_MAP["cast_i8_ub"] = temp_tensor
        elif "reform" in temp_tensor.op.name:
            TENSOR_MAP["reform_op"] = temp_tensor
        elif temp_tensor.op.name in ("scale_sqrt_ub", "offset_ub", "dequant_relu"):
            elewise_cache_list.append(temp_tensor)
        elif temp_tensor.op.name == "input_ub":
            TENSOR_MAP["input_ub"] = temp_tensor
            if temp_tensor.op.attrs["c_out"].value % 2:
                quant_para["q_padding"] = True
        elif len(temp_tensor.op.input_tensors) == 2 and "elewise" in temp_tensor.op.tag:
            temp_tensor = input_tensor_info(temp_tensor)
            continue
        elif "elewise" in temp_tensor.op.tag:
            elewise_cache_list.append(temp_tensor)
        elif "dequant2" in temp_tensor.op.tag:
            TENSOR_MAP["dequant_sqrt"] = temp_tensor
        elif temp_tensor.op.tag in ("dequant_vector", "dequant1_vector"):
            TENSOR_MAP["deq"] = _get_src_tensor(temp_tensor, 1)
            TENSOR_MAP["c_ub"] = temp_tensor
            quant_para["deq_vector"] = True
        elif temp_tensor.op.tag in ("dequant_scale", "dequant1_scale"):
            TENSOR_MAP["deq"] = _get_src_tensor(temp_tensor, 1)
            TENSOR_MAP["c_ub"] = temp_tensor
        elif "dequant_remove_pad" in temp_tensor.op.tag:
            TENSOR_MAP["dequant_remove_pad"] = temp_tensor
        temp_tensor = _get_src_tensor(temp_tensor, 0)
    TENSOR_MAP["elewise_tensor"] = elewise_cache_list
    TENSOR_MAP["input_tensor"] = input_cache_buffer
    return quant_para


def _dequant_tensor_info(dx_res, quant_para):
    """
    check dequant
    """
    DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_DEQUANT)
    temp_tensor = dx_res
    while temp_tensor.op.input_tensors:
        if "dequant2" in temp_tensor.op.tag:
            TENSOR_MAP["dequant_sqrt"] = temp_tensor
        elif temp_tensor.op.tag in ("dequant_vector", "dequant1_vector"):
            TENSOR_MAP["deq"] = _get_src_tensor(temp_tensor, 1)
            TENSOR_MAP["c_ub"] = temp_tensor
            quant_para["deq_vector"] = True
        elif temp_tensor.op.tag in ("dequant_scale", "dequant1_scale"):
            TENSOR_MAP["deq"] = _get_src_tensor(temp_tensor, 1)
            TENSOR_MAP["c_ub"] = temp_tensor
        elif temp_tensor.op.name == "dequant_relu":
            TENSOR_MAP["dequant_relu"] = temp_tensor
        elif "dequant_remove_pad" in temp_tensor.op.tag:
            TENSOR_MAP["dequant_remove_pad"] = temp_tensor
        temp_tensor = _get_src_tensor(temp_tensor, 0)
    return quant_para


def _requant_tesnor_info(dx_res, quant_para):
    """
    check requant
    """
    DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_REQUANT)
    TENSOR_MAP["data_transfer"] = _get_src_tensor(dx_res, 0)
    TENSOR_MAP["c_ub"] = _get_src_tensor(TENSOR_MAP["data_transfer"], 0)
    if "vector" in TENSOR_MAP["c_ub"].op.tag:
        quant_para["req_vector"] = True
    TENSOR_MAP["deq"] = _get_src_tensor(TENSOR_MAP["c_ub"], 1)
    return quant_para


def _dequant_elewise_infor(dx_res, quant_para):
    """
    check dequant + elewise
    """
    temp_tensor = dx_res
    elewise_cache_list = []
    input_cache_buffer = []
    while temp_tensor.op.input_tensors:
        if "dequant" in temp_tensor.op.tag:
            DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_DEQUANT)
        if len(temp_tensor.op.input_tensors) == 2 and "elewise" in temp_tensor.op.tag:
            input_tensor = _get_src_tensor(temp_tensor, 0)
            if not input_tensor.op.input_tensors:
                input_cache_buffer.append([input_tensor, temp_tensor])
                temp_tensor = _get_src_tensor(temp_tensor, 1)
                continue
            input_tensor = _get_src_tensor(temp_tensor, 1)
            input_cache_buffer.append([input_tensor, temp_tensor])
        elif "elewise" in temp_tensor.op.tag:
            elewise_cache_list.append(temp_tensor)
        elif "dequant2" in temp_tensor.op.tag:
            TENSOR_MAP["dequant_sqrt"] = temp_tensor
        elif temp_tensor.op.tag in ("dequant_vector", "dequant1_vector"):
            TENSOR_MAP["deq"] = _get_src_tensor(temp_tensor, 1)
            TENSOR_MAP["c_ub"] = temp_tensor
            quant_para["deq_vector"] = True
        elif temp_tensor.op.tag in ("dequant_scale", "dequant1_scale"):
            TENSOR_MAP["deq"] = _get_src_tensor(temp_tensor, 1)
            TENSOR_MAP["c_ub"] = temp_tensor
        elif temp_tensor.op.name == "dequant_relu":
            TENSOR_MAP["dequant_relu"] = temp_tensor
        elif "dequant_remove_pad" in temp_tensor.op.tag:
            TENSOR_MAP["dequant_remove_pad"] = temp_tensor
        temp_tensor = _get_src_tensor(temp_tensor, 0)
    if DeconvParam.get_para_map("FUSION_TYPE") == FUSION_DX_DEQUANT:
        TENSOR_MAP["elewise_tensor"] = elewise_cache_list
        TENSOR_MAP["input_tensor"] = input_cache_buffer
    return quant_para


def _check_quant_fusion(dx_res):
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
    if dx_res.op.tag == "quant":
        quant_para = _quant_tensor_info(dx_res, quant_para)
    elif "dequant" in dx_res.op.tag:
        quant_para = _dequant_tensor_info(dx_res, quant_para)
    elif "requant_remove_pad" in dx_res.op.tag:
        quant_para = _requant_tesnor_info(dx_res, quant_para)
    else:
        quant_para = _dequant_elewise_infor(dx_res, quant_para)

    return quant_para


def _set_data_layout(
    res, dex_res, sch, dynamic_para, var_range
):  # pylint: disable=R0914,R0915,R0912
    """
    get DIM_MAP which contains all ops

    Parameter:
    ----------------------------------------------------------
    :param res: op
    :param dex_res: op
    :param sch: schedule
    :param dynamic_para: string, "dynamic_hw" or "dynamic_batch"
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
        global TENSOR_MAP  # pylint: disable=W0603
        tensor_add_left = tensor_add_res.op.input_tensors[0]
        tensor_add_right = tensor_add_res.op.input_tensors[1]
        if tensor_add_left.op.tag == "conv2d_backprop_input_opti":
            tensor_dx_gm = tensor_add_left
            tensor_add_input_gm = tensor_add_right
        else:
            tensor_dx_gm = tensor_add_right
            tensor_add_input_gm = tensor_add_left

        tensor_add_input_ub = sch.cache_read(
            tensor_add_input_gm, cce_params.scope_ubuf, [tensor_add_res]
        )
        TENSOR_MAP["add_input_ub"] = tensor_add_input_ub
        return tensor_dx_gm

    def _check_dx_fusion_type(res, fusion_tensor_map):  # pylint: disable=R0915
        """
        check fusion type and set buffer
        """

        def handle_elewise_tensor():
            elewise_tensor = fusion_tensor_map.get("elewise_tensor")
            if elewise_tensor:
                for elewise_tensor_mem in elewise_tensor:
                    if elewise_tensor_mem.op.tag != res.op.tag:
                        sch[elewise_tensor_mem].set_scope(cce_params.scope_ubuf)

            input_tensor = fusion_tensor_map.get("input_tensor")
            if input_tensor:
                for input_tensor_mem in input_tensor:
                    if input_tensor_mem[1].op.tag != res.op.tag:
                        sch[input_tensor_mem[1]].set_scope(cce_params.scope_ubuf)
                        input_tensor_ub = sch.cache_read(
                            input_tensor_mem[0],
                            cce_params.scope_ubuf,
                            input_tensor_mem[1]
                        )
                    else:
                        input_tensor_ub = sch.cache_read(
                            input_tensor_mem[0], cce_params.scope_ubuf, c_ub_res
                        )
                    input_tensor_mem[0] = input_tensor_ub

        def handle_deq_tensor():
            deq_scale = fusion_tensor_map.get("deq")

            if "dequant_sqrt" in fusion_tensor_map:
                dequant_sqrt = fusion_tensor_map.get("dequant_sqrt")
                if (
                    "c_ub_res" in fusion_tensor_map
                    and c_ub_res.op.tag == dequant_sqrt.op.tag
                ):
                    dequant_sqrt = c_ub_res
                deq_ub = sch.cache_read(
                    deq_scale,
                    cce_params.scope_ubuf,
                    [fusion_tensor_map.get("c_ub"), dequant_sqrt]
                )
            else:
                deq_ub = sch.cache_read(
                    deq_scale, cce_params.scope_ubuf, fusion_tensor_map.get("c_ub")
                )
            fusion_tensor_map["deq"] = deq_ub

        def handle_elewise_fusion():
            DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_ELEWISE)
            temp_tensor = res
            input_tensor_list = []
            ub_list = []
            c_ub_res = sch.cache_write(res, cce_params.scope_ubuf)
            while temp_tensor.op.input_tensors:
                if len(temp_tensor.op.input_tensors) == 2:
                    if temp_tensor != res:
                        sch[temp_tensor].set_scope(cce_params.scope_ubuf)
                        input_tensor_des = temp_tensor
                    else:
                        input_tensor_des = c_ub_res
                    ub_list.append(input_tensor_des)
                    input_tensor = temp_tensor.op.input_tensors[0]
                    if not input_tensor:
                        temp_tensor = temp_tensor.op.input_tensors[1]
                    else:
                        input_tensor = temp_tensor.op.input_tensors[1]
                        temp_tensor = temp_tensor.op.input_tensors[0]
                    input_tensor_ub = sch.cache_read(
                        input_tensor, cce_params.scope_ubuf, [input_tensor_des]
                    )
                    input_tensor_list.append(input_tensor_ub)
                    continue
                if temp_tensor.op.tag == "conv2d_backprop_input_opti":
                    tensor_dx_gm = temp_tensor
                    sch[tensor_dx_gm].compute_inline()
                    break
                if temp_tensor != res:
                    sch[temp_tensor].set_scope(cce_params.scope_ubuf)
                    ub_list.append(temp_tensor)
                else:
                    ub_list.append(c_ub_res)
                temp_tensor = temp_tensor.op.input_tensors[0]
            fusion_tensor_map["input_tensor_list"] = input_tensor_list
            fusion_tensor_map["ub_list"] = ub_list
            fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm
            return tensor_dx_gm

        if DeconvParam.get_para_map("FUSION_TYPE") in (
            FUSION_DX_DEQUANT,
            FUSION_DX_DEQUANT_QUANT,
            FUSION_DX_REQUANT
        ):
            if (
                res.op.tag != "dequant_remove_pad"
                and DeconvParam.get_para_map("FUSION_TYPE") == FUSION_DX_DEQUANT
            ):
                c_ub_res = sch.cache_write(res, cce_params.scope_ubuf)
                fusion_tensor_map["c_ub_res"] = c_ub_res

            for tensor in fusion_tensor_map:
                if (
                    tensor not in ("elewise_tensor", "input_tensor", "deq")
                    and res.op.tag != fusion_tensor_map[tensor].op.tag
                ):
                    sch[fusion_tensor_map[tensor]].set_scope(cce_params.scope_ubuf)

            handle_elewise_tensor()
            handle_deq_tensor()
            tensor_dx_gm = fusion_tensor_map["c_ub"].op.input_tensors[0]
        elif res.op.tag == "emit_insn_elewise_multiple_sel|bool":
            drelu_gm = res
            # dx+add+drelu
            if "elewise_binary_add" in drelu_gm.op.input_tensors[1].op.tag:
                DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_ADD_DRELU)
                tensor_add_res = drelu_gm.op.input_tensors[1]
                sch[tensor_add_res].set_scope(cce_params.scope_ubuf)
                TENSOR_MAP["add_res_ub"] = tensor_add_res
                tensor_dx_gm = _get_tensor_dx_gm(tensor_add_res)
            # dx+drelu
            else:
                DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_DRELU)
                tensor_dx_gm = drelu_gm.op.input_tensors[1]

            tensor_bitmask_gm = drelu_gm.op.input_tensors[0]
            sch[tensor_dx_gm].set_scope(cce_params.scope_ubuf)
            tensor_bitmask = sch.cache_read(
                tensor_bitmask_gm, cce_params.scope_ubuf, [drelu_gm]
            )
            tensor_drelu = sch.cache_write(drelu_gm, cce_params.scope_ubuf)

            fusion_tensor_map["bitmask_ub"] = tensor_bitmask
            fusion_tensor_map["drelu_ub"] = tensor_drelu
            fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm  # inter_gm
        # dx+add
        elif "elewise" in res.op.tag:
            tensor_dx_gm = handle_elewise_fusion()
        elif res.op.tag == "conv2d_backprop_input_opti":
            DeconvParam.update_para_map("FUSION_TYPE", FUSION_NONE)
            tensor_dx_gm = res
        else:
            _raise_dx_opti_err(DX_SUPPORT_TAG_LOG_PREFIX + " unsupported data flow")
        return tensor_dx_gm, fusion_tensor_map

    def _get_ub_tensor(fusion_type):
        if tensor_dx_gm.op.input_tensors[0].op.name == "bias_add_vector":
            bias_add_vector = tensor_dx_gm.op.input_tensors[0]
            tensor_dilate_ub = bias_add_vector.op.input_tensors[0]
            tensor_bias = bias_add_vector.op.input_tensors[1]
            sch[bias_add_vector].set_scope(cce_params.scope_ubuf)
            bias_ub = sch.cache_read(
                tensor_bias, cce_params.scope_ubuf, [bias_add_vector]
            )
            TENSOR_MAP["bias_add_vector"] = bias_add_vector
            TENSOR_MAP["bias_ub"] = bias_ub
        else:
            tensor_dilate_ub = tensor_dx_gm.op.input_tensors[0]

        if (
            tensor_dilate_ub is not None
            and tensor_dilate_ub.op.tag == "conv2d_backprop_input_opti"
        ):
            TENSOR_MAP["dilate_ub"] = tensor_dilate_ub
            sch[tensor_dilate_ub].set_scope(cce_params.scope_ubuf)
            if dynamic_para == "dynamic_hw":
                tensor_vn = tensor_dilate_ub.op.input_tensors[1]
                TENSOR_MAP["tensor_vn"] = tensor_vn
                sch[tensor_vn].set_scope(cce_params.scope_ubuf)
                tensor_cub = tensor_dilate_ub.op.input_tensors[0]
                tensor_fillling_zero = tensor_vn.op.input_tensors[0]
                TENSOR_MAP["tensor_fillling_zero"] = tensor_fillling_zero
                sch[tensor_fillling_zero].set_scope(cce_params.scope_ubuf)
                tensor_fillling_one = tensor_vn.op.input_tensors[1]
                TENSOR_MAP["tensor_fillling_one"] = tensor_fillling_one
                sch[tensor_fillling_one].set_scope(cce_params.scope_ubuf)
            else:
                tensor_cub = tensor_dilate_ub.op.input_tensors[0]
                tensor_fillling_zero = tensor_dilate_ub.op.input_tensors[1]
                TENSOR_MAP["tensor_fillling_zero"] = tensor_fillling_zero
                sch[tensor_fillling_zero].set_scope(cce_params.scope_ubuf)
        else:
            if tensor_dx_gm.op.input_tensors[0].op.name == "bias_add_vector":
                tensor_cub = tensor_dilate_ub
            else:
                tensor_cub = tensor_dx_gm.op.input_tensors[0]
        if fusion_type in (
            FUSION_DX_DEQUANT,
            FUSION_DX_DEQUANT_QUANT,
            FUSION_DX_REQUANT
        ):
            tensor_cub = TENSOR_MAP["c_ub"]

        return tensor_cub

    def _get_l1_fusion_para():
        fusion_para = DeConvKernelSize1Pattern.fusion_para
        DeconvParam.update_para_map(
            "input_memory_type", [fusion_para["input_memory_type"]]
        )
        DeconvParam.update_para_map("valid_shape", fusion_para["valid_shape"])
        DeconvParam.update_para_map("slice_offset", fusion_para["slice_offset"])
        DeconvParam.update_para_map("l1_fusion_type", fusion_para["l1_fusion_type"])
        DeconvParam.update_para_map(
            "fmap_l1_addr_flag", fusion_para["fmap_l1_addr_flag"]
        )
        DeconvParam.update_para_map(
            "fmap_l1_valid_size", fusion_para["fmap_l1_valid_size"]
        )

        load3d_flag = bool(fusion_para["l1_fusion_type"] != -1)
        DeconvParam.update_para_map("load3d_flag", load3d_flag)
        out_mem = fusion_para["output_memory_type"]
        if out_mem == "fuse_flag":
            if "addr_type" in dex_res.op.attrs:
                res_addr_type = dex_res.op.attrs["addr_type"].value
            else:
                res_addr_type = 0
            output_memory_type = [res_addr_type]
            if res_addr_type == 1:
                sch[dex_res].set_scope(cce_params.scope_cbuf_fusion)
        else:
            if out_mem == 1:
                sch[dex_res].set_scope(cce_params.scope_cbuf_fusion)
            output_memory_type = [out_mem]
        DeconvParam.update_para_map("output_memory_type", output_memory_type)

    def _al1_fusion_handle():
        if (
            DeconvParam.get_para_map("load3d_flag")
            and DeconvParam.get_para_map("input_memory_type")[0] == 1
        ):
            a_l0a_before = a_l0a.op.input_tensors[0]
            dedy = a_l0a_before.op.input_tensors[0]
            dedy_col = sch.cache_read(
                dedy, cce_params.scope_cbuf_fusion, [a_l0a_before]
            )
            sch[dedy].set_scope(cce_params.scope_cbuf_fusion)
            TENSOR_MAP["a_l0a_before"] = a_l0a_before
            al1_shape = dedy_col.shape
            sch[dedy_col].buffer_align(
                (1, 1),
                (1, 1),
                (al1_shape[2], al1_shape[2]),
                (al1_shape[3], al1_shape[3]),
                (1, 1)
            )
        elif (
            DeconvParam.get_para_map("load3d_flag")
            and DeconvParam.get_para_map("input_memory_type")[0] == 0
        ):
            a_l0a_before = a_l0a.op.input_tensors[0]
            dedy_col = a_l0a_before.op.input_tensors[0]
            dedy = dedy_col.op.input_tensors[0]
            sch[dedy_col].set_scope(cce_params.scope_cbuf_fusion)
            TENSOR_MAP["a_l0a_before"] = a_l0a_before
            if DeconvParam.get_para_map("l1_fusion_type") == 1:
                al1_shape = dedy_col.shape
                sch[dedy_col].buffer_align(
                    (1, 1),
                    (1, 1),
                    (al1_shape[2], al1_shape[2]),
                    (al1_shape[3], al1_shape[3]),
                    (1, 1)
                )
        else:
            dedy_col = a_l0a.op.input_tensors[0]
            dedy = dedy_col.op.input_tensors[0]
            sch[dedy_col].set_scope(cce_params.scope_cbuf)
            a_l0a_before = None
        return dedy_col, dedy, a_l0a_before

    def al1_buffer_align():
        storage_align_size = 256
        if dedy_col.dtype == "int8":
            storage_align_size = 512

        if DeconvParam.get_para_map("load3d_flag"):
            sch[a_l0a_before].set_scope(cce_params.scope_cbuf)
            dx_w = DIM_MAP["img_shape"][3]
            sch[a_l0a_before].buffer_align(
                (1, 1),
                (dx_w, dx_w),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, cce_params.CUBE_MKN[a_l0a_before.dtype]["mac"][1])
            )
        else:
            sch[dedy_col].storage_align(sch[dedy_col].op.axis[1], storage_align_size, 0)

    global TENSOR_MAP  # pylint: disable=W0603
    global DIM_MAP  # pylint: disable=W0603
    # L1 fusion write select

    _print_debug("dx fusion tag:", res.op.tag)
    tensor_dx_gm, TENSOR_MAP = _check_dx_fusion_type(res, TENSOR_MAP)
    cube_vector_split_flag = cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT")
    DeconvParam.update_para_map("cube_vector_split_flag", cube_vector_split_flag)
    _get_l1_fusion_para()
    fusion_type = DeconvParam.get_para_map("FUSION_TYPE")

    # get tensor of ub by fusion_type
    if DeconvParam.get_para_map("cube_vector_split_flag"):
        tensor_cub = None
    else:
        tensor_cub = _get_ub_tensor(fusion_type)

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
            bias_ub = sch.cache_read(tensor_bias, cce_params.scope_ubuf, [bias_ub_brc])
            TENSOR_MAP["c_add_bias"] = c_add_bias
            TENSOR_MAP["bias_l0c"] = bias_l0c
            TENSOR_MAP["bias_ub_brc"] = bias_ub_brc
            TENSOR_MAP["bias_ub"] = bias_ub
            TENSOR_MAP["tensor_bias"] = tensor_bias
        else:
            tensor_mmad = tensor_cub.op.input_tensors[0]
    else:
        if DeconvParam.get_para_map("cube_vector_split_flag"):
            tensor_mmad = tensor_dx_gm.op.input_tensors[0]

    a_l0a = tensor_mmad.op.input_tensors[0]
    dedy_col, dedy, a_l0a_before = _al1_fusion_handle()
    weight_l0 = tensor_mmad.op.input_tensors[1]
    weight_l1 = weight_l0.op.input_tensors[0]

    # set scope
    if fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT, FUSION_DX_REQUANT):
        tensor_cub = TENSOR_MAP["c_ub"]
    if tensor_cub is not None:
        sch[tensor_cub].set_scope(cce_params.scope_ubuf)
        TENSOR_MAP["c_ub"] = tensor_cub

    # TO DO
    TENSOR_MAP["img_placehold"] = dedy
    if dynamic_para is not None:
        DIM_MAP["img_shape"] = shape_to_list(TENSOR_MAP["img_placehold"].shape)
    else:
        DIM_MAP["img_shape"] = [int(i) for i in TENSOR_MAP["img_placehold"].shape]
    al1_buffer_align()

    TENSOR_MAP["a_l1"] = dedy_col
    sch[a_l0a].set_scope(cce_params.scope_ca)

    l0a_m0, l0a_k0, _ = cce_params.CUBE_MKN[a_l0a.dtype]["mac"]
    sch[a_l0a].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, l0a_m0), (1, l0a_k0))
    TENSOR_MAP["a_l0a"] = a_l0a
    if TENSOR_MAP.get("c_add_bias") is not None:
        sch[c_add_bias].set_scope(cce_params.scope_cc)
        sch[bias_l0c].set_scope(cce_params.scope_cc)
        sch[bias_ub_brc].set_scope(cce_params.scope_ubuf)
    sch[weight_l1].set_scope(cce_params.scope_cbuf)
    TENSOR_MAP["b_l1"] = weight_l1
    sch[weight_l0].set_scope(cce_params.scope_cb)
    TENSOR_MAP["b_l0b"] = weight_l0

    sch[tensor_mmad].set_scope(cce_params.scope_cc)
    TENSOR_MAP["c_l0c"] = tensor_mmad
    TENSOR_MAP["c_gm"] = dex_res
    TENSOR_MAP["filter_placehold"] = weight_l1.op.input_tensors[0]

    # fill in dimmap
    DIM_MAP["group_dict"] =  tensor_dx_gm.op.attrs["group_dict"]
    group_dict_map = DIM_MAP["group_dict"]
    DIM_MAP[GroupDictKeys.g_extend] = group_dict_map[GroupDictKeys.g_extend].value
    DIM_MAP[GroupDictKeys.dy_c1_extend] = group_dict_map[GroupDictKeys.dy_c1_extend].value
    DIM_MAP[GroupDictKeys.dx_c1_extend] = group_dict_map[GroupDictKeys.dx_c1_extend].value
    if dynamic_para is not None:
        DIM_MAP["out_img_shape"] = shape_to_list(res.shape)
        DIM_MAP["img_shape"] = shape_to_list(TENSOR_MAP["img_placehold"].shape)
        DIM_MAP["A_matrix_dim"] = list(dedy_col.shape)
        DIM_MAP["B_matrix_dim"] = list(weight_l0.shape)
        DIM_MAP["filter_shape"] = list(weight_l1.op.input_tensors[0].shape)
        DIM_MAP["dx_5D_shape"] = list(tensor_dx_gm.op.attrs["dx_5D_shape"])
        DIM_MAP["dx_6GD_shape"] = [DIM_MAP[GroupDictKeys.g_extend],
                                  DIM_MAP["dx_5D_shape"][0],
                                  DIM_MAP[GroupDictKeys.dx_c1_extend]] + DIM_MAP["dx_5D_shape"][2:]
    else:
        DIM_MAP["out_img_shape"] = [int(i) for i in res.shape]
        DIM_MAP["img_shape"] = [int(i) for i in TENSOR_MAP["img_placehold"].shape]
        DIM_MAP["A_matrix_dim"] = [int(i) for i in dedy_col.shape]
        DIM_MAP["B_matrix_dim"] = [int(i) for i in weight_l0.shape]
        DIM_MAP["filter_shape"] = [int(i) for i in weight_l1.op.input_tensors[0].shape]
        DIM_MAP["dx_5D_shape"] = [int(i) for i in tensor_dx_gm.op.attrs["dx_5D_shape"]]
        DIM_MAP["dx_6GD_shape"] = [DIM_MAP[GroupDictKeys.g_extend],
                                  DIM_MAP["dx_5D_shape"][0],
                                  DIM_MAP[GroupDictKeys.dx_c1_extend]] + DIM_MAP["dx_5D_shape"][2:]

        DIM_MAP["dy_6GD_shape"] = [DIM_MAP[GroupDictKeys.g_extend],
                                   DIM_MAP["img_shape"][0],
                                   DIM_MAP[GroupDictKeys.dy_c1_extend]]+ DIM_MAP["img_shape"][2:]


    if TENSOR_MAP.get("dilate_ub") is not None:
        if dynamic_para == "dynamic_hw":
            DIM_MAP["dilate_dim"] = list(TENSOR_MAP["dilate_ub"].op.attrs["dilate"])
            DIM_MAP["out_hwdim"] = list(TENSOR_MAP["dilate_ub"].op.attrs["out_hw"])
        else:
            DIM_MAP["dilate_dim"] = [
                int(i) for i in TENSOR_MAP["dilate_ub"].op.attrs["dilate"]
            ]
            DIM_MAP["out_hwdim"] = [
                int(i) for i in TENSOR_MAP["dilate_ub"].op.attrs["out_hw"]
            ]

    if dynamic_para == "dynamic_hw":
        sch.set_var_range(DIM_MAP["img_shape"][2], *var_range.get("dedy_h"))
        sch.set_var_range(DIM_MAP["img_shape"][3], *var_range.get("dedy_w"))
        sch.set_var_range(
            shape_to_list(res.op.attrs["dx_5D_shape"])[2], *var_range.get("dx_h")
        )
        sch.set_var_range(
            shape_to_list(res.op.attrs["dx_5D_shape"])[3], *var_range.get("dx_w")
        )
    elif dynamic_para == "dynamic_batch":
        sch.set_var_range(DIM_MAP["img_shape"][0], *var_range.get("batch_n"))
        sch.set_var_range(
            shape_to_list(res.op.attrs["dx_5D_shape"])[0], *var_range.get("batch_n")
        )

    return tensor_dx_gm


def _get_aicore_tiling_factor(
    is_conv1d_bool, dynamic_para, sch
):  # pylint: disable=R0915
    """
    using tilling parameter calculate factor

    :return: tilling factor from ub to ddr
         tilling factor from l0c to ub
         tilling factor from ddr to AL1
         tilling factor from ddr to Bl1
    """

    def _get_undilate_loc_m(l0c_tiling_factor):

        if l0c_tiling_factor[1] < DIM_MAP.get("img_shape")[-2]:
            _raise_dx_opti_err("mc of CL0_matrix small than weight of Image")
        if DIM_MAP["img_shape"][3] > block_m:
            check_ifmc_falg = bool(
                (mc_from_tiling // DIM_MAP["img_shape"][3])
                * DIM_MAP["img_shape"][3]
                * DIM_MAP["dilate_dim"][0]
                * DIM_MAP["dilate_dim"][1]
                <= CUB_BUFFER_LIMIT
            )
            if (
                mc_from_tiling % DIM_MAP["img_shape"][3] == 0
                and check_ifmc_falg
                and DIM_MAP["img_shape"][2]
                % (mc_from_tiling // DIM_MAP["img_shape"][3])
                == 0
            ):
                n_is_hfactor = (mc_from_tiling) // DIM_MAP["img_shape"][3]
            else:
                n_is_hfactor = (mc_from_tiling - block_m) // DIM_MAP["img_shape"][3]
        else:
            check_ifmc_falg_s = False
            if mc_from_tiling % DIM_MAP["img_shape"][3] == 0:
                n_is_hfactor = mc_from_tiling // DIM_MAP["img_shape"][3]
                while DIM_MAP["img_shape"][2] % n_is_hfactor != 0:
                    n_is_hfactor = n_is_hfactor - 1
                check_ifmc_falg_s = bool(
                    n_is_hfactor
                    * DIM_MAP["img_shape"][3]
                    * DIM_MAP["dilate_dim"][0]
                    * DIM_MAP["dilate_dim"][1]
                    > CUB_BUFFER_LIMIT
                )
            if mc_from_tiling % DIM_MAP["img_shape"][3] != 0 or check_ifmc_falg_s:
                n_is_hfactor = max((mc_from_tiling - block_m), block_m) // DIM_MAP["img_shape"][3]
                while DIM_MAP["img_shape"][2] % n_is_hfactor != 0:
                    n_is_hfactor = n_is_hfactor - 1

        l0c_tiling_factor[1] = (
            DIM_MAP.get("out_hwdim")[1] * n_is_hfactor * DIM_MAP["dilate_dim"][0]
        )
        if l0c_tiling_factor[1] == 0:
            _raise_dx_opti_err("nw can not be zero")
        undilate_l0c_m = n_is_hfactor * DIM_MAP["img_shape"][3]
        return undilate_l0c_m

    def _get_undilate_loc_m_dynamic(l0c_tiling_factor, sch):
        n_is_hfactor = tvm.var("n_is_hfactor")
        sch.set_var_value(
            n_is_hfactor,
            tvm.select(
                tvm.all(
                    (mc_from_tiling // DIM_MAP["img_shape"][3])
                    * DIM_MAP["img_shape"][3]
                    == mc_from_tiling,
                    (mc_from_tiling // DIM_MAP["img_shape"][3])
                    * DIM_MAP["img_shape"][3]
                    * DIM_MAP["dilate_dim"][0]
                    * DIM_MAP["dilate_dim"][1]
                    <= CUB_BUFFER_LIMIT,
                    DIM_MAP["img_shape"][2]
                    % (mc_from_tiling // DIM_MAP["img_shape"][3])
                    == 0
                ),
                mc_from_tiling // DIM_MAP["img_shape"][3],
                (mc_from_tiling - block_m) // DIM_MAP["img_shape"][3]
            )
        )
        sch.set_var_range(n_is_hfactor, 1, mc_from_tiling)
        l0c_tiling_factor[1] = (
            DIM_MAP.get("out_hwdim")[1] * n_is_hfactor * DIM_MAP["dilate_dim"][0]
        )
        undilate_l0c_m = n_is_hfactor * DIM_MAP["img_shape"][3]
        return undilate_l0c_m

    block_m = cce_params.CUBE_MKN[TENSOR_MAP.get("c_l0c").dtype]["mac"][0]
    block_k = cce_params.CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"][1]
    # get factor from l0c, ub to ddr
    mc_from_tiling = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]
    l0c_tiling_factor = [TILING["CL0_matrix"][0], mc_from_tiling]
    undilate_l0c_m = (mc_from_tiling // DIM_MAP["img_shape"][3]) * DIM_MAP["img_shape"][3]

    need_buffer_tile = False
    if DIM_MAP.get("dilate_dim") is not None:
        # get fh*w(fh is factor of H),
        # and update l0c_tiling_factor[1] to dilate*fh*w
        if is_conv1d_bool:
            l0c_tiling_factor[1] = mc_from_tiling * DIM_MAP["dilate_dim"][1]
            undilate_l0c_m = mc_from_tiling
            need_buffer_tile = True
        else:
            if dynamic_para != "dynamic_hw":
                undilate_l0c_m = _get_undilate_loc_m(l0c_tiling_factor)
            else:
                undilate_l0c_m = _get_undilate_loc_m_dynamic(l0c_tiling_factor, sch)
            if undilate_l0c_m % block_m != 0:
                need_buffer_tile = True
    if TENSOR_MAP.get("drelu_ub") is not None:
        if (
            l0c_tiling_factor[1] % block_m != 0
            or (
                DIM_MAP["out_img_shape"][2] -
                DIM_MAP["out_img_shape"][2] // l0c_tiling_factor[1] * l0c_tiling_factor[1]
            ) % block_m != 0
        ):
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

    if dynamic_para == "dynamic_hw":
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
    _print_debug(
        "l0c_ub_tiling_factor:", TILING["CUB_matrix"], "l0c_ub_parts:", l0c_ub_parts
    )

    al1_parts = [1, 1]
    if TILING["AL1_shape"]:  # AL1_shape = [C1,H*W,16,16],batch=1
        # parts of k-axis from DDR to L1---need div by H*W
        al1_parts = [
            int_ceil_div(
                DIM_MAP[GroupDictKeys.dy_c1_extend],
                int_ceil_div(TILING["AL1_shape"][0], block_k)
            ),
            int_ceil_div(l0c_parts[1], TILING["AL1_shape"][1])
        ]

    bl1_parts = [1, 1]
    if TILING["BL1_shape"]:
        if (l0c_parts[0] % TILING["BL1_shape"][1]) != 0:
            _raise_dx_opti_err(
                "second value of BL1_shape should be factor of n block num"
            )
        bl1_parts = [
            int_ceil_div(
                DIM_MAP["B_matrix_dim"][1],
                int_ceil_div(TILING["BL1_shape"][0], block_k)
            ),
            int_ceil_div(l0c_parts[0], TILING["BL1_shape"][1])
        ]

    return (
        l0c_tiling_factor,
        l0c_ub_parts,
        al1_parts,
        bl1_parts,
        undilate_l0c_m,
        need_buffer_tile
    )


def _get_mmad_factor():
    """
    get tilling factor in mmad

    :return:tilling factor for al0
            tilling factor for bl0
            tilling factor for reduce axis
    """
    al0_factor = [TILING.get("AL0_matrix")[0], TILING.get("AL0_matrix")[1]]
    bl0_factor = [TILING.get("BL0_matrix")[0], TILING.get("BL0_matrix")[1]]
    reduce_factor = TILING.get("BL0_matrix")[0]
    return al0_factor, bl0_factor, reduce_factor


def _bind_multi_core(  # pylint: disable=R0913,R0914
    sch,
    c_gm,
    g_dim,
    l1_n_outer_outer,
    l1_n_out_inner,
    l1_m_outer_outer,
    l1_m_outer_inner,
    dynamic_para
):
    if "block_dim" in TILING and not DeconvParam.get_para_map("load3d_flag"):
        block_dim = TILING["block_dim"]
    else:
        block_dim = [1, 1, 1, 1]
    blockidx_list = []
    # split batch axis
    if dynamic_para == "dynamic_batch":
        batch_dim_factor = int_ceil_div(DIM_MAP["out_img_shape"][0], block_dim[0])
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


def _get_l0c_and_l1_axis(  # pylint: disable=R0914,R0913,W0613
    sch, c_gm, l0c_factor, al1_parts, bl1_parts, num_batch, g_extend, dynamic_para
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

    def _get_reorder_flag(al1_parts, bl1_parts, reorder_flag):
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
        _print_debug("reorder_flag:", reorder_flag)
        return reorder_flag

    # split c_gm according to factor of loc and out_shape
    g_dim, c_gm_inner = sch[c_gm].split(c_gm.op.axis[1], nparts=g_extend)
    l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm_inner, l0c_factor[0])
    l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[2], l0c_factor[1])
    sch[c_gm].reorder(g_dim, c_gm.op.axis[0], l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)

    # split c_gm according to factor of a_l1 and b_l1
    l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(
        l0c_m_outer, nparts=al1_parts[1]
    )
    l1_n_outer_outer, l1_n_out_inner = sch[c_gm].split(l0c_n_outer, nparts=bl1_parts[1])
    _print_ir_conv("split gm by loc_factor and l1_parts", sch)
    [batch_in, g_inner, l1_m_outer_inner_in, l1_n_out_inner_out,
     l1_n_out_inner_in, blockidx_list] = _bind_multi_core(
        sch,
        c_gm,
        g_dim,
        l1_n_outer_outer,
        l1_n_out_inner,
        l1_m_outer_outer,
        l1_m_outer_inner,
        dynamic_para
    )
    _print_ir_conv("bind multi core", sch)
    # reorder al1 and bl1 axis according to double buffer
    batch_in_out_axis, batch_in_inner_axis = sch[c_gm].split(batch_in, factor=1)

    # m or n reorder flag, if m_outer is smaller, reorder is true
    reorder_flag = False
    if not dynamic_para:
        reorder_flag = _get_reorder_flag(al1_parts, bl1_parts, reorder_flag)
    _print_ir_conv("before reorder", sch)
    if reorder_flag:
        sch[c_gm].reorder(l1_m_outer_outer, batch_in_inner_axis, l1_n_outer_outer)
        overload_axis = l1_m_outer_outer
        overload_flag_gm = False
    else:
        sch[c_gm].reorder(l1_n_outer_outer, l1_m_outer_outer, batch_in_inner_axis)
        overload_axis = l1_n_outer_outer
        overload_flag_gm = True
    _print_ir_conv("after reorder", sch)

    return (
        batch_in_out_axis,
        l1_n_outer_outer,
        batch_in_inner_axis,
        l1_n_out_inner_out,
        l1_m_outer_inner_in,
        l0c_n_inner,
        l0c_m_inner,
        l1_m_outer_outer,
        l1_n_out_inner_in,
        blockidx_list,
        g_inner,
        overload_axis,
        overload_flag_gm,
        l0c_m_outer
    )


def _get_l0a_and_l0b_axis(  # pylint: disable=R0913,R0914
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
        al0_axis_factor[0] * cce_params.CUBE_MKN[c_l0c.dtype]["mac"][0]
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


def _get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer):
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
        _raise_dx_opti_err("illegal value of AL1_shape & BL1_shape")

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


def _dilate_schedule(sch, dilate_ub, out_w, dilate_w, dilate_h):
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
    sch[dilate_ub].unroll(hi_axis)
    sch[dilate_ub].reorder(
        wi_axis, dilate_ub.op.axis[0], dilate_ub.op.axis[1], ho_axis, wo_axis
    )
    sch[dilate_ub].unroll(wi_axis)
    return wo_axis


def opti_schedule(
    tensor, sch_list, tiling_case=None, var_range=None
):  # pylint: disable=R0915,R0914
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
            block_m = cce_params.CUBE_MKN[TENSOR_MAP.get("c_ub").dtype]["mac"][0]
            mm_coefficient_factor = undilate_l0c_m
            moo_coefficient_unzero = int_ceil_div(
                int_ceil_div(DIM_MAP["out_img_shape"][2], l0c_factor[1]), al1_parts[1]
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
                int_ceil_div(DIM_MAP["dx_6GD_shape"][2], l0c_factor[0]), bl1_parts[1]
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
                else int_ceil_div(DIM_MAP["dx_6GD_shape"][0], TILING["block_dim"][3])
            )
            gi_coefficient = (
                0
                if DIM_MAP["dx_6GD_shape"][0] == 1
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
                (
                go_coefficient * group_axis
                + gi_coefficient * g_inner) * DIM_MAP["dx_6GD_shape"][2]
                + bl1_at_c_axis.var * noo_coefficient
                + noio_coefficient * noio_axis
                + noii_coefficient * noii_axis.var
            ) * no_coefficient + nio_coefficient * l0c_n_inner_outer.var
            return cub_buffertile_n_min

        l0c_factor_tile = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]

        # multi core and one core
        group_axis, batcho_axis, noio_axis, moio_axis = blockidx_list
        # cub buffertile batch axis
        batch_factor = int_ceil_div(DIM_MAP["img_shape"][0], TILING["block_dim"][0])
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
                "dequant_relu",
                "dequant_sqrt",
                "input_ub",
                "deq",
                "reform_op",
                "quant_vmuls",
                "quant_vadds",
                "cast_i8_ub",
                "data_transfer"
            ]
            for tensor in TENSOR_MAP:
                if (
                    tensor == "dequant_remove_pad"
                    and c_gm.op.tag != TENSOR_MAP[tensor].op.tag
                ):
                    sch[TENSOR_MAP[tensor]].compute_inline()
                elif tensor == "elewise_tensor":
                    tensor_list = TENSOR_MAP[tensor]
                    for elewise_tensor in tensor_list:
                        if elewise_tensor.op.tag != c_gm.op.tag:
                            sch[elewise_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
                elif tensor == "input_tensor":
                    tensor_list = TENSOR_MAP[tensor]
                    for input_tensor in tensor_list:
                        sch[input_tensor[0]].compute_at(sch[c_gm], l0c_m_inner_outer)
                        if input_tensor[1].op.tag != c_gm.op.tag:
                            sch[input_tensor[1]].compute_at(
                                sch[c_gm], l0c_m_inner_outer
                            )
                elif tensor == "input_ub" and not quan_para["q_padding"]:
                    sch[TENSOR_MAP[tensor]].compute_inline()
                elif (
                    tensor in ub_attach_list
                    and c_gm.op.tag != TENSOR_MAP[tensor].op.tag
                ):
                    sch[TENSOR_MAP[tensor]].compute_at(sch[c_gm], l0c_m_inner_outer)
                elif tensor == "c_ub_res":
                    sch[TENSOR_MAP[tensor]].compute_at(sch[c_gm], l0c_m_inner_outer)

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
                    sch[bias_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                else:
                    sch[bias_ub].compute_at(sch[c_gm], batch_in_out_axis)

        if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
            if fusion_type == FUSION_DX_ADD_DRELU:
                sch[add_res_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[add_input_ub].compute_at(sch[c_gm], add_input_at)
            sch[drelu_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[bitmask_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[fusion_dx_gm].compute_at(sch[c_gm], l0c_m_inner_outer)
        elif fusion_type == FUSION_DX_ELEWISE:
            for input_tensor in TENSOR_MAP["input_tensor_list"]:
                sch[input_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
            for ub_tensor in TENSOR_MAP["ub_list"]:
                sch[ub_tensor].compute_at(sch[c_gm], l0c_m_inner_outer)
        elif fusion_type in (
            FUSION_DX_DEQUANT,
            FUSION_DX_DEQUANT_QUANT,
            FUSION_DX_REQUANT
        ):
            _attach_ub_quant()

        _attach_ub_bias()

        if dilate_ub is not None:
            filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
            sch[dilate_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[filling_zero_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            if dynamic_para == "dynamic_hw":
                filling_one_ub = TENSOR_MAP["tensor_fillling_one"]
                sch[filling_one_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                tensor_vn = TENSOR_MAP["tensor_vn"]
                sch[tensor_vn].compute_at(sch[c_gm], l0c_m_inner_outer)
        if not DeconvParam.get_para_map("cube_vector_split_flag"):
            sch[c_ub].compute_at(sch[c_gm], l0c_m_inner_outer)

        if "data_transfer" in TENSOR_MAP:
            sch[c_ub].compute_inline()
            sch[TENSOR_MAP["data_transfer"]].buffer_align(
                (1, 1),
                (1, 1),
                (1, cce_params.CUBE_MKN["int8"]["mac"][0]),
                (1, cce_params.CUBE_MKN["int8"]["mac"][0])
            )

    def _attach_al1_bl1():
        # attach tensor of al1 and bl1 to c_l0c
        if TILING["AL1_shape"]:
            _print_debug("al1_parts[0]:", al1_parts[0])
            if al1_parts[0] != 1:
                sch[a_l1].compute_at(sch[c_l0c], al1_at_l0c_axis)
                if DeconvParam.get_para_map("load3d_flag"):
                    sch[a_l0a_before].compute_at(sch[c_l0c], al1_at_l0c_axis)
            else:
                sch[a_l1].compute_at(sch[c_gm], al1_at_c_axis)
                if DeconvParam.get_para_map("load3d_flag"):
                    sch[a_l0a_before].compute_at(sch[c_gm], al1_at_c_axis)
        else:  # TILING["AL1_shape"]=[]
            sch[a_l1].compute_at(sch[c_gm], batch_in_out_axis)
            if DeconvParam.get_para_map("load3d_flag"):
                sch[a_l0a_before].compute_at(sch[c_gm], batch_in_out_axis)

        if TILING["BL1_shape"]:
            _print_debug("bl1_parts[0]:", bl1_parts[0])
            if bl1_parts[0] != 1:
                sch[b_l1].compute_at(sch[c_l0c], bl1_at_l0c_axis)
            else:  # bl1_parts[0] == 1
                sch[b_l1].compute_at(sch[c_gm], bl1_at_c_axis)
        else:  # TILING["BL1_shape"]=[]
            sch[b_l1].compute_at(sch[c_gm], batch_in_out_axis)

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
                filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
                sch[dilate_ub].double_buffer()
                sch[filling_zero_ub].double_buffer()
                if dynamic_para == "dynamic_hw":
                    filling_one_ub = TENSOR_MAP["tensor_fillling_one"]
                    sch[filling_one_ub].double_buffer()
                    tensor_vn = TENSOR_MAP["tensor_vn"]
                    sch[tensor_vn].double_buffer()
            if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
                sch[fusion_dx_gm].double_buffer()
                sch[drelu_ub].double_buffer()
                if fusion_type == FUSION_DX_ADD_DRELU:
                    sch[add_res_ub].double_buffer()
                    if dilate_ub is not None:
                        sch[add_input_ub].double_buffer()
                        sch[add_input_ub].preload()
            elif fusion_type == FUSION_DX_ELEWISE:
                for ub_tensor in TENSOR_MAP["ub_list"]:
                    sch[ub_tensor].double_buffer()

    def _do_reused_by(fusion_type):
        if dilate_ub is not None:
            dx_output_ub = dilate_ub
        else:
            dx_output_ub = c_ub
        if fusion_type == FUSION_DX_ADD_DRELU:
            if dilate_ub is not None:
                filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
                sch[filling_zero_ub].reused_by(add_input_ub)
                sch[dx_output_ub].reused_by(fusion_dx_gm, add_res_ub)
            else:
                sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub, add_res_ub)
        elif fusion_type == FUSION_DX_DRELU:
            sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub)
        elif fusion_type == FUSION_DX_ELEWISE:
            iv_c_gm = calc_info_of_iter_vars(sch[c_gm])
            len_axis = iv_c_gm[-2][1].extent
            for ub_tensor in TENSOR_MAP["ub_list"]:
                len_align = (
                    tvm.min(
                        len_axis, dx_output_ub.shape[2] - l0c_m_outer.var * len_axis
                    )
                    * ub_tensor.op.axis[3].dom.extent
                )

                sch[ub_tensor].bind_buffer(ub_tensor.op.axis[1], len_align, 0)
                sch[dx_output_ub].reused_by(ub_tensor)

    def _fusion_intrin_mapping(fusion_type):  # pylint: disable=R0915
        def _add_res_ub_insn():
            if dilate_ub is None:
                sch[add_res_ub].emit_insn(add_res_ub.op.axis[0], "vector_add")
            else:
                sch[add_res_ub].emit_insn(add_res_ub.op.axis[0], "phony_insn")

        def _quant_vector_insn():
            for tensor_name in TENSOR_MAP:
                if (
                    "dequant" in tensor_name
                    and tensor_name != "dequant_remove_pad"
                    and dx_res.op.tag != TENSOR_MAP[tensor_name].op.tag
                ):
                    sch[TENSOR_MAP[tensor_name]].emit_insn(
                        TENSOR_MAP[tensor_name].op.axis[0], "vector_auto"
                    )
                elif tensor_name == "c_ub_res":
                    sch[TENSOR_MAP[tensor_name]].emit_insn(
                        TENSOR_MAP[tensor_name].op.axis[0], "vector_auto"
                    )
                elif tensor_name == "reform_op":
                    reform_ub = TENSOR_MAP[tensor_name]
                    ndim = len(sch[reform_ub].op.axis)
                    coo, _ = sch[reform_ub].split(
                        sch[reform_ub].op.axis[ndim - 1],
                        cce_params.CUBE_MKN["float16"]["mac"][1]
                    )
                    axis_list = sch[reform_ub].op.axis[0 : ndim - 1]
                    sch[reform_ub].reorder(coo, *axis_list)
                    sch[reform_ub].emit_insn(sch[reform_ub].op.axis[2], "vector_auto")
                elif tensor_name == "elewise_tensor":
                    for elewise_tensor in TENSOR_MAP[tensor_name]:
                        if elewise_tensor.op.tag != dx_res.op.tag:
                            sch[elewise_tensor].emit_insn(
                                elewise_tensor.op.axis[0], "vector_auto"
                            )

        def _quant_copy_insn():
            for tensor_name in TENSOR_MAP:
                if tensor_name == "deq":
                    sch[TENSOR_MAP[tensor_name]].emit_insn(
                        TENSOR_MAP[tensor_name].op.axis[0], "dma_copy"
                    )
                elif tensor_name == "input_tensor":
                    for input_tensor in TENSOR_MAP[tensor_name]:
                        if input_tensor[1].op.tag != dx_res.op.tag:
                            sch[input_tensor[1]].emit_insn(
                                input_tensor[1].op.axis[0], "vector_auto"
                            )
                        sch[input_tensor[0]].emit_insn(
                            input_tensor[0].op.axis[0], "dma_copy"
                        )
                elif tensor_name == "data_transfer":
                    c_ub_reform = TENSOR_MAP[tensor_name]
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
                    axis_index = 2 if quan_para["deq_vector"] else 0
                    if cce_conf.is_v200_version_new():
                        sch[TENSOR_MAP[tensor_name]].emit_insn(
                            TENSOR_MAP[tensor_name].op.axis[axis_index], "dma_copy"
                        )
                    else:
                        emit = "vector" if axis_index == 2 else "scalar"
                        sch[TENSOR_MAP[tensor_name]].pragma(
                            TENSOR_MAP[tensor_name].op.axis[axis_index],
                            "deq_scale",
                            emit
                        )
                elif tensor_name == "input_ub" and quan_para["q_padding"]:
                    sch[TENSOR_MAP[tensor_name]].emit_insn(
                        TENSOR_MAP[tensor_name].op.axis[0], "dma_padding"
                    )
                elif tensor_name == "cast_i8_ub":
                    cast_i8_ub = TENSOR_MAP[tensor_name]
                    round_mode_emit_insn = (
                        "vector_conv_%s" % quan_para["q_round"].lower()
                    )
                    if cce_conf.CceProductParams().is_mini_version():
                        round_mode_emit_insn = "vector_conv"
                    sch[cast_i8_ub].emit_insn(
                        cast_i8_ub.op.axis[0], round_mode_emit_insn
                    )

        if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
            sch[bitmask_ub].emit_insn(bitmask_ub.op.axis[0], "dma_copy")
            sch[drelu_ub].emit_insn(drelu_ub.op.axis[0], "vector_selects_bool")
            sch[fusion_dx_gm].emit_insn(fusion_dx_gm.op.axis[0], "phony_insn")
            if fusion_type == FUSION_DX_ADD_DRELU:
                sch[add_input_ub].emit_insn(add_input_ub.op.axis[0], "dma_copy")
                _add_res_ub_insn()
        elif fusion_type == FUSION_DX_ELEWISE:
            for input_tensor in TENSOR_MAP["input_tensor_list"]:
                sch[input_tensor].emit_insn(input_tensor.op.axis[0], "dma_copy")
            for ub_tensor in TENSOR_MAP["ub_list"]:
                sch[ub_tensor].emit_insn(ub_tensor.op.axis[0], "vector_auto")
        elif fusion_type in (
            FUSION_DX_DEQUANT,
            FUSION_DX_DEQUANT_QUANT,
            FUSION_DX_REQUANT
        ):
            _quant_ub_insn()

    def _intrin_mapping(fusion_type):  # pylint: disable=R0915,R0912
        def l1fusion_intrin():
            valid_shape = DeconvParam.get_para_map("valid_shape")
            slice_offset = DeconvParam.get_para_map("slice_offset")
            if TILING["AL1_shape"] is not None:
                if DeconvParam.get_para_map("input_memory_type")[0] == 1:
                    sch[a_l1].emit_insn(a_l1.op.axis[0], "phony_insn")
                else:
                    sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy")
                    if DeconvParam.get_para_map("l1_fusion_type") != -1:
                        sch[a_l1].pragma(a_l1.op.axis[0], "jump_data", 1)
            if DeconvParam.get_para_map("load3d_flag"):
                conv_fm_h = valid_shape[2] if valid_shape else a_l1.shape[2]
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
                    "conv_fm_h": conv_fm_h,
                    "conv_fm_w": a_l1.shape[3]
                }
                if (
                    valid_shape
                    and DeconvParam.get_para_map("input_memory_type")[0] == 1
                ):
                    setfmatrix_dict["conv_fm_offset_h"] = slice_offset[2]
                sch[a_l0a_before].emit_insn(
                    a_l0a_before.op.axis[0], "set_fmatrix", setfmatrix_dict
                )
                sch[a_l0a].emit_insn(a_l0a.op.axis[1], "im2col")
            else:
                sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy")
            if DeconvParam.get_para_map("write_select"):
                sch[c_gm.op.input_tensors[0]].compute_inline()
                align_length = c_gm.op.attrs["HWC0"]
                sch[c_gm].bind_buffer(c_gm.op.axis[1], align_length, 0)

        l1fusion_intrin()
        sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy")
        sch[b_l0b].emit_insn(b_l0b.op.axis[0], "dma_copy")

        if fusion_type not in (
            FUSION_DX_DEQUANT,
            FUSION_DX_DEQUANT_QUANT,
            FUSION_DX_REQUANT
        ):
            if c_ub is not None:
                sch[c_ub].emit_insn(c_ub.op.axis[0], "dma_copy")

        sch[c_gm].emit_insn(l0c_n_inner_inner, "dma_copy")

        if dilate_ub is not None:
            filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]

            if bias_add_vector_ub is not None:
                sch[dilate_ub].reused_by(filling_zero_ub, bias_add_vector_ub)
            else:
                if dynamic_para == "dynamic_hw":
                    filling_one_ub = TENSOR_MAP["tensor_fillling_one"]
                    sch[filling_zero_ub].reused_by(filling_one_ub)
                    sch[filling_one_ub].emit_insn(
                        sch[filling_one_ub].op.axis[0], "vector_dup"
                    )
                    tensor_vn = TENSOR_MAP["tensor_vn"]
                    sch[filling_zero_ub].reused_by(tensor_vn)
                    sch[tensor_vn].emit_insn(sch[tensor_vn].op.axis[0], "phony_insn")
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
            if dynamic_para == "dynamic_hw":
                vmul_at_axis = _dilate_schedule(
                    sch,
                    dilate_ub,
                    DIM_MAP.get("out_hwdim")[1],
                    DIM_MAP.get("dilate_dim")[1],
                    DIM_MAP.get("dilate_dim")[0]
                )
                sch[dilate_ub].emit_insn(vmul_at_axis, "vector_mul")
            else:
                vadd_at_axis = _dilate_schedule(
                    sch,
                    dilate_ub,
                    DIM_MAP.get("out_hwdim")[1],
                    DIM_MAP.get("dilate_dim")[1],
                    DIM_MAP.get("dilate_dim")[0]
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

        sch[c_l0c].emit_insn(batch_l0c_inner, "mad", mad_dict)
        _print_ir_conv("intrin mapping", sch)

    def _get_al1_bound():
        def int_ceil_div_tvm(num_a, num_b):
            return tvm.floordiv((num_a + num_b - 1), num_b)

        dedy_shape_nc1hwc0 = DIM_MAP["img_shape"]
        tiling_m0 = TILING["CL0_matrix"][2]
        if len(TILING["AL1_shape"]) != 0:
            k_al1, multi_m_al1 = TILING["AL1_shape"][:2]
            al1_m = multi_m_al1 * TILING["CL0_matrix"][1] * tiling_m0
            al1_bound = al1_m * k_al1
        else:
            al1_bound = (
                dedy_shape_nc1hwc0[1]
                * int_ceil_div_tvm(
                    dedy_shape_nc1hwc0[2] * dedy_shape_nc1hwc0[3], tiling_m0
                )
                * tiling_m0
                * dedy_shape_nc1hwc0[4]
            )
        return al1_bound

    def _get_dilate_ub_bound():
        nc_factor, mc_factor, tiling_m0, tiling_n0 = TILING["CUB_matrix"][:4]
        cub_bound = nc_factor * mc_factor * tiling_m0 * tiling_n0
        sch[c_ub].set_storage_bound(cub_bound)
        l0c_bound = (
            TILING["CL0_matrix"][0] * TILING["CL0_matrix"][1] * tiling_m0 * tiling_n0
        )
        sch[c_l0c].set_storage_bound(l0c_bound)
        dilate_bound = l0c_factor[1] * nc_factor * tiling_n0
        sch[dilate_ub].set_storage_bound(dilate_bound)
        sch[dilate_ub].mem_unique()

    def _is_conv1d():
        return (
            DIM_MAP["dx_5D_shape"][2] == 1
            and DIM_MAP["img_shape"][2] == 1
            and (TENSOR_MAP.get("dilate_ub") is None or DIM_MAP["dilate_dim"][0] == 1)
        )

    def _res_select_write(res):
        # selet write
        write_select_flag = bool(res.op.tag == "write_select")
        DeconvParam.update_para_map("write_select", write_select_flag)
        if write_select_flag:
            res_before_write_select = res.op.input_tensors[0]
        else:
            res_before_write_select = res
        return res_before_write_select

    def _get_fusion_type(fusion_type):
        # get the fusion num
        fusion_type_num = FUSION_TYPE_2_NUM[fusion_type]
        if isinstance(fusion_type_num, tuple):
            if dx_res.dtype == "float16":
                fusion_type_num = 1
            else:
                fusion_type_num = 2
        if DeconvParam.get_para_map("write_select"):
            fusion_type_num += 20
        DeconvParam.update_para_map("fusion_type_num", fusion_type_num)

    def _config_dynamic_para(var_range):
        if var_range is None:
            return None
        if "batch_n" in var_range and len(var_range) == 1:
            return "dynamic_batch"
        for var in ("dedy_h", "dedy_w", "dx_h", "dx_w"):
            if var in var_range:
                return "dynamic_hw"
        return None

    def _handle_dynamic_shape():
        # set storage bound
        if dynamic_para is not None:
            al1_bound = _get_al1_bound()
            sch[a_l1].set_storage_bound(al1_bound)
        if dynamic_para == "dynamic_hw" and dilate_ub is not None:
            _get_dilate_ub_bound()
        # disable_allocate
        sch.disable_allocate(cce_params.scope_cbuf)
        sch.disable_allocate(cce_params.scope_ca)
        sch.disable_allocate(cce_params.scope_cb)
        sch.disable_allocate(cce_params.scope_cc)
        sch.disable_allocate(cce_params.scope_ubuf)
        # mem_unique
        sch[a_l1].mem_unique()
        sch[a_l0a].mem_unique()
        sch[b_l1].mem_unique()
        sch[b_l0b].mem_unique()
        sch[c_l0c].mem_unique()
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

    def _buffer_tile_loc_c1():
        no_coefficient = (l0c_factor[0] * 2 if c_gm.dtype == "int8" else l0c_factor[0])
        noo_coefficient_unzero = int_ceil_div(
                int_ceil_div(DIM_MAP["dx_6GD_shape"][2], l0c_factor[0]), bl1_parts[1]
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
            (None, None),
            (None, None),
            (cub_buffertile_n_min, no_coefficient),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
        )
        if c_add_bias is not None:
            sch[c_add_bias].buffer_tile(
            (None, None),
            (None, None),
            (cub_buffertile_n_min, no_coefficient),
            (None, None),
            (None, None),
            )

    TILING.clear()
    dx_res = tensor
    sch = sch_list[0]
    _print_ir_conv("schedule", sch)

    dx_res_write = _res_select_write(dx_res)
    quan_para = _check_quant_fusion(dx_res_write)
    dynamic_para = _config_dynamic_para(var_range)

    # set scope for all tensor
    tensor_dx_gm = _set_data_layout(dx_res_write, dx_res, sch, dynamic_para, var_range)
    kernel_name = tensor_dx_gm.op.attrs["kernel_name"]
    fusion_type = DeconvParam.get_para_map("FUSION_TYPE")
    _get_fusion_type(fusion_type)
    is_conv1d_bool = _is_conv1d()
    _print_debug("IS_CONV1D:", is_conv1d_bool)

    _print_ir_conv("set scope", sch)

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
    if c_gm.dtype == "int8":
        # In quant or requant scenes, co of ddr is 32, c1_ddr is c1_loc//2
        DIM_MAP["dx_6GD_shape"][2] = (DIM_MAP["dx_6GD_shape"][2] + 1) // 2
        DIM_MAP["dx_6GD_shape"][5] = DIM_MAP["dx_6GD_shape"][5] * 2

    if DeconvParam.get_para_map("load3d_flag"):
        a_l0a_before = TENSOR_MAP.get("a_l0a_before")
    drelu_ub, bitmask_ub, add_res_ub, add_input_ub, fusion_dx_gm = (
        TENSOR_MAP.get("drelu_ub"),
        TENSOR_MAP.get("bitmask_ub"),
        TENSOR_MAP.get("add_res_ub"),
        TENSOR_MAP.get("add_input_ub"),
        TENSOR_MAP.get("fusion_dx_gm")
    )
    bias_add_vector_ub, bias_ub = TENSOR_MAP.get("bias_add_vector"), TENSOR_MAP.get(
        "bias_ub"
    )
    bias_ub_brc, bias_l0c, c_add_bias = (
        TENSOR_MAP.get("bias_ub_brc"),
        TENSOR_MAP.get("bias_l0c"),
        TENSOR_MAP.get("c_add_bias")
    )

    _get_tiling(
        tensor, fusion_type, kernel_name, is_conv1d_bool, dynamic_para, tiling_case
    )

    # get factor and parts from tiling
    (
        l0c_factor,
        l0c_ub_parts,
        al1_parts,
        bl1_parts,
        undilate_l0c_m,
        need_buffer_tile
    ) = _get_aicore_tiling_factor(is_conv1d_bool, dynamic_para, sch)
    al0_axis_factor, bl0_axis_factor, reduce_axis_factor = _get_mmad_factor()
    num_batch = DIM_MAP["img_shape"][0]
    _print_ir_conv("before split", sch)
    g_extend = DIM_MAP[GroupDictKeys.g_extend]
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
        l0c_m_outer
    ) = _get_l0c_and_l1_axis(
        sch, c_gm, l0c_factor, al1_parts, bl1_parts, num_batch, g_extend, dynamic_para
    )
    al1_at_c_axis = batch_in_inner_axis
    bl1_at_c_axis = l1_n_outer_outer
    c_slice_axis = l1_m_outer_inner_in
    _print_ir_conv("split with al1 and bl1 factor", sch)

    # attach tensor of CUB
    l0c_n_inner_outer, l0c_n_inner_inner = sch[c_gm].split(
        l0c_n_inner, nparts=l0c_ub_parts[0]
    )
    l0c_m_inner_outer, l0c_m_inner_inner = sch[c_gm].split(l0c_m_inner, nparts=1)
    add_input_at, l0c_m_inner_outer = sch[c_gm].split(l0c_m_inner_outer, nparts=1)
    sch[c_gm].reorder(
        l0c_n_inner_outer,
        add_input_at,
        l0c_m_inner_outer,
        l0c_n_inner_inner,
        l0c_m_inner_inner
    )
    _print_ir_conv("reorder loc", sch)

    _attach_ub(fusion_type)
    _print_ir_conv("attach CUB", sch)

    # attach tensor of l0c
    new_c_col_axis = [
        sch[c_l0c].op.axis[1],
        sch[c_l0c].op.axis[2],
        sch[c_l0c].op.axis[3],
        sch[c_l0c].op.axis[4]
    ]
    sch[c_l0c].compute_at(sch[c_gm], c_slice_axis)
    if bias_l0c is not None:
        sch[bias_l0c].compute_at(sch[c_gm], c_slice_axis)
        sch[c_add_bias].compute_at(sch[c_gm], c_slice_axis)
        sch[bias_ub_brc].compute_at(sch[c_gm], c_slice_axis)

    _print_ir_conv("attach l0c", sch)

    # split and get axis of reduce, al0_at_axis, bl0_at_axis
    al0_m_out, bl0_n_outer, k_outer_outer, batch_l0c_inner = _get_l0a_and_l0b_axis(
        sch, c_l0c, new_c_col_axis, al0_axis_factor, bl0_axis_factor, reduce_axis_factor
    )
    _print_ir_conv("split with al0/bl0/reduce factor", sch)

    # attach tensor of a_l0a
    sch[a_l0a].compute_at(sch[c_l0c], al0_m_out)
    sch[b_l0b].compute_at(sch[c_l0c], bl0_n_outer)
    _print_ir_conv("attach l0a/l0b", sch)

    # split and get axis of al1_at_l0c_axis, bl1_at_l0c_axis
    (
        al1_at_l0c_axis,
        bl1_at_l0c_axis,
        reduce_axis_serial,
        overload_flag_l0c
    ) = _get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer)

    _attach_al1_bl1()
    _print_ir_conv("attach al1/bl1", sch)

    # do buffer_tile or buffer_align for cub
    if need_buffer_tile:
        _do_buffer_tile()
        _print_ir_conv("after_tile", sch)
    else:
        if c_ub is not None:
            sch[c_ub].buffer_align(
                (1, 1),
                (1, 1),
                (1, cce_params.CUBE_MKN["float16"]["mac"][0]),
                (1, cce_params.CUBE_MKN["float16"]["mac"][0])
            )
        if bias_add_vector_ub is not None and dilate_ub is None:
            sch[bias_add_vector_ub].buffer_align(
                (1, 1),
                (1, 1),
                (1, cce_params.CUBE_MKN["float16"]["mac"][0]),
                (1, cce_params.CUBE_MKN["float16"]["mac"][0])
            )
    _buffer_tile_loc_c1()
    # double buffer
    _do_double_buffer(fusion_type)
    _print_ir_conv("enable double buffer", sch)

    _do_reused_by(fusion_type)
    _print_ir_conv("reused_by", sch)

    # preload
    if DeconvParam.get_para_map("DATA_AMOUNT_CUB") * (
        1 + 2 * FUSION_TYPE_2_OPERAND_NUM.get(fusion_type)
    ) <= cce_conf.get_soc_spec("UB_SIZE"):
        _print_debug("dx opti ub preload enable.")
        if fusion_type == FUSION_DX_DRELU:
            sch[bitmask_ub].double_buffer()
            sch[bitmask_ub].preload()
        elif fusion_type == FUSION_DX_ADD_DRELU:
            sch[bitmask_ub].double_buffer()
            sch[bitmask_ub].preload()
            if dilate_ub is None:
                sch[add_input_ub].double_buffer()
                sch[add_input_ub].preload()

        _print_ir_conv("preload", sch)
    # intrin mapping
    _intrin_mapping(fusion_type)

    overload_flag = _check_overload_dy(overload_flag_gm, overload_flag_l0c)
    _set_overload_flag(sch[c_gm], overload_flag, overload_axis)

    def _handle_workspace():
        l1_tensor_map = {}
        if not DeconvParam.get_para_map("fmap_l1_addr_flag"):
            l1_tensor_map = None
        else:
            fmap = DeConvKernelSize1Pattern.dedy
            if (
                DeconvParam.get_para_map("l1_fusion_type") != -1
                and DeconvParam.get_para_map("input_memory_type")[0] == 0
            ):
                sch[a_l1].set_storage_bound(
                    DeconvParam.get_para_map("fmap_l1_valid_size")
                )
                l1_tensor_map[fmap] = a_l1
            else:
                l1_tensor_map = None
        L1CommonParam.l1_fusion_tensors_map = l1_tensor_map

    _handle_workspace()

    # clear global cache
    if dynamic_para:
        _handle_dynamic_shape()

    TILING.clear()
    DIM_MAP.clear()
    TENSOR_MAP.clear()
    return sch
