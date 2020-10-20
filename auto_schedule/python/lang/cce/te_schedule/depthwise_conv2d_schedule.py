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
Schedule of depthwise conv2d.
"""
# pylint: disable=too-many-lines
from te.lang.cce.te_compute.depthwise_conv2d_compute import DepthwiseConv2dParam
from te.lang.cce.te_schedule.util import L1CommonParam
from te.platform import cce_conf
from te.platform import cce_params
from te.tvm import api as tvm
from te.tvm.schedule import create_schedule
from te.domain.tiling.tiling_query import tiling_query
from te.domain.tiling.get_tiling import get_tiling

TILING_HO_TIMES = 16

BLOCK_SIZE = cce_params.BLOCK_REDUCE

# fp16 dtype size 2
FP16_SIZE = 2
# l1 and l0, ping pong read/write mechanism
DOUBLE_BUFFER = 2
# l1 memory size 1M byte
L1_MEM_LIMIT = cce_conf.get_soc_spec(cce_conf.L1_SIZE)
# tilling batch in l1
RESHAPE_BATCH_SIZE_IN_L1 = 1
# l1 include 1 batch
BATCH_SIZE_IN_L1 = 1
# Split ho and wo for mad_cc as 1.
SPLIT_SLOT_TO_BE_FILL_BY_COMPUTE_AT = 1
# L1 size cell
CUBE_M_SIZE_CELL = \
    cce_conf.get_soc_spec(cce_conf.L0B_SIZE) \
    // DOUBLE_BUFFER // FP16_SIZE // BLOCK_SIZE

# fmap h/w 14, N,C fusion tiling
SMALL_FEATURE_MAP_SIZE = 14
# N,C fusion tiling, N max 32
BATCH_TILING_FACTOR = 32

# tiling check
TILING_AL1_SHAPWE_DIM = 4
TILING_BL1_SHAPWE_DIM = 4
TILING_AL0_MATRIX_DIM = 6
TILING_BL0_MATRIX_DIM = 6
TILING_CL0_MATRIX_DIM = 6
TILING_CUB_MATRIX_DIM = 6
TILING_BLOCK_DIM_DIM = 4
TILING_FLOAT16_M = 16
TILING_FLOAT16_K = 16
TILING_FLOAT16_N = 16
TILING_INT8_M = 16
TILING_INT8_K = 32
TILING_INT8_N = 16

C0_32 = TILING_INT8_K
C0_16 = TILING_FLOAT16_K


def _ceil(x):
    """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
    """
    return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

def set_pragma_for_cache_read_mode(is_overload, stage, first_axis):
    """
    set pragma on the first axis for cache read mode

    Parameters
    ----------
    is_overload: True means overload

    stage: a Stage represents schedule for one operation

    first_axis: axis to set flag

    Returns
    -------
    """
    cache_read_mode = 0 if is_overload else 1
    stage.pragma(first_axis, "json_info_cache_read_mode", cache_read_mode)


def _common_tiling_check(tiling):
    def _check_shape_valid(keyname):
        return (tiling[keyname] != []) and (tiling[keyname] is not None)

    def _check_shape(keyname, length, force_check=False):
        if _check_shape_valid(keyname) or force_check:
            if len(tiling[keyname]) != length:
                raise RuntimeError("wrong tiling: %s dim must be %d" %
                                   (keyname, length))

    _check_shape("AL1_shape", TILING_AL1_SHAPWE_DIM)
    _check_shape("BL1_shape", TILING_BL1_SHAPWE_DIM)
    _check_shape("AL0_matrix", TILING_AL0_MATRIX_DIM)
    _check_shape("BL0_matrix", TILING_BL0_MATRIX_DIM)
    _check_shape("CL0_matrix", TILING_CL0_MATRIX_DIM, True)
    _check_shape("CUB_matrix", TILING_CUB_MATRIX_DIM, True)
    _check_shape("block_dim", TILING_BLOCK_DIM_DIM, True)

    def _check_type_bool(keyname):
        if not isinstance(tiling[keyname], bool):
            RuntimeError("%s must be a bool" % keyname)

    _check_type_bool("n_bef_batch_flag")
    _check_type_bool("n_bef_group_flag")
    _check_type_bool("A_overhead_opt_flag")
    _check_type_bool("B_overhead_opt_flag")

    def _check_pingpong_key(keyname):
        if keyname not in tiling["manual_pingpong_buffer"].keys():
            raise RuntimeError("manual_pingpong_buffer must have key <%s>" %
                               (keyname))

    if not isinstance(tiling["manual_pingpong_buffer"], dict):
        RuntimeError("tiling manual_pingpong_buffer must be a dict")
    _check_pingpong_key("AL1_pbuffer")
    _check_pingpong_key("BL1_pbuffer")
    _check_pingpong_key("AL0_pbuffer")
    _check_pingpong_key("BL0_pbuffer")
    _check_pingpong_key("CL0_pbuffer")
    _check_pingpong_key("CUB_pbuffer")

    if _check_shape_valid("AL0_matrix"):
        if tiling["AL0_matrix"][0] != tiling["CL0_matrix"][1]:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][0]
                                 must equal to tiling['CL0_matrix'][1]""")
        if tiling["AL0_matrix"][2] != TILING_FLOAT16_M:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][2]
                 must be equal to %d when w_dtype is float16""" %
                               TILING_FLOAT16_M)
        if tiling["AL0_matrix"][3] != TILING_FLOAT16_K:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][3]
                 must be equal to %d when w_dtype is float16""" %
                               TILING_FLOAT16_K)
    if _check_shape_valid("BL0_matrix"):
        if tiling["BL0_matrix"][1] != tiling["CL0_matrix"][0]:
            raise RuntimeError("""wrong tiling: tiling['BL0_matrix'][1]
                                must equal to tiling['CL0_matrix'][0]""")
        if tiling["BL0_matrix"] != []:
            if tiling["BL0_matrix"][2] != TILING_FLOAT16_N:
                raise RuntimeError("""wrong tiling: tiling['BL0_matrix'][2]
                 must be equal to %d when w_dtype is float16""" %
                                   TILING_FLOAT16_N)
            if tiling["BL0_matrix"][3] != TILING_FLOAT16_K:
                raise RuntimeError("""wrong tiling: tiling['BL0_matrix'][3]
                 must be equal to %d when w_dtype is float16""" %
                                   TILING_FLOAT16_K)
    if _check_shape_valid("AL0_matrix") and _check_shape_valid("BL0_matrix"):
        if tiling["AL0_matrix"][1] != tiling["BL0_matrix"][0]:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][1]
                                must equal to tiling['BL0_matrix'][0]""")
    if tiling["CL0_matrix"][2] != TILING_FLOAT16_M:
        raise RuntimeError("""wrong tiling: tiling['CL0_matrix'][2]
             must be equal to %d when w_dtype is float16""" % TILING_FLOAT16_M)
    if tiling["CL0_matrix"][3] != TILING_FLOAT16_N:
        raise RuntimeError("""wrong tiling: tiling['CL0_matrix'][3]
             must be equal to %d when w_dtype is float16""" % TILING_FLOAT16_N)


def check_type_bool(input_type, string):
    if not isinstance(input_type, bool):
        RuntimeError(string)


def check_tiling_raise_error(tiling, dtype):
    if dtype == "int8":
        tilling_fractor_k = TILING_INT8_K
    elif dtype == "float16":
        tilling_fractor_k = TILING_FLOAT16_K
    if tiling["AL1_shape"] != []:
        if len(tiling["AL1_shape"]) != TILING_AL1_SHAPWE_DIM:
            raise RuntimeError("wrong tiling: AL1_shape dim must be %d" %
                               TILING_AL1_SHAPWE_DIM)
    chek_Bl1_shape = (tiling["BL1_shape"] != []) and \
                     (tiling["BL1_shape"] is not None)
    if chek_Bl1_shape:
        if len(tiling["BL1_shape"]) != TILING_BL1_SHAPWE_DIM:
            raise RuntimeError("wrong tiling: BL1_shape dim must be %d" %
                               TILING_BL1_SHAPWE_DIM)

    if len(tiling["AL0_matrix"]) != TILING_AL0_MATRIX_DIM:
        raise RuntimeError("wrong tiling: AL0_matrix dim must be %d" %
                           TILING_AL0_MATRIX_DIM)

    if tiling["BL0_matrix"] != []:
        if len(tiling["BL0_matrix"]) != TILING_BL0_MATRIX_DIM:
            raise RuntimeError("wrong tiling: BL0_matrix dim must be %d" %
                               TILING_BL0_MATRIX_DIM)

    if len(tiling["CL0_matrix"]) != TILING_CL0_MATRIX_DIM:
        raise RuntimeError("wrong tiling: CL0_matrix dim must be %d" %
                           TILING_CL0_MATRIX_DIM)

    if len(tiling["CUB_matrix"]) != TILING_CUB_MATRIX_DIM:
        raise RuntimeError("wrong tiling: CUB_matrix dim must be %d" %
                           TILING_CUB_MATRIX_DIM)

    if len(tiling["block_dim"]) != TILING_BLOCK_DIM_DIM:
        raise RuntimeError("wrong tiling: block_dim dim must be %d" %
                           TILING_BLOCK_DIM_DIM)

    check_type_bool(
        tiling["n_bef_batch_flag"], """tiling n_bef_batch_flag
                                                         must be a bool""")
    check_type_bool(
        tiling["n_bef_group_flag"], """tiling n_bef_group_flag
                                                         must be a bool""")
    check_type_bool(
        tiling["batch_bef_group_flag"], """tiling
                                        batch_bef_group_flag must be a bool""")
    check_type_bool(
        tiling["A_overhead_opt_flag"], """tiling
                                         A_overhead_opt_flag must be a bool""")
    check_type_bool(
        tiling["B_overhead_opt_flag"], """tiling
                                         B_overhead_opt_flag must be a bool""")

    if not isinstance(tiling["manual_pingpong_buffer"], dict):
        RuntimeError("tiling manual_pingpong_buffer must be a dict")

    if tiling["AL0_matrix"][0] != tiling["CL0_matrix"][1]:
        raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][0]
                             must equal to tiling['CL0_matrix'][1]""")

    if tiling["BL0_matrix"] != []:
        if tiling["AL0_matrix"][1] != tiling["BL0_matrix"][0]:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][1]
                                must equal to tiling['BL0_matrix'][0]""")
        if tiling["BL0_matrix"][1] != tiling["CL0_matrix"][0]:
            raise RuntimeError("""wrong tiling: tiling['BL0_matrix'][1]
                                must equal to tiling['CL0_matrix'][0]""")
        if tiling["AL0_matrix"][2] != TILING_FLOAT16_M:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][2]
            must be equal to %d when w_dtype is float16""" % TILING_FLOAT16_M)
        if tiling["AL0_matrix"][3] != tilling_fractor_k:
            raise RuntimeError("""wrong tiling: tiling['AL0_matrix'][3]
            no equal to %d """ % tilling_fractor_k)
        if tiling["BL0_matrix"] != []:
            if tiling["BL0_matrix"][2] != TILING_FLOAT16_N:
                raise RuntimeError("""wrong tiling: tiling['BL0_matrix'][2]
            must be equal to %d when w_dtype is float16""" % TILING_FLOAT16_N)
            if tiling["BL0_matrix"][3] != tilling_fractor_k:
                raise RuntimeError("""wrong tiling: tiling['BL0_matrix'][3]
            no equal to %d """ % tilling_fractor_k)
        if tiling["CL0_matrix"][2] != TILING_FLOAT16_M:
            raise RuntimeError("""wrong tiling: tiling['CL0_matrix'][2]
            must be equal to %d when w_dtype is float16""" % TILING_FLOAT16_M)
        if tiling["CL0_matrix"][3] != TILING_FLOAT16_N:
            raise RuntimeError("""wrong tiling: tiling['CL0_matrix'][3]
            must be equal to %d when w_dtype is float16""" % TILING_FLOAT16_N)


def check_tiling(tiling, dtype):
    if tiling["AL0_matrix"][2] == 32:
        return False
    check_tiling_raise_error(tiling, dtype)
    return True


def tiling_new_check_empty(input_module, string, place):
    if input_module[string] != []:
        return input_module[string][0:place]
    return []


def tiling_new_check_none(input_module, string, place):
    if input_module[string] is not None:
        return input_module[string][0:place]
    return input_module[string]


def tiling_new_check_none_empty(input_module, string, place):
    if input_module[string] is not None and input_module[string] != []:
        return input_module[string][0:place]
    return input_module[string]


def get_tiling_dict_first(tiling_new):
    tiling = {}
    tiling["AL0_matrix"] = tiling_new["AL0_matrix"][0:6]
    tiling["CL0_matrix"] = tiling_new["CL0_matrix"][0:6]
    tiling["CUB_matrix"] = tiling_new["CUB_matrix"][0:6]
    tiling["BL0_matrix"] = tiling_new_check_empty(tiling_new, "BL0_matrix", 6)
    tiling["manual_pingpong_buffer"] = tiling_new["manual_pingpong_buffer"]
    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]
    tiling["AUB_shape"] = tiling_new_check_none(tiling_new, "AUB_shape", 4)
    tiling["AL1_shape"] = tiling_new_check_empty(tiling_new, "AL1_shape", 4)
    tiling["BL1_shape"] = tiling_new_check_none_empty(tiling_new, "BL1_shape",
                                                      5)
    tiling["block_dim"] = tiling_new["block_dim"][0:4]

    tiling["scale_drq_split_flag"] = False
    tiling["bias_split_flag"] = False
    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]
    tiling["n_bef_group_flag"] = tiling_new["n_bef_group_flag"]
    tiling["batch_bef_group_flag"] = tiling_new["batch_bef_group_flag"]
    tiling["A_overhead_opt_flag"] = tiling_new["A_overhead_opt_flag"]
    tiling["B_overhead_opt_flag"] = tiling_new["B_overhead_opt_flag"]

    return tiling


def get_tiling_dict_second(tiling, shape_input, padding, stride, dtype):
    if not check_tiling(tiling, dtype):
        tiling = {}
        mBitLength = {
            "float32": 32,
            "float16": 16,
            "uint8": 8,
            "int8": 8,
            "uint4": 4,
            "int4": 4
        }
        mBitRatio = {
            "int32": 4,
            "float32": 4,
            "float16": 2,
            "uint8": 1,
            "int8": 1,
            "uint4": 1.0 / 2,
            "int4": 1.0 / 2
        }
        wo = (shape_input[0][3] +
              (2 * padding[0]) - shape_input[1].shape[2]) // stride[1] + 1
        gen_m_target = 0
        for m_target in range(32, 0, -1):
            tmp1 = ((m_target * mBitLength['float16']) + wo - 1) // wo
            tmp2 = ((tmp1 * padding[1]) +
                    shape_input[1].shape[1]) * shape_input[0][3]
            MaxFeatureMap = tmp2 * \
                            2 * mBitRatio[dtype]
            if int(MaxFeatureMap) < L1_MEM_LIMIT:
                gen_m_target = m_target
                break

        m = gen_m_target
        tiling["AL1_shape"] = [1, 1, 1, 1]
        tiling["BL1_shape"] = None
        tiling["AL0_matrix"] = [m, 1, 16, 16, 1, 1]
        tiling["BL0_matrix"] = [1, 2, 16, 16, 1, 1]
        tiling["CL0_matrix"] = [2, m, 16, 16, 1, 1]
        tiling["CUB_matrix"] = [2, m, 16, 16, 1, 1]
        tiling["manual_pingpong_buffer"] = {
            'AL1_pbuffer': 1,
            'BL1_pbuffer': 1,
            'AL0_pbuffer': 1,
            'BL0_pbuffer': 1,
            'CL0_pbuffer': 1,
            'CUB_pbuffer': 1,
            'UBG_pbuffer': 1,
        }

        tiling["scale_drq_split_flag"] = True
        tiling["bias_split_flag"] = True
        tiling["block_dim"] = [1, 1, 1, 1]

    return tiling


def _get_fused_double_operand_num(tensor_dict):
    fused_double_operand_num = tensor_dict["fused_double_operand_num"]
    not_fused_flag = False
    if tensor_dict["flag_is_dequant2"] or (
            tensor_dict["flag_is_dequant_quant"]
            and tensor_dict["flag_is_dequant_sqrt"]):
        fused_double_operand_num = 3
    elif tensor_dict["flag_is_dequant"] or (
            tensor_dict["flag_is_dequant_quant"]
            and not tensor_dict["flag_is_dequant_sqrt"]):
        fused_double_operand_num = 2
    elif tensor_dict["flag_is_requant"]:
        fused_double_operand_num = 2
    elif tensor_dict["flag_is_dequant_sigmoid_mul"] or tensor_dict[
        "flag_is_dequant2_sigmoid_mul"]:
        fused_double_operand_num = 4 if tensor_dict[
            "flag_is_dequant2_sigmoid_mul"] else 3
    else:
        not_fused_flag = True

    return fused_double_operand_num, not_fused_flag


def _tiling_fetch_all(fmap_shape,
                      shape_w,
                      group_num,
                      padding,
                      stride,
                      mad_dtype,
                      fused_num,
                      input_memory_type,
                      output_memory_type,
                      l1_fusion_type,
                      fusion_type_new,
                      fm_l1_valid_size,
                      bias=False,
                      c_dtype="int8",
                      not_fused_flag=False,
                      kernel_name="depthwise_fused_tiling_fetch"):
    if len(fmap_shape) == 6:
        fmap_shape_NC1HWCO = fmap_shape[0], fmap_shape[1], \
                             fmap_shape[3], fmap_shape[4], \
                             fmap_shape[5]
    else:
        fmap_shape_NC1HWCO = fmap_shape[0], fmap_shape[1], \
                             fmap_shape[2], fmap_shape[3], \
                             fmap_shape[4]

    shape_w_NC1HWCO = shape_w.shape[3], shape_w.shape[0], shape_w.shape[1], \
                      shape_w.shape[2], shape_w.shape[4]
    if mad_dtype == "int32":
        dtype = "int8"
        res_dtype = "int32"
    else:
        dtype = "float16"
        res_dtype = "float16"
        fmap_shape_NC1HWCO = list(map(int, fmap_shape_NC1HWCO))
    if not_fused_flag:
        c_dtype = res_dtype

    fmap_shape_NC1HWCO = list(map(int, fmap_shape_NC1HWCO))
    shape_w_NC1HWCO = list(map(int, shape_w_NC1HWCO))
    if input_memory_type == 0 and output_memory_type == 0:
        temp_k = 0
    elif input_memory_type == 0 and output_memory_type == 1:
        temp_k = 1
    elif input_memory_type == 1 and output_memory_type == 0:
        temp_k = 2
    elif input_memory_type == 1 and output_memory_type == 1:
        temp_k = 3
    else:
        temp_k = 4
    fusion_type_new = fusion_type_new * 10 + temp_k
    info_dict = {
        "op_type": 'depthwise_conv2d_forward',
        "A_shape": fmap_shape_NC1HWCO,
        "B_shape": shape_w_NC1HWCO,
        "A_dtype": dtype,
        "B_dtype": dtype,
        "C_dtype": c_dtype,
        "mad_dtype": mad_dtype,
        "padu": padding[0],
        "padd": padding[1],
        "padl": padding[2],
        "padr": padding[3],
        "strideH": stride[0],
        "strideW": stride[1],
        "dilationH": 1,
        "dilationW": 1,
        "group": group_num,
        "bias_flag": bias,
        "fused_double_operand_num": fused_num,
        "in_fm_memory_type": [input_memory_type],
        "out_fm_memory_type": [output_memory_type],
        "l1_fusion_type": l1_fusion_type,
        "fusion_type": fusion_type_new,
        "fm_l1_valid_size": fm_l1_valid_size,
        "kernel_name": kernel_name.value
    }

    tiling = get_tiling(info_dict)
    return tiling


def _get_tiling_fetch(mad_dtype, tensor_dict):
    pad_top = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][0])
    pad_right = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][3])
    pad_left = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][2])
    pad_bottom = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][1])
    stride_w = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][1])
    stride_h = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][0])
    kernel_name = tensor_dict["kernel_name"]

    bias_flag = tensor_dict["flag_is_dequant_bias"] or tensor_dict[
        "flag_is_requant_bias"] or tensor_dict["bias_flag"]
    fused_double_operand_num, not_fused_flag = _get_fused_double_operand_num(
        tensor_dict)

    fmap_shape = tensor_dict["fmap"].shape
    if tensor_dict["fmap_valid_shape"]:
        fmap_shape = tensor_dict["fmap_valid_shape"]
    tiling = _tiling_fetch_all(
        fmap_shape, tensor_dict["filter_buf"], tensor_dict["group_num"],
        [pad_top, pad_bottom, pad_left, pad_right], [stride_h, stride_w],
        mad_dtype, fused_double_operand_num, tensor_dict["input_memory_type"],
        tensor_dict["output_memory_type"], tensor_dict["l1_fusion_type"],
        tensor_dict["fusion_type_new"], tensor_dict["fm_l1_valid_size"],
        bias_flag, tensor_dict["fused_c_dtype"], not_fused_flag, kernel_name)
    return tiling


def _set_a_cbuf_row_major(mad_dtype, a_cbuf_row_major, wo, sch):
    if mad_dtype == "int32":
        sch[a_cbuf_row_major].buffer_align((1, 1), (1, 1), (wo, wo), (1, 1),
                                           (1, 1), (1, 1), (1, 32))
    else:
        sch[a_cbuf_row_major].buffer_align((1, 1), (1, 1), (wo, wo), (1, 1),
                                           (1, 1), (1, 1), (1, BLOCK_SIZE))
    return sch


def _set_common_flag():
    tensor_dict = {}
    tensor_dict["bias_flag"] = False
    tensor_dict["flag_is_dequant"] = False
    tensor_dict["flag_is_dequant_bias"] = False
    tensor_dict["flag_is_requant_bias"] = False
    tensor_dict["flag_is_dequant_sqrt"] = False
    tensor_dict["flag_is_dequant2"] = False
    tensor_dict["flag_is_dequant_quant"] = False
    tensor_dict["flag_is_quant_sqrt"] = False
    tensor_dict["flag_is_quant_relu6_dequant"] = False
    tensor_dict["flag_is_quant_mul_dequant"] = False
    tensor_dict["flag_is_dequant_mul"] = False
    tensor_dict["flag_is_dequant2_mul"] = False
    tensor_dict["flag_is_dequant_sigmoid_mul"] = False
    tensor_dict["flag_is_dequant2_sigmoid_mul"] = False
    tensor_dict["flag_is_requant"] = False
    tensor_dict["flag_is_sigmoid_mul"] = False
    tensor_dict["flag_is_write_select"] = False
    tensor_dict["flag_is_broadcast"] = False
    tensor_dict["fusion_type_new"] = 0

    tensor_dict["fused_double_operand_num"] = 0
    tensor_dict["fused_c_dtype"] = "int8"
    # group_num is used to distin pre_relu
    tensor_dict["group_num"] = 1

    return tensor_dict


def _deq_scalar_mode(tensor_dict):
    if "deq_reg" in tensor_dict:
        if int(tensor_dict["deq_reg"].shape[1]) == 1:
            tensor_dict["sca_vec_flag"] = 0
        else:
            tensor_dict["sca_vec_flag"] = 1
    if "req_reg" in tensor_dict:
        if int(tensor_dict["req_reg"].shape[1]) == 1:
            tensor_dict["sca_vec_flag"] = 0
        else:
            tensor_dict["sca_vec_flag"] = 1
    return tensor_dict


def _set_tensor_by_op_tag(out):
    tensor_dict = _set_common_flag()
    if out.op.tag == "dequant2_remove_pad":
        tensor_dict = _dequant2_remove_pad(out, tensor_dict)
    elif out.op.tag == "dequant_remove_pad":
        tensor_dict = _dequant_remove_pad(out, tensor_dict)
    elif out.op.tag == "quant":
        tensor_dict = _quant(out, tensor_dict)
    elif out.op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
        tensor_dict = _elewise_single_relu(out, tensor_dict)
    elif out.op.tag == "elewise_binary_mul":
        tensor_dict = _elewise_binary_mul(out, tensor_dict)
    elif out.op.tag == "elewise_single_VS_min":
        tensor_dict = _elewise_single_VS_min(out, tensor_dict)
    elif out.op.tag == "depthwise_conv2d":
        tensor_dict = _depthwise_conv2d(out, tensor_dict)
    elif out.op.tag == "requant_remove_pad":
        tensor_dict = _requant_remove_pad(out, tensor_dict)
    elif out.op.tag == "write_select":
        tensor_dict = _write_select(out, tensor_dict)
    else:
        raise RuntimeError("schedule model no surport op.tag %s" %
                           (out.op.tag))
    if out.op.tag == "depthwise_conv2d":
        tensor_dict["kernel_name"] = out.op.attrs["kernel_name"]
    else:
        tensor_dict["kernel_name"] = \
            tensor_dict["depthwise_res"].op.attrs['kernel_name']
    tensor_dict = _deq_scalar_mode(tensor_dict)

    return tensor_dict


def _l1_fusion_check(tensor_dict):
    OFFSET = DepthwiseConv2dParam.fusion_para.get("slice_offset")
    VALID_SHAPE = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    INPUT_MEM_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    if OFFSET and VALID_SHAPE:
        if INPUT_MEM_TYPE == 1:
            tensor_dict["fmap"] = \
                tensor_dict["im2col_row_major"].op.input_tensors[0]
        else:
            tensor_dict["fusion_fmap_select"] = \
                tensor_dict["im2col_row_major"].op.input_tensors[0]
            tensor_dict["fmap"] = \
                tensor_dict["fusion_fmap_select"].op.input_tensors[0]
    else:
        tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[
            0]
    return tensor_dict


def _dequant2_remove_pad(out, tensor_dict):
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant2"] = out.op.input_tensors[0]
    tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict[
            "mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict[
            "mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict[
            "mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict[
            "mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[
        0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict = _l1_fusion_check(tensor_dict)
    tensor_dict["flag_is_dequant2"] = True
    tensor_dict["fusion_type_new"] = 6

    return tensor_dict


def _dequant_remove_pad(out, tensor_dict):
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant1"] = out.op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict[
            "mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict[
            "mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict[
            "mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict[
            "mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[
        0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    tensor_dict = _l1_fusion_check(tensor_dict)
    tensor_dict["flag_is_dequant"] = True
    tensor_dict["fusion_type_new"] = 5

    return tensor_dict


def _requant_remove_pad(out, tensor_dict):
    tensor_dict["data_transfer"] = out.op.input_tensors[0]
    tensor_dict["requant"] = tensor_dict["data_transfer"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["requant"].op.input_tensors[0]
    tensor_dict["vreq_reg"] = tensor_dict["requant"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]

    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_requant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict[
            "mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict[
            "mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict[
            "mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict[
            "mad_bias_ub_brc"].op.input_tensors[0]

    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]

    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[
        0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict = _l1_fusion_check(tensor_dict)
    tensor_dict["flag_is_requant"] = True

    return tensor_dict


def _quant_dequant2(tensor_dict):
    tensor_dict["flag_is_dequant_sqrt"] = True
    if ("max" in tensor_dict and tensor_dict["max"].op.input_tensors[0].name ==
            "dequant2_remove_pad"):
        tensor_dict["quant_remove_pad"] = \
            tensor_dict["max"].op.input_tensors[0]
    elif "mul_res" in tensor_dict and tensor_dict["mul_res"].op.input_tensors[
        0].name == "dequant2_remove_pad":
        tensor_dict["quant_remove_pad"] = \
            tensor_dict["mul_res"].op.input_tensors[0]
    else:
        tensor_dict["quant_remove_pad"] = \
            tensor_dict["input_ub"].op.input_tensors[0]

    tensor_dict["dequant2"] = tensor_dict["quant_remove_pad"].op.input_tensors[
        0]

    tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]

    tensor_dict["depthwise_res"] = \
        tensor_dict["dequant1"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["bias_flag"] = True
        if tensor_dict["depthwise_res"].op.attrs['dsl_flag'].value == 1:
            tensor_dict["flag_is_dequant_bias"] = True
            tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[
                1]
            tensor_dict["mad_ubuf"] = tensor_dict[
                "depthwise_res"].op.input_tensors[0]
            tensor_dict["mad_after_bias"] = tensor_dict[
                "mad_ubuf"].op.input_tensors[0]
            tensor_dict["mad_bias"] = tensor_dict[
                "mad_after_bias"].op.input_tensors[0]
            tensor_dict["mad"] = tensor_dict[
                "mad_after_bias"].op.input_tensors[1]
            tensor_dict["mad_bias_ub_brc"] = tensor_dict[
                "mad_bias"].op.input_tensors[0]
            tensor_dict["bias_gm"] = tensor_dict[
                "mad_bias_ub_brc"].op.input_tensors[0]
            tensor_dict["im2col_fractal"] = tensor_dict[
                "mad"].op.input_tensors[0]
            tensor_dict["filter_reshape"] = tensor_dict[
                "mad"].op.input_tensors[1]
            tensor_dict["filter_buf"] = tensor_dict[
                "filter_reshape"].op.input_tensors[0]
            tensor_dict["im2col_row_major"] = tensor_dict[
                "im2col_fractal"].op.input_tensors[0]
            tensor_dict = _l1_fusion_check(tensor_dict)
            tensor_dict["bias_flag"] = False
    else:
        tensor_dict["flag_is_dequant_bias"] = False
        tensor_dict["mad_ubuf_ori"] = tensor_dict["mad_ubuf"].op.input_tensors[
            0]
        tensor_dict["mad"] = tensor_dict["mad_ubuf_ori"].op.input_tensors[0]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["filter_buf"] = tensor_dict[
            "filter_reshape"].op.input_tensors[0]
        tensor_dict["im2col_row_major"] = tensor_dict[
            "im2col_fractal"].op.input_tensors[0]
        tensor_dict = _l1_fusion_check(tensor_dict)

    return tensor_dict


def _quant_dequant1(tensor_dict):
    tensor_dict["flag_is_dequant_sqrt"] = False
    if "max" in tensor_dict and tensor_dict["max"].op.input_tensors[
        0].name == "dequant_remove_pad":
        tensor_dict["quant_remove_pad"] = \
            tensor_dict["max"].op.input_tensors[0]
    elif "mul_res" in tensor_dict and tensor_dict["mul_res"].op.input_tensors[
        0].name == "dequant_remove_pad":
        tensor_dict["quant_remove_pad"] = \
            tensor_dict["mul_res"].op.input_tensors[0]
    else:
        tensor_dict["quant_remove_pad"] = \
            tensor_dict["input_ub"].op.input_tensors[0]
    tensor_dict["dequant1"] = \
        tensor_dict["quant_remove_pad"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = \
        tensor_dict["dequant1"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["bias_flag"] = True
        if tensor_dict["depthwise_res"].op.attrs['dsl_flag'].value == 1:
            tensor_dict["flag_is_dequant_bias"] = True
            tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[
                1]
            tensor_dict["mad_ubuf"] = tensor_dict[
                "depthwise_res"].op.input_tensors[0]
            tensor_dict["mad_after_bias"] = tensor_dict[
                "mad_ubuf"].op.input_tensors[0]
            tensor_dict["mad_bias"] = tensor_dict[
                "mad_after_bias"].op.input_tensors[0]
            tensor_dict["mad"] = tensor_dict[
                "mad_after_bias"].op.input_tensors[1]
            tensor_dict["mad_bias_ub_brc"] = tensor_dict[
                "mad_bias"].op.input_tensors[0]
            tensor_dict["bias_gm"] = tensor_dict[
                "mad_bias_ub_brc"].op.input_tensors[0]
            tensor_dict["im2col_fractal"] = tensor_dict[
                "mad"].op.input_tensors[0]
            tensor_dict["filter_reshape"] = tensor_dict[
                "mad"].op.input_tensors[1]
            tensor_dict["filter_buf"] = tensor_dict[
                "filter_reshape"].op.input_tensors[0]
            tensor_dict["im2col_row_major"] = tensor_dict[
                "im2col_fractal"].op.input_tensors[0]
            tensor_dict = _l1_fusion_check(tensor_dict)
            tensor_dict["bias_flag"] = False
    else:
        tensor_dict["flag_is_dequant_bias"] = False
        tensor_dict["mad_ubuf_ori"] = tensor_dict["mad_ubuf"].op.input_tensors[
            0]
        tensor_dict["mad"] = tensor_dict["mad_ubuf_ori"].op.input_tensors[0]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["filter_buf"] = tensor_dict[
            "filter_reshape"].op.input_tensors[0]
        tensor_dict["im2col_row_major"] = tensor_dict[
            "im2col_fractal"].op.input_tensors[0]
        tensor_dict = _l1_fusion_check(tensor_dict)

    return tensor_dict


def _quant(out, tensor_dict):
    tensor_dict["flag_is_dequant_quant"] = True
    tensor_dict["fusion_type_new"] = 7
    if out.op.attrs['scale'].value == 1 and out.op.attrs[
        'sqrt_mode'].value == 0:
        tensor_dict["flag_is_quant_sqrt"] = False
        tensor_dict["cast_i8_ub"] = out.op.input_tensors[0]
        tensor_dict["reform_by_vadds"] = tensor_dict[
            "cast_i8_ub"].op.input_tensors[0]

        tensor_dict["input_ub"] = tensor_dict[
            "reform_by_vadds"].op.input_tensors[0]

        if "min" in tensor_dict["input_ub"].op.input_tensors[0].name:
            tensor_dict["flag_is_quant_relu6_dequant"] = True
            tensor_dict["fusion_type_new"] = 8
            tensor_dict["min"] = tensor_dict["input_ub"].op.input_tensors[0]
            if "max" in tensor_dict["min"].op.input_tensors[0].name:
                tensor_dict["max"] = tensor_dict["min"].op.input_tensors[0]
        if "mul" in tensor_dict["input_ub"].op.input_tensors[0].name:
            tensor_dict["flag_is_quant_mul_dequant"] = True
            tensor_dict["fusion_type_new"] = 9
            tensor_dict["mul_res"] = tensor_dict["input_ub"].op.input_tensors[0]
            if "broadcast" in tensor_dict["mul_res"].op.input_tensors[1].name:
                tensor_dict["broadcast_tensor_0"] = tensor_dict[
                    "mul_res"].op.input_tensors[1]
                tensor_dict["float16_mul_input_tensor"] = tensor_dict[
                    "broadcast_tensor_0"].op.input_tensors[0]
                tensor_dict["flag_is_broadcast"] = True
            else:
                tensor_dict["float16_mul_input_tensor"] = tensor_dict[
                    "mul_res"].op.input_tensors[1]

        if tensor_dict["input_ub"].op.input_tensors[
            0].name == "dequant2_remove_pad" or \
                ("max" in tensor_dict and tensor_dict["max"].op.input_tensors[
                    0].name == "dequant2_remove_pad"):
            tensor_dict = _quant_dequant2(tensor_dict)
        else:
            tensor_dict = _quant_dequant1(tensor_dict)
    else:
        raise RuntimeError(
            "quant model only surport scale ==1 and sqrt == 0,"
            " but scale %d, sqrt %d" %
            (out.op.attrs['scale'].value, out.op.attrs['sqrt_mode'].value))

    return tensor_dict


def _elewise_single_relu(out, tensor_dict):
    tensor_dict["depthwise_res"] = out.op.input_tensors[0]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.input_tensors[
        0].name == "bias_add_vector_cc":
        tensor_dict["bias_add"] = tensor_dict[
            "depthwise_res"].op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[
            1]
        tensor_dict["bias_flag"] = True
        tensor_dict["fused_double_operand_num"] = 1
    tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[
        0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _elewise_deq_sigmiod_mul(out, tensor_dict):
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["rec_7"] = out.op.input_tensors[1]
    tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[0]
    tensor_dict["rec_6"] = tensor_dict["rec_7"].op.input_tensors[0]
    tensor_dict["rec_5"] = tensor_dict["rec_6"].op.input_tensors[0]
    tensor_dict["rec_4"] = tensor_dict["rec_5"].op.input_tensors[0]
    tensor_dict["add_2"] = tensor_dict["rec_4"].op.input_tensors[0]
    tensor_dict["rec_3"] = tensor_dict["rec_4"].op.input_tensors[1]
    tensor_dict["exp"] = tensor_dict["add_2"].op.input_tensors[0]
    tensor_dict["muls"] = tensor_dict["exp"].op.input_tensors[0]
    if tensor_dict["muls"].op.input_tensors[0].op.tag == "dequant_remove_pad":
        tensor_dict["dequant_remove_pad"] = \
            tensor_dict["muls"].op.input_tensors[0]
        tensor_dict["dequant1"] = \
            tensor_dict["dequant_remove_pad"].op.input_tensors[0]
        tensor_dict["flag_is_dequant_sigmoid_mul"] = True
        tensor_dict["fusion_type_new"] = 12
    elif tensor_dict["muls"].op.input_tensors[
        0].op.tag == "dequant2_remove_pad":
        tensor_dict["dequant2_remove_pad"] = \
            tensor_dict["muls"].op.input_tensors[0]
        tensor_dict["dequant2"] = \
            tensor_dict["dequant2_remove_pad"].op.input_tensors[0]
        tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]
        tensor_dict["flag_is_dequant2_sigmoid_mul"] = True
        tensor_dict["fusion_type_new"] = 13
    if tensor_dict["muls"].op.input_tensors[0].op.tag == "depthwise_conv2d":
        tensor_dict["depthwise_res"] = tensor_dict["muls"].op.input_tensors[0]
        tensor_dict["flag_is_sigmoid_mul"] = True
        tensor_dict["fusion_type_new"] = 3
    else:
        tensor_dict["depthwise_res"] = \
            tensor_dict["dequant1"].op.input_tensors[0]
        tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.input_tensors[0].name == \
            "bias_add_vector_cc":
        tensor_dict["bias_add"] = tensor_dict[
            "depthwise_res"].op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[
            1]
        tensor_dict["bias_flag"] = True
        tensor_dict["fused_double_operand_num"] = 2
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    elif tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict[
            "mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict[
            "mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict[
            "mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict[
            "mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = \
        tensor_dict["filter_reshape"].op.input_tensors[
            0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _elewise_deq_mul(out, tensor_dict):
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant_remove_pad"] = out.op.input_tensors[0]
    if "broadcast" in out.op.input_tensors[1].name:
        tensor_dict["broadcast_tensor_0"] = out.op.input_tensors[1]
        tensor_dict["float16_mul_input_tensor"] = tensor_dict[
            "broadcast_tensor_0"].op.input_tensors[0]
        tensor_dict["flag_is_broadcast"] = True
    else:
        tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[1]
    tensor_dict["dequant1"] = \
        tensor_dict["dequant_remove_pad"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[
        0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[
        0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict[
            "mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict[
            "mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[
            1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict[
            "mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict[
            "mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = \
        tensor_dict["filter_reshape"].op.input_tensors[
            0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[
        0]
    tensor_dict["flag_is_dequant_mul"] = True
    tensor_dict["fusion_type_new"] = 10

    return tensor_dict


def _elewise_binary_mul(out, tensor_dict):
    if out.op.input_tensors[1].op.name[:3] == "rec":
        tensor_dict = _elewise_deq_sigmiod_mul(out, tensor_dict)
    elif out.op.input_tensors[0].op.tag == 'dequant_remove_pad':
        tensor_dict = _elewise_deq_mul(out, tensor_dict)

    elif out.op.input_tensors[0].op.tag == 'dequant2_remove_pad':
        tensor_dict["fused_c_dtype"] = "float16"
        tensor_dict["dequant2_remove_pad"] = out.op.input_tensors[0]
        if "broadcast" in out.op.input_tensors[1].name:
            tensor_dict["broadcast_tensor_0"] = out.op.input_tensors[1]
            tensor_dict["float16_mul_input_tensor"] = tensor_dict[
                "broadcast_tensor_0"].op.input_tensors[0]
            tensor_dict["flag_is_broadcast"] = True
        else:
            tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[1]
        tensor_dict["dequant2"] = \
            tensor_dict["dequant2_remove_pad"].op.input_tensors[0]
        tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]
        tensor_dict["depthwise_res"] = tensor_dict[
            "dequant1"].op.input_tensors[0]
        tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
        tensor_dict["mad_ubuf"] = tensor_dict[
            "depthwise_res"].op.input_tensors[0]
        if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
            tensor_dict["flag_is_dequant_bias"] = True
            tensor_dict["mad_after_bias"] = tensor_dict[
                "mad_ubuf"].op.input_tensors[0]
            tensor_dict["mad_bias"] = tensor_dict[
                "mad_after_bias"].op.input_tensors[0]
            tensor_dict["mad"] = tensor_dict[
                "mad_after_bias"].op.input_tensors[1]
            tensor_dict["mad_bias_ub_brc"] = tensor_dict[
                "mad_bias"].op.input_tensors[0]
            tensor_dict["bias_gm"] = tensor_dict[
                "mad_bias_ub_brc"].op.input_tensors[0]
        else:
            tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["filter_buf"] = \
            tensor_dict["filter_reshape"].op.input_tensors[
                0]
        tensor_dict["im2col_row_major"] = tensor_dict[
            "im2col_fractal"].op.input_tensors[0]
        tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[
            0]
        tensor_dict["flag_is_dequant2_mul"] = True
        tensor_dict["fusion_type_new"] = 11
    else:
        tensor_dict["fusion_type_new"] = 2
        tensor_dict["depthwise_res"] = out.op.input_tensors[0]

        if "broadcast" in out.op.input_tensors[1].name:
            tensor_dict["broadcast_tensor_0"] = out.op.input_tensors[1]
            tensor_dict["float16_mul_input_tensor"] = tensor_dict[
                "broadcast_tensor_0"].op.input_tensors[0]
            tensor_dict["flag_is_broadcast"] = True
        else:
            tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[1]
        tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[
            0]

        if tensor_dict["depthwise_res"].op.input_tensors[0].name == \
                "bias_add_vector_cc":
            tensor_dict["bias_add"] = tensor_dict[
                "depthwise_res"].op.input_tensors[0]
            tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[
                0]
            tensor_dict["bias_tensor"] = \
                tensor_dict["bias_add"].op.input_tensors[
                    1]
            tensor_dict["bias_flag"] = True
            tensor_dict["fused_double_operand_num"] = 1
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_buf"] = \
            tensor_dict["filter_reshape"].op.input_tensors[
                0]
        tensor_dict["im2col_row_major"] = tensor_dict[
            "im2col_fractal"].op.input_tensors[0]
        tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[
            0]
    return tensor_dict


def _depthwise_conv2d(out, tensor_dict):
    tensor_dict["mad_ubuf"] = out.op.input_tensors[0]
    if tensor_dict["mad_ubuf"].dtype == "float16" \
            and out.op.attrs['bias_flag'].value == 1 \
            or (tensor_dict["mad_ubuf"].dtype != "float16" and
                out.op.attrs['bias_flag'].value == 1 and
                out.op.attrs['dsl_flag'].value == 0):
        tensor_dict["bias_add"] = out.op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[
            1]
        tensor_dict["bias_flag"] = True

    tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[
        0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
        tensor_dict["group_num"] = 2
        tensor_dict["relu_0"] = tensor_dict[
            "im2col_row_major"].op.input_tensors[0]
        tensor_dict["fmap"] = tensor_dict["relu_0"].op.input_tensors[0]
    return tensor_dict


def _elewise_single_VS_min(out, tensor_dict):
    tensor_dict["max_0"] = out.op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["max_0"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.input_tensors[
        0].name == "bias_add_vector_cc":
        tensor_dict["bias_add"] = tensor_dict[
            "depthwise_res"].op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[
            1]
        tensor_dict["bias_flag"] = True
        tensor_dict["fused_double_operand_num"] = 1
    else:
        tensor_dict["mad_ubuf"] = tensor_dict[
            "depthwise_res"].op.input_tensors[0]
    tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[
        0]
    tensor_dict["im2col_row_major"] = tensor_dict[
        "im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _write_select(out, tensor_dict):
    tensor_dict["flag_is_write_select"] = True
    tensor_dict["write_select"] = out.op.input_tensors[0]
    if out.op.input_tensors[0].op.tag == "quant":
        tensor_dict = _quant(out.op.input_tensors[0], tensor_dict)
    elif out.op.input_tensors[0].op.tag == "dequant2_remove_pad":
        tensor_dict = _dequant2_remove_pad(out.op.input_tensors[0],
                                           tensor_dict)
    elif out.op.input_tensors[0].op.tag == "dequant_remove_pad":
        tensor_dict = _dequant_remove_pad(out.op.input_tensors[0], tensor_dict)
    else:
        raise RuntimeError("schedule model no surport op.tag %s" %
                           (out.op.tag))
    return tensor_dict


def _check_broadcast(tensor_dict, sch, attrs_dict):
    if tensor_dict["flag_is_broadcast"]:
        float16_mul_input_ubuf = sch.cache_read(
            tensor_dict["float16_mul_input_tensor"],
            cce_params.scope_ubuf,
            [tensor_dict["broadcast_tensor_0"]])
        sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
    else:
        float16_mul_input_ubuf = sch.cache_read(
            tensor_dict["float16_mul_input_tensor"],
            cce_params.scope_ubuf, [attrs_dict["mul_ubuf"]])
    return float16_mul_input_ubuf, sch


# phase1, set scope
def _set_sch_int32_phase1(tensor_dict, attrs_dict, out, sch):
    dequant_ubuf = None
    deq_reg_ubuf = None
    requant_ubuf = None
    req_reg_ubuf = None
    bias_ub = None
    buf = (dequant_ubuf, deq_reg_ubuf, requant_ubuf, req_reg_ubuf, bias_ub)

    def _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch):
        dequant_ubuf, deq_reg_ubuf, requant_ubuf, req_reg_ubuf, bias_ub = buf
        if tensor_dict["flag_is_write_select"]:
            sch[tensor_dict["write_select"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["write_select"]].compute_inline()
        if tensor_dict["flag_is_dequant"]:
            deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"],
                                          cce_params.scope_ubuf,
                                          tensor_dict["dequant1"])
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            dequant_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        elif tensor_dict["flag_is_dequant2"]:
            sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
            deq_reg_ubuf = sch.cache_read(
                tensor_dict["deq_reg"], cce_params.scope_ubuf,
                (tensor_dict["dequant1"], tensor_dict["dequant2"]))
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["dequant1"]].compute_inline()
        elif tensor_dict["flag_is_requant"]:
            sch[tensor_dict["data_transfer"]].set_scope(cce_params.scope_ubuf)
            req_reg_ubuf = sch.cache_read(tensor_dict["vreq_reg"],
                                          cce_params.scope_ubuf,
                                          tensor_dict["requant"])
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            sch[tensor_dict["requant"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["requant"]].compute_inline()

        if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
            bias_ub = sch.cache_read(tensor_dict["bias_gm"],
                                     cce_params.scope_ubuf,
                                     [tensor_dict["mad_bias_ub_brc"]])
            sch[tensor_dict["mad_bias_ub_brc"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
            sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
        if tensor_dict["flag_is_requant"]:
            return deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch
        return deq_reg_ubuf, bias_ub, dequant_ubuf, sch

    def _set_sch_int32_phase1_dequant_quant(tensor_dict, attrs_dict, out, buf,
                                            sch):
        dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
        if tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict[
            "flag_is_quant_sqrt"]:
            deq_reg_ubuf = sch.cache_read(
                tensor_dict["deq_reg"], cce_params.scope_ubuf,
                (tensor_dict["dequant1"], tensor_dict["dequant2"]))
            sch[tensor_dict["cast_i8_ub"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["reform_by_vadds"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["input_ub"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["input_ub"]].compute_inline()
            sch[tensor_dict["quant_remove_pad"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["quant_remove_pad"]].compute_inline()
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].set_scope(
                    cce_params.scope_ubuf)
                sch[tensor_dict["write_select"]].compute_inline()
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_dict["min"]].compute_inline()
                sch[tensor_dict["max"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_dict["max"]].compute_inline()
            elif tensor_dict["flag_is_quant_mul_dequant"]:
                sch[tensor_dict["mul_res"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_dict["mul_res"]].compute_inline()
                mul_ubuf = sch.cache_write(tensor_dict["mul_res"],
                                           cce_params.scope_ubuf)
                attrs_dict["mul_ubuf"] = mul_ubuf
                float16_mul_input_ubuf, sch = _check_broadcast(tensor_dict, sch,
                                                               attrs_dict)

                tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
            sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["dequant1"]].compute_inline()
            sch[tensor_dict["dequant2"]].compute_inline()
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            sch[tensor_dict["mad_ubuf"]].compute_inline()
            if tensor_dict["flag_is_dequant_bias"]:
                bias_ub = sch.cache_read(tensor_dict["bias_gm"],
                                         cce_params.scope_ubuf,
                                         [tensor_dict["mad_bias_ub_brc"]])
                sch[tensor_dict["mad_bias_ub_brc"]].set_scope(
                    cce_params.scope_ubuf)
                sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
                sch[tensor_dict["mad_after_bias"]].set_scope(
                    cce_params.scope_cc)
        elif not tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict[
            "flag_is_quant_sqrt"]:
            deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"],
                                          cce_params.scope_ubuf,
                                          tensor_dict["dequant1"])
            sch[tensor_dict["cast_i8_ub"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["reform_by_vadds"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["input_ub"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["quant_remove_pad"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["quant_remove_pad"]].compute_inline()
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].set_scope(
                    cce_params.scope_ubuf)
                sch[tensor_dict["write_select"]].compute_inline()
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_dict["min"]].compute_inline()
                sch[tensor_dict["max"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_dict["max"]].compute_inline()
            elif tensor_dict["flag_is_quant_mul_dequant"]:
                sch[tensor_dict["mul_res"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_dict["mul_res"]].compute_inline()
                mul_ubuf = sch.cache_write(tensor_dict["mul_res"],
                                           cce_params.scope_ubuf)
                attrs_dict["mul_ubuf"] = mul_ubuf
                float16_mul_input_ubuf, sch = _check_broadcast(tensor_dict, sch,
                                                               attrs_dict)

                tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["dequant1"]].compute_inline()
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            sch[tensor_dict["mad_ubuf"]].compute_inline()
            if tensor_dict["flag_is_dequant_bias"]:
                bias_ub = sch.cache_read(tensor_dict["bias_gm"],
                                         cce_params.scope_ubuf,
                                         [tensor_dict["mad_bias_ub_brc"]])
                sch[tensor_dict["mad_bias_ub_brc"]].set_scope(
                    cce_params.scope_ubuf)
                sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
                sch[tensor_dict["mad_after_bias"]].set_scope(
                    cce_params.scope_cc)
        else:
            raise RuntimeError(
                "quant model only surport scale ==1 and sqrt == 0,"
                " but scale %d, sqrt %d" %
                (out.op.attrs['scale'].value, out.op.attrs['sqrt_mode'].value))
        return deq_reg_ubuf, bias_ub, dequant_ubuf, sch

    def _set_sch_int32_phase1_dequant_mul(tensor_dict, attrs_dict, out, buf,
                                          sch):
        dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
        if tensor_dict["flag_is_dequant_mul"]:
            deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"],
                                          cce_params.scope_ubuf,
                                          tensor_dict["dequant1"])
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            sch[mul_ubuf].compute_inline()
            attrs_dict["mul_ubuf"] = mul_ubuf
            if tensor_dict["flag_is_broadcast"]:
                float16_mul_input_ubuf = sch.cache_read(
                    tensor_dict["float16_mul_input_tensor"],
                    cce_params.scope_ubuf,
                    [tensor_dict["broadcast_tensor_0"]])
                sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
            else:
                float16_mul_input_ubuf = sch.cache_read(
                    tensor_dict["float16_mul_input_tensor"],
                    cce_params.scope_ubuf, [attrs_dict["mul_ubuf"]])
            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
            sch[tensor_dict["dequant_remove_pad"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["dequant_remove_pad"]].compute_inline()
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["mad_ubuf"]].compute_inline()
            dequant_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            sch[dequant_ubuf].compute_inline()
        elif tensor_dict["flag_is_dequant2_mul"]:
            sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
            deq_reg_ubuf = sch.cache_read(
                tensor_dict["deq_reg"], cce_params.scope_ubuf,
                (tensor_dict["dequant1"], tensor_dict["dequant2"]))
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            sch[mul_ubuf].compute_inline()
            attrs_dict["mul_ubuf"] = mul_ubuf

            if tensor_dict["flag_is_broadcast"]:
                float16_mul_input_ubuf = sch.cache_read(
                    tensor_dict["float16_mul_input_tensor"],
                    cce_params.scope_ubuf,
                    [tensor_dict["broadcast_tensor_0"]])
                sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
            else:
                float16_mul_input_ubuf = sch.cache_read(
                    tensor_dict["float16_mul_input_tensor"],
                    cce_params.scope_ubuf, [attrs_dict["mul_ubuf"]])

            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
            sch[tensor_dict["dequant2_remove_pad"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["dequant2_remove_pad"]].compute_inline()
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["mad_ubuf"]].compute_inline()
            sch[tensor_dict["dequant1"]].compute_inline()
            sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["dequant2"]].compute_inline()
        if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
            bias_ub = sch.cache_read(tensor_dict["bias_gm"],
                                     cce_params.scope_ubuf,
                                     [tensor_dict["mad_bias_ub_brc"]])
            sch[tensor_dict["mad_bias_ub_brc"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
            sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
        return deq_reg_ubuf, bias_ub, dequant_ubuf, sch

    def _set_sch_int32_phase1_dequant_sigmoid_mul(tensor_dict, attrs_dict, out,
                                                  buf, sch):
        dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
        if tensor_dict["flag_is_dequant_sigmoid_mul"]:
            deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"],
                                          cce_params.scope_ubuf,
                                          tensor_dict["dequant1"])
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            sch[mul_ubuf].compute_inline()
            attrs_dict["mul_ubuf"] = mul_ubuf
            float16_mul_input_ubuf = sch.cache_read(
                tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                [attrs_dict["mul_ubuf"]])
            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
            sch[tensor_dict["rec_7"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_7"]].compute_inline()
            sch[tensor_dict["rec_6"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_6"]].compute_inline()
            sch[tensor_dict["rec_5"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_5"]].compute_inline()
            sch[tensor_dict["rec_4"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_4"]].compute_inline()
            sch[tensor_dict["rec_3"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_3"]].compute_inline()
            sch[tensor_dict["muls"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["muls"]].compute_inline()
            sch[tensor_dict["exp"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["exp"]].compute_inline()
            sch[tensor_dict["add_2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["add_2"]].compute_inline()
            sch[tensor_dict["dequant_remove_pad"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["dequant_remove_pad"]].compute_inline()
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["mad_ubuf"]].compute_inline()
            dequant_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            sch[dequant_ubuf].compute_inline()
        elif tensor_dict["flag_is_dequant2_sigmoid_mul"]:
            sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
            deq_reg_ubuf = sch.cache_read(
                tensor_dict["deq_reg"], cce_params.scope_ubuf,
                (tensor_dict["dequant1"], tensor_dict["dequant2"]))
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            sch[mul_ubuf].compute_inline()
            attrs_dict["mul_ubuf"] = mul_ubuf
            float16_mul_input_ubuf = sch.cache_read(
                tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                [attrs_dict["mul_ubuf"]])
            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
            sch[tensor_dict["rec_7"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_7"]].compute_inline()
            sch[tensor_dict["rec_6"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_6"]].compute_inline()
            sch[tensor_dict["rec_5"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_5"]].compute_inline()
            sch[tensor_dict["rec_4"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_4"]].compute_inline()
            sch[tensor_dict["rec_3"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_3"]].compute_inline()
            sch[tensor_dict["muls"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["muls"]].compute_inline()
            sch[tensor_dict["exp"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["exp"]].compute_inline()
            sch[tensor_dict["add_2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["add_2"]].compute_inline()
            sch[tensor_dict["dequant2_remove_pad"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["dequant2_remove_pad"]].compute_inline()
            sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["mad_ubuf"]].compute_inline()
            sch[tensor_dict["dequant1"]].compute_inline()
            sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["dequant2"]].compute_inline()
        if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
            bias_ub = sch.cache_read(tensor_dict["bias_gm"],
                                     cce_params.scope_ubuf,
                                     [tensor_dict["mad_bias_ub_brc"]])
            sch[tensor_dict["mad_bias_ub_brc"]].set_scope(
                cce_params.scope_ubuf)
            sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
            sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
        return deq_reg_ubuf, bias_ub, dequant_ubuf, sch

    if tensor_dict["flag_is_dequant"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant2"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch)
    elif tensor_dict["flag_is_requant"]:
        deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch = \
            _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch)

    elif tensor_dict["flag_is_dequant_quant"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant_quant(tensor_dict, attrs_dict, out,
                                                buf, sch)

    elif tensor_dict["flag_is_dequant_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant_mul(tensor_dict, attrs_dict, out, buf,
                                              sch)
    elif tensor_dict["flag_is_dequant2_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant_mul(tensor_dict, attrs_dict, out, buf,
                                              sch)

    elif tensor_dict["flag_is_dequant_sigmoid_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant_sigmoid_mul(
                tensor_dict, attrs_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = \
            _set_sch_int32_phase1_dequant_sigmoid_mul(
                tensor_dict, attrs_dict, out, buf, sch)
    return deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch


# phase2, compute at
def _sch_flag_is_dequant2(sch, tensor_dict, attrs_dict, res_cut_dict):
    a2_axis = None
    a3_axis = None
    sch[attrs_dict["deq_reg_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                               res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]],
                                            res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant2"]].compute_at(sch[attrs_dict["out"]],
                                            res_cut_dict["res_mcut_iio"])
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(
            sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(
            sch[tensor_dict["mad_bias"]].op.axis[0],
            sch[tensor_dict["mad_bias"]].op.axis[1],
            sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
            sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]],
                                                res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]],
                                              res_cut_dict["res_mcut_io"])
    if tensor_dict["flag_is_dequant2_mul"]:
        sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                               res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                 (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                               res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_7"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_6"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_5"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_4"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_3"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["muls"]].compute_at(sch[attrs_dict["out"]],
                                            res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["add_2"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["exp"]].compute_at(sch[attrs_dict["out"]],
                                           res_cut_dict["res_mcut_iio"])
        sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                 (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant2"]].buffer_align(
        (1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))

    return a2_axis, a3_axis, sch


def _sch_flag_is_dequant(sch, tensor_dict, attrs_dict, res_cut_dict):
    a2_axis = None
    a3_axis = None
    sch[attrs_dict["deq_reg_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                               res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]],
                                            res_cut_dict["res_mcut_iio"])
    if not tensor_dict["flag_is_dequant_sigmoid_mul"] and not tensor_dict[
        "flag_is_dequant2_sigmoid_mul"] and not tensor_dict[
        "flag_is_dequant_mul"]:
        sch[attrs_dict["dequant_ubuf"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(
            sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(
            sch[tensor_dict["mad_bias"]].op.axis[0],
            sch[tensor_dict["mad_bias"]].op.axis[1],
            sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
            sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]],
                                                res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]],
                                              res_cut_dict["res_mcut_io"])
        if tensor_dict["flag_is_dequant_mul"]:
            sch[attrs_dict["mul_ubuf"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                     (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_dequant_sigmoid_mul"]:
        sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                               res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_7"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_6"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_5"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_4"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_3"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["muls"]].compute_at(sch[attrs_dict["out"]],
                                            res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["add_2"]].compute_at(sch[attrs_dict["out"]],
                                             res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["exp"]].compute_at(sch[attrs_dict["out"]],
                                           res_cut_dict["res_mcut_iio"])
        sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                 (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant1"]].buffer_align(
        (1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))

    return a2_axis, a3_axis, sch


def _sch_flag_is_dequant_quant(sch, double_buffer_flag, tensor_dict,
                               attrs_dict, res_cut_dict):
    a2_axis = None
    a3_axis = None

    def _sch_deqaunt_quant_compute_at():
        if tensor_dict["flag_is_dequant_bias"]:
            a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(
                sch[tensor_dict["mad_bias"]].op.axis[3], 16)
            sch[tensor_dict["mad_bias"]].reorder(
                sch[tensor_dict["mad_bias"]].op.axis[0],
                sch[tensor_dict["mad_bias"]].op.axis[1],
                sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
                sch[tensor_dict["mad_bias"]].op.axis[4])
            sch[tensor_dict["mad_after_bias"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])

            sch[tensor_dict["mad_bias"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
            sch[tensor_dict["mad_bias_ub_brc"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])

            sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]],
                                                  res_cut_dict["res_cccut_i"])
            sch[attrs_dict["deq_reg_ubuf"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_cccut_i"])

            sch[tensor_dict["mad_bias_ub_brc"]].double_buffer()
            sch[tensor_dict["mad_bias_ub_brc"]].preload()
            if double_buffer_flag["CL0_pbuffer"] == 2:
                sch[tensor_dict["mad_bias"]].double_buffer()
                sch[tensor_dict["mad_bias"]].preload()

            sch[tensor_dict["cast_i8_ub"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["reform_by_vadds"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["input_ub"]].compute_at(
                sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].compute_at(
                    sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].compute_at(
                    sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["max"]].compute_at(
                    sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            if tensor_dict["flag_is_quant_mul_dequant"]:
                sch[attrs_dict["mul_ubuf"]].compute_at(
                    sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
                    sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[attrs_dict["mul_ubuf"]].buffer_align(
                    (1, 1), (1, 1), (1, 1), (1, 16), (1, BLOCK_SIZE))
        else:
            raise RuntimeError(
                "unspport mode now, dequant fused mode must have bias")

        return a2_axis, a3_axis, sch

    if tensor_dict[
        "flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        a2_axis, a3_axis, sch = _sch_deqaunt_quant_compute_at()
        sch[tensor_dict["dequant2"]].compute_at(sch[attrs_dict["out"]],
                                                res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]],
                                                res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["dequant2"]].buffer_align(
            (1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    elif not tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict[
        "flag_is_quant_sqrt"]:
        a2_axis, a3_axis, sch = _sch_deqaunt_quant_compute_at()
        sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]],
                                                res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["dequant1"]].buffer_align(
            (1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    else:
        raise RuntimeError("quant model only surport scale ==1 and sqrt == 0,"
                           " but scale %d, sqrt %d" %
                           (attrs_dict["out"].op.attrs['scale'].value,
                            attrs_dict["out"].op.attrs['sqrt_mode'].value))

    return a2_axis, a3_axis, sch


def _sch_flag_is_requant(sch, tensor_dict, attrs_dict, res_cut_dict):
    a2_axis = None
    a3_axis = None
    sch[attrs_dict["req_reg_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                               res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["data_transfer"]].compute_at(sch[attrs_dict["out"]],
                                                 res_cut_dict["res_mcut_iio"])

    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(
            sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(
            sch[tensor_dict["mad_bias"]].op.axis[0],
            sch[tensor_dict["mad_bias"]].op.axis[1],
            sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
            sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]],
                                                res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]],

                                              res_cut_dict["res_mcut_io"])
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].compute_at(
            sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["data_transfer"]].buffer_align(
        (1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    return a2_axis, a3_axis, sch


def _set_sch_int32_phase2(mad_dtype, double_buffer_flag, tensor_dict,
                          attrs_dict, res_cut_dict, sch):
    a2_axis = None
    a3_axis = None
    if mad_dtype == "int32":
        if tensor_dict["flag_is_dequant2"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant2(
                sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant(
                sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant_quant"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant_quant(
                sch, double_buffer_flag, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant(
                sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant_sigmoid_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant(
                sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant2_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant2(
                sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_requant"]:
            a2_axis, a3_axis, sch = _sch_flag_is_requant(
                sch, tensor_dict, attrs_dict, res_cut_dict)

        elif tensor_dict["flag_is_dequant2_sigmoid_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant2(
                sch, tensor_dict, attrs_dict, res_cut_dict)
    return a2_axis, a3_axis, sch


def _emit_insn_dequant1(tensor_dict, sch):
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(
            sch[tensor_dict["dequant1"]].op.axis[3], 'dma_copy')
    else:
        sch[tensor_dict["dequant1"]].pragma(
            sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')


# phase 3, emit insn phase


def _flag_is_dequant_quant(tensor_dict, sch, attrs_dict):
    if tensor_dict[
        "flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        if tensor_dict["flag_is_dequant_bias"]:
            sch[tensor_dict["mad_after_bias"]].emit_insn(
                tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
            sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"],
                                                   'dma_copy')
            sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(
                sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
            sch[attrs_dict["deq_reg_ubuf"]].emit_insn(
                sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
            _emit_insn_dequant1(tensor_dict, sch)
            sch[tensor_dict["dequant2"]].emit_insn(
                sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].emit_insn(
                    sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].emit_insn(
                    sch[tensor_dict["min"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["max"]].emit_insn(
                    sch[tensor_dict["max"]].op.axis[0], 'vector_auto')
            if tensor_dict["flag_is_quant_mul_dequant"]:
                sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(
                    tensor_dict["float16_mul_input_ubuf"].op.axis[0],
                    'dma_copy')
                sch[attrs_dict["mul_ubuf"]].emit_insn(
                    sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
            sch[attrs_dict["bias_ub"]].emit_insn(
                sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
            sch[tensor_dict["input_ub"]].emit_insn(
                sch[tensor_dict["input_ub"]].op.axis[0], 'dma_padding')

            ndim = len(sch[tensor_dict["reform_by_vadds"]].op.axis)
            factor = 16
            coo, _ = sch[tensor_dict["reform_by_vadds"]].split(
                sch[tensor_dict["reform_by_vadds"]].op.axis[ndim - 1], factor)
            axis_list = sch[tensor_dict["reform_by_vadds"]].op.axis[0:ndim - 1]
            sch[tensor_dict["reform_by_vadds"]].reorder(coo, *axis_list)
            sch[tensor_dict["reform_by_vadds"]].emit_insn(
                sch[tensor_dict["reform_by_vadds"]].op.axis[3], 'vector_auto')

            sch[tensor_dict["cast_i8_ub"]].emit_insn(
                sch[tensor_dict["cast_i8_ub"]].op.axis[0], 'vector_conv')
    elif not tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict[
        "flag_is_quant_sqrt"]:
        if tensor_dict["flag_is_dequant_bias"]:
            sch[tensor_dict["mad_after_bias"]].emit_insn(
                tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
            sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"],
                                                   'dma_copy')
            sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(
                sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')

            sch[attrs_dict["deq_reg_ubuf"]].emit_insn(
                sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].emit_insn(
                    sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].emit_insn(
                    sch[tensor_dict["min"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["max"]].emit_insn(
                    sch[tensor_dict["max"]].op.axis[0], 'vector_auto')
            if tensor_dict["flag_is_quant_mul_dequant"]:
                sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(
                    tensor_dict["float16_mul_input_ubuf"].op.axis[0],
                    'dma_copy')
                sch[attrs_dict["mul_ubuf"]].emit_insn(
                    sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
            _emit_insn_dequant1(tensor_dict, sch)
            sch[attrs_dict["bias_ub"]].emit_insn(
                sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
            sch[tensor_dict["input_ub"]].emit_insn(
                sch[tensor_dict["input_ub"]].op.axis[0], 'dma_padding')

            ndim = len(sch[tensor_dict["reform_by_vadds"]].op.axis)
            factor = 16
            coo, _ = sch[tensor_dict["reform_by_vadds"]].split(
                sch[tensor_dict["reform_by_vadds"]].op.axis[ndim - 1], factor)
            axis_list = sch[tensor_dict["reform_by_vadds"]].op.axis[0:ndim - 1]
            sch[tensor_dict["reform_by_vadds"]].reorder(coo, *axis_list)
            sch[tensor_dict["reform_by_vadds"]].emit_insn(
                sch[tensor_dict["reform_by_vadds"]].op.axis[3], 'vector_auto')

            sch[tensor_dict["cast_i8_ub"]].emit_insn(
                sch[tensor_dict["cast_i8_ub"]].op.axis[0], 'vector_conv')
    else:
        raise RuntimeError("quant model only surport scale ==1 and sqrt == 0,"
                           " but scale %d, sqrt %d" %
                           (attrs_dict["out"].op.attrs['scale'].value,
                            attrs_dict["out"].op.attrs['sqrt_mode'].value))
    return sch


def _flag_is_dequant_sigmoid_mul(tensor_dict, sch, attrs_dict):
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        sch[tensor_dict["mad_after_bias"]].emit_insn(
            tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
        sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"],
                                               'dma_copy')
        sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(
            sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["bias_ub"]].emit_insn(
            sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(
        sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
    if tensor_dict['sca_vec_flag'] == 0:
        sch[tensor_dict["dequant1"]].pragma(
            sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
    else:
        sch[tensor_dict["dequant1"]].pragma(
            sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    if tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        sch[tensor_dict["dequant2"]].emit_insn(
            sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["rec_7"]].emit_insn(sch[tensor_dict["rec_7"]].op.axis[0],
                                        'vector_auto')
    sch[tensor_dict["rec_6"]].emit_insn(sch[tensor_dict["rec_6"]].op.axis[0],
                                        'vector_auto')
    sch[tensor_dict["rec_5"]].emit_insn(sch[tensor_dict["rec_5"]].op.axis[0],
                                        'vector_auto')
    sch[tensor_dict["rec_4"]].emit_insn(sch[tensor_dict["rec_4"]].op.axis[0],
                                        'vector_auto')
    sch[tensor_dict["rec_3"]].emit_insn(sch[tensor_dict["rec_3"]].op.axis[0],
                                        'vector_auto')
    sch[tensor_dict["muls"]].emit_insn(sch[tensor_dict["muls"]].op.axis[0],
                                       'vector_auto')
    sch[tensor_dict["add_2"]].emit_insn(sch[tensor_dict["add_2"]].op.axis[0],
                                        'vector_auto')
    sch[tensor_dict["exp"]].emit_insn(sch[tensor_dict["exp"]].op.axis[0],
                                      'vector_auto')
    sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(
        tensor_dict["float16_mul_input_ubuf"].op.axis[0], 'dma_copy')
    sch[attrs_dict["mul_ubuf"]].emit_insn(
        sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
    return sch


def _flag_is_dequant_mul(tensor_dict, sch, attrs_dict):
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        sch[tensor_dict["mad_after_bias"]].emit_insn(
            tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
        sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"],
                                               'dma_copy')
        sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(
            sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["bias_ub"]].emit_insn(
            sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(
        sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(
            sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
    else:
        if tensor_dict['sca_vec_flag'] == 0:
            sch[tensor_dict["dequant1"]].pragma(
                sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
        else:
            sch[tensor_dict["dequant1"]].pragma(
                sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    if tensor_dict["flag_is_dequant2_mul"]:
        sch[tensor_dict["dequant2"]].emit_insn(
            sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(
        tensor_dict["float16_mul_input_ubuf"].op.axis[0], 'dma_copy')
    sch[attrs_dict["mul_ubuf"]].emit_insn(
        sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
    return sch


def _set_sch_int32_phase3(tensor_dict, sch, attrs_dict, res_cut_dict, out):
    def _phase3_avoid_complexity(tensor_dict, sch, attrs_dict):
        sch[attrs_dict["deq_reg_ubuf"]].emit_insn(
            sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
        if cce_conf.is_v200_version_new():
            sch[tensor_dict["dequant1"]].emit_insn(
                sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
        else:
            if tensor_dict['sca_vec_flag'] == 0:
                sch[tensor_dict["dequant1"]].pragma(
                    sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale',
                    'scalar')
            else:
                sch[tensor_dict["dequant1"]].pragma(
                    sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale',
                    'vector')
        sch[tensor_dict["dequant2"]].emit_insn(
            sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')

        return sch

    def _emit_insn_phase3(tensor_dict, sch, attrs_dict):
        if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
            sch[tensor_dict["mad_after_bias"]].emit_insn(
                tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
            sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"],
                                                   'dma_copy')
            sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(
                sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
            sch[attrs_dict["bias_ub"]].emit_insn(
                sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')

        if tensor_dict["flag_is_dequant"]:
            sch[attrs_dict["dequant_ubuf"]].emit_insn(
                sch[attrs_dict["dequant_ubuf"]].op.axis[0], 'dma_copy')
            sch[attrs_dict["deq_reg_ubuf"]].emit_insn(
                sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
            if cce_conf.is_v200_version_new():
                sch[tensor_dict["dequant1"]].emit_insn(
                    sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
            else:
                if tensor_dict['sca_vec_flag'] == 0:
                    sch[tensor_dict["dequant1"]].pragma(
                        sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale',
                        'scalar')
                else:
                    sch[tensor_dict["dequant1"]].pragma(
                        sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale',
                        'vector')

        elif tensor_dict["flag_is_dequant2"]:
            sch = _phase3_avoid_complexity(tensor_dict, sch, attrs_dict)

        elif tensor_dict["flag_is_requant"]:
            sch[attrs_dict["req_reg_ubuf"]].emit_insn(
                sch[attrs_dict["req_reg_ubuf"]].op.axis[0], 'dma_copy')
            if cce_conf.is_v200_version_new():
                sch[tensor_dict["data_transfer"]].emit_insn(
                    sch[tensor_dict["data_transfer"]].op.axis[3], 'dma_copy')
            else:
                if tensor_dict['sca_vec_flag'] == 0:
                    sch[tensor_dict["requant"]].pragma(
                        sch[tensor_dict["requant"]].op.axis[3], 'deq_scale',
                        'scalar')
                else:
                    sch[tensor_dict["requant"]].pragma(
                        sch[tensor_dict["requant"]].op.axis[3], 'deq_scale',
                        'vector')

        if tensor_dict["flag_is_write_select"]:
            sch[tensor_dict["write_select"]].emit_insn(
                sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')

        return sch

    if tensor_dict["flag_is_dequant2"]:
        sch = _emit_insn_phase3(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant"]:
        sch = _emit_insn_phase3(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_requant"]:
        sch = _emit_insn_phase3(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant_quant"]:
        sch = _flag_is_dequant_quant(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant_mul"] or tensor_dict[
        "flag_is_dequant2_mul"]:
        sch = _flag_is_dequant_mul(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant_sigmoid_mul"] or tensor_dict[
        "flag_is_dequant2_sigmoid_mul"]:
        sch = _flag_is_dequant_sigmoid_mul(tensor_dict, sch, attrs_dict)
    elif attrs_dict["out"].op.tag in [
        "elewise_single_relu", "elewise_single_lrelu"
    ]:
        sch[attrs_dict["relu_ubuf"]].emit_insn(
            sch[attrs_dict["relu_ubuf"]].op.axis[0], 'vector_auto')
    elif attrs_dict["out"].op.tag == "elewise_binary_mul":
        if tensor_dict["flag_is_sigmoid_mul"]:
            sch[tensor_dict["rec_7"]].emit_insn(
                sch[tensor_dict["rec_7"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["rec_6"]].emit_insn(
                sch[tensor_dict["rec_6"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["rec_5"]].emit_insn(
                sch[tensor_dict["rec_5"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["rec_4"]].emit_insn(
                sch[tensor_dict["rec_4"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["rec_3"]].emit_insn(
                sch[tensor_dict["rec_3"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["muls"]].emit_insn(
                sch[tensor_dict["muls"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["add_2"]].emit_insn(
                sch[tensor_dict["add_2"]].op.axis[0], 'vector_auto')
            sch[tensor_dict["exp"]].emit_insn(
                sch[tensor_dict["exp"]].op.axis[0], 'vector_auto')
        else:
            sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(
                tensor_dict["float16_mul_input_ubuf"].op.axis[0], 'dma_copy')
        sch[attrs_dict["mul_ubuf"]].emit_insn(
            sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
    elif attrs_dict["out"].op.tag == "elewise_single_VS_min":
        sch[tensor_dict["max_0"]].emit_insn(
            sch[tensor_dict["max_0"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["relu_ubuf"]].emit_insn(
            sch[attrs_dict["relu_ubuf"]].op.axis[0], 'vector_auto')
    # STRIDE WRITE
    if out.op.tag == "write_select":
        align_length = int(out.op.attrs["HWC0"])
        sch[out].bind_buffer(out.op.axis[1], align_length, 0)
    sch[out].emit_insn(res_cut_dict["res_mcut_iii"], 'dma_copy')

    return sch


def _dequant_out_cg(mad_dtype, attrs_dict, block_dim_tiling):
    if mad_dtype == "int32":
        if attrs_dict["deq_reg_ubuf"] is not None:
            _, dequant_out_cg, _, _, _ = \
                (int(i.value) for i in attrs_dict["deq_reg_ubuf"].shape)
            if dequant_out_cg % 2 != 0:
                block_dim_tiling[3] = 1
        elif attrs_dict["req_reg_ubuf"] is not None:
            _, requant_out_cg, _, _, _ = \
                (int(i.value) for i in attrs_dict["req_reg_ubuf"].shape)
            if requant_out_cg % 2 != 0:
                block_dim_tiling[3] = 1

    return block_dim_tiling


def _avoid_complexity_mul(out, tensor_dict, attrs_dict, sch):
    if not tensor_dict["flag_is_dequant_sigmoid_mul"] and \
            not tensor_dict["flag_is_dequant2_sigmoid_mul"] and \
            not tensor_dict["flag_is_dequant_mul"] and not \
            tensor_dict["flag_is_dequant2_mul"]:
        if tensor_dict["flag_is_sigmoid_mul"]:
            sch[tensor_dict["rec_7"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_7"]].compute_inline()
            sch[tensor_dict["rec_6"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_6"]].compute_inline()
            sch[tensor_dict["rec_5"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_5"]].compute_inline()
            sch[tensor_dict["rec_4"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_4"]].compute_inline()
            sch[tensor_dict["rec_3"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_3"]].compute_inline()
            sch[tensor_dict["muls"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["muls"]].compute_inline()
            sch[tensor_dict["exp"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["exp"]].compute_inline()
            sch[tensor_dict["add_2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["add_2"]].compute_inline()
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            attrs_dict["mul_ubuf"] = mul_ubuf
        else:
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            attrs_dict["mul_ubuf"] = mul_ubuf
            if tensor_dict["flag_is_broadcast"]:
                float16_mul_input_ubuf = sch.cache_read(
                    tensor_dict["float16_mul_input_tensor"],
                    cce_params.scope_ubuf, [tensor_dict["broadcast_tensor_0"]])
                sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
            else:
                float16_mul_input_ubuf = sch.cache_read(
                    tensor_dict["float16_mul_input_tensor"],
                    cce_params.scope_ubuf, [attrs_dict["mul_ubuf"]])

            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
    return tensor_dict, attrs_dict, sch


def _l1_fusion_phase1(sch, tensor_dict):
    INPUT_MEM_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    VALID_SHAPE = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    L1_FUSION_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))

    pad_top = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][0])
    pad_right = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][3])
    pad_left = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][2])
    pad_bottom = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][1])

    if VALID_SHAPE:
        tensor_dict["fmap_valid_shape"] = VALID_SHAPE
    if int(INPUT_MEM_TYPE) == 1:
        sch[tensor_dict["fmap"]].set_scope(cce_params.scope_cbuf_fusion)
        a_cbuf_nc1hwc0 = sch.cache_read(tensor_dict["fmap"],
                                        cce_params.scope_cbuf_fusion,
                                        [tensor_dict["im2col_row_major"]])

        if VALID_SHAPE:
            if len(tensor_dict["fmap"].op.shape) == 6:
                sch[a_cbuf_nc1hwc0].buffer_tile(
                    (None, None), (None, None), (None, None),
                    (-pad_top,
                     tensor_dict["fmap"].op.shape[3] + pad_top + pad_bottom),
                    (-pad_left,
                     tensor_dict["fmap"].op.shape[4] + pad_left + pad_right),
                    (None, None))
            else:
                sch[a_cbuf_nc1hwc0].buffer_tile(
                    (None, None), (None, None),
                    (-pad_top,
                     tensor_dict["fmap"].op.shape[2] + pad_top + pad_bottom),
                    (-pad_left,
                     tensor_dict["fmap"].op.shape[3] + pad_left + pad_right),
                    (None, None))
    else:
        # need L1 fusion buffer to storage L1 data
        if L1_FUSION_TYPE == 0 or L1_FUSION_TYPE == 1:
            # DDR in and select
            if VALID_SHAPE:
                sch[tensor_dict["fusion_fmap_select"]].set_scope(
                    cce_params.scope_cbuf_fusion)
                a_cbuf_nc1hwc0 = tensor_dict["fusion_fmap_select"]
            else:
                a_cbuf_nc1hwc0 = sch.cache_read(
                    tensor_dict["fmap"], cce_params.scope_cbuf_fusion,
                    [tensor_dict["im2col_row_major"]])
        else:
            # DDR in and not select, not fusion
            a_cbuf_nc1hwc0 = sch.cache_read(tensor_dict["fmap"],
                                            cce_params.scope_cbuf,
                                            [tensor_dict["im2col_row_major"]])
    return sch, a_cbuf_nc1hwc0


def _save_workspace(tensor_dict, a_cbuf_nc1hwc0, sch):
    INPUT_MEM_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    L1_FUSION_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))
    fmap_l1_addr_flag = int(DepthwiseConv2dParam.fusion_para.get(
        "fmap_l1_addr_flag"))
    fmap_l1_valid_size = int(DepthwiseConv2dParam.fusion_para.get(
        "fmap_l1_valid_size"))
    l1_tensor_map = {}
    if fmap_l1_addr_flag == "nothing":
        l1_tensor_map = None
    else:
        if int(INPUT_MEM_TYPE) in (0, 2) \
                and int(L1_FUSION_TYPE) in (0, 1):
            l1_tensor_map[tensor_dict["fmap"]] = a_cbuf_nc1hwc0
            if fmap_l1_valid_size > 0:
                sch[a_cbuf_nc1hwc0].set_storage_bound(fmap_l1_valid_size)
        else:
            l1_tensor_map = None

    return l1_tensor_map


def _fmp_emit_insn(sch, a_cbuf_nc1hwc0):
    input_mem_type = int(
        DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    l1_fusion_type = int(
        DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))
    valid_shape = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    # no l1fusion
    if l1_fusion_type == -1:
        sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0], 'dma_copy')
    # L1 in, do nothing
    if input_mem_type == 1:
        sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0], 'phony_insn')
    else:
        if valid_shape:
            sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0],
                                          'dma_copy', {"mem_align": 1})
            sch[a_cbuf_nc1hwc0].pragma(a_cbuf_nc1hwc0.op.axis[0], 'jump_data',
                                       1)
        else:
            sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0],
                                          'dma_copy')
            sch[a_cbuf_nc1hwc0].pragma(a_cbuf_nc1hwc0.op.axis[0], 'jump_data',
                                       1)
    return sch

def _set_sch_ph1(out, sch, tensor_dict, attrs_dict, mad_dtype):
    if "addr_type" in out.op.attrs:
        out_addr_type = int(out.op.attrs["addr_type"])
        if out_addr_type == 1:
            sch[out].set_scope(cce_params.scope_cbuf_fusion)
            # when ub fusion, depthwise out may not be final out
            tensor_dict["output_memory_type"] = 1
    if True in [tensor_dict["flag_is_dequant"],
                tensor_dict["flag_is_dequant2"],
                tensor_dict["flag_is_requant"]]:
        sch[tensor_dict["mad_ubuf"]].compute_inline()
    if tensor_dict["bias_flag"]:
        bias_ubuf = sch.cache_read(tensor_dict["bias_tensor"],
                                   cce_params.scope_ubuf,
                                   [tensor_dict["bias_add"]])
        attrs_dict["bias_ubuf"] = bias_ubuf
        sch[bias_ubuf].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["bias_add"]].set_scope(cce_params.scope_ubuf)
    if out.op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        relu_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        attrs_dict["relu_ubuf"] = relu_ubuf
    if out.op.tag == "elewise_binary_mul":
        tensor_dict, attrs_dict, sch = \
            _avoid_complexity_mul(out, tensor_dict, attrs_dict, sch)
    if out.op.tag == "elewise_single_VS_min":
        sch[tensor_dict["max_0"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["max_0"]].compute_inline()
        sch[tensor_dict["depthwise_res"]].compute_inline()
        relu_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        attrs_dict["relu_ubuf"] = relu_ubuf
    c_0 = C0_16
    if mad_dtype == "int32":
        deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch \
            = _set_sch_int32_phase1(tensor_dict, attrs_dict, out, sch)
        attrs_dict["bias_ub"] = bias_ub
        attrs_dict["deq_reg_ubuf"] = deq_reg_ubuf
        attrs_dict["dequant_ubuf"] = dequant_ubuf
        attrs_dict["req_reg_ubuf"] = req_reg_ubuf
        attrs_dict["requant_ubuf"] = requant_ubuf
        c_0 = C0_32
    attrs_dict["c_0"] = c_0
    return sch, tensor_dict, attrs_dict

def depthwise_conv2d_schedule(out):
    is_overload = False
    OFFSET = DepthwiseConv2dParam.fusion_para.get("slice_offset")
    VALID_SHAPE = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    INPUT_MEM_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    OUTPUT_MEM_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("output_memory_type"))
    L1_FUSION_TYPE = int(
        DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))
    L1_VALID_SIZE = int(
        DepthwiseConv2dParam.fusion_para.get("fmap_l1_valid_size"))

    sch = create_schedule(out.op)
    # Prepare tensors.
    tensor_dict = _set_tensor_by_op_tag(out)
    attrs_dict = {}
    attrs_dict["out"] = out
    tensor_dict["input_memory_type"] = INPUT_MEM_TYPE
    tensor_dict["output_memory_type"] = OUTPUT_MEM_TYPE
    tensor_dict["l1_fusion_type"] = L1_FUSION_TYPE
    tensor_dict["fm_l1_valid_size"] = L1_VALID_SIZE
    tensor_dict["fmap_valid_shape"] = None

    # set data flow
    if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
        pre_relu_ubuf = sch.cache_read(tensor_dict["fmap"],
                                       cce_params.scope_ubuf,
                                       [tensor_dict["relu_0"]])
        pre_relu_cbuf = sch.cache_read(tensor_dict["relu_0"],
                                       cce_params.scope_cbuf,
                                       [tensor_dict["im2col_row_major"]])
        sch[tensor_dict["relu_0"]].set_scope(cce_params.scope_ubuf)
        fmp_shape = tensor_dict["fmap"].op.shape
    else:
        sch, a_cbuf_nc1hwc0 = _l1_fusion_phase1(sch, tensor_dict)
        L1CommonParam.l1_fusion_tensors_map = _save_workspace(
            tensor_dict, a_cbuf_nc1hwc0, sch)
        fmp_shape = a_cbuf_nc1hwc0.shape

    a_cbuf_row_major = sch.cache_write(tensor_dict["im2col_row_major"],
                                       cce_params.scope_cbuf)
    sch[tensor_dict["im2col_row_major"]].compute_inline()
    a_ca = sch.cache_write(tensor_dict["im2col_fractal"], cce_params.scope_ca)
    sch[tensor_dict["im2col_fractal"]].compute_inline()

    b_cbuf = sch.cache_read(tensor_dict["filter_buf"], cce_params.scope_cbuf,
                            [tensor_dict["filter_reshape"]])
    b_cb = sch.cache_write(tensor_dict["filter_reshape"], cce_params.scope_cb)
    sch[tensor_dict["filter_reshape"]].compute_inline()

    mad_cc = sch.cache_write(tensor_dict["mad"], cce_params.scope_cc)
    sch[tensor_dict["mad"]].compute_inline()

    mad_dtype = mad_cc.dtype
    sch[tensor_dict["mad_ubuf"]].set_scope(cce_params.scope_ubuf)

    # out to L1
    sch, tensor_dict, attrs_dict = _set_sch_ph1(out, sch, tensor_dict, attrs_dict, mad_dtype)

    if len(fmp_shape) == 6:
        _, _, _, fmap_h, fmap_w, fmap_c0 = (int(i.value) for i in fmp_shape)
    else:
        _, _, fmap_h, fmap_w, fmap_c0 = (int(i.value) for i in fmp_shape)
    pad_top = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][0])
    pad_right = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][3])
    pad_left = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][2])
    pad_bottom = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][1])
    kw = (int)(tensor_dict["mad_ubuf"].op.attrs['kernel_w'])
    kh = (int)(tensor_dict["mad_ubuf"].op.attrs['kernel_h'])
    stride_w = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][1])
    stride_h = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][0])
    howo_one_flag = (tensor_dict["mad_ubuf"].op.attrs['howo_one_flag'])
    if hasattr(howo_one_flag, "value"):
        howo_one_flag = howo_one_flag.value
    wo = (fmap_w + pad_left + pad_right - kw) // stride_w + 1
    ho = (fmap_h + pad_top + pad_bottom - kh) // stride_h + 1
    # get tiling params
    tiling = _get_tiling_fetch(mad_dtype, tensor_dict)
    def _default_tiling():
        tiling = {}
        if tensor_dict["fused_c_dtype"] == "int32":
            dtype = "int8"
        else:
            dtype = "float16"
        mBitLength = {
            "float32": 32,
            "float16": 16,
            "uint8": 8,
            "int8": 8,
            "uint4": 4,
            "int4": 4
        }
        mBitRatio = {
            "int32": 4,
            "float32": 4,
            "float16": 2,
            "uint8": 1,
            "int8": 1,
            "uint4": 1.0 / 2,
            "int4": 1.0 / 2
        }
        wo = (fmap_w + (2 * pad_top) - \
              kw) // stride_w + 1
        gen_m_target = 0
        for m_target in range(32, 0, -1):
            tmp1 = ((m_target * mBitLength['float16']) + wo - 1) // wo
            tmp2 = ((tmp1 * pad_bottom) + \
                    kh) * fmap_w
            MaxFeatureMap = tmp2 * \
                            2 * mBitRatio[dtype]
            if int(MaxFeatureMap) < L1_MEM_LIMIT:
                gen_m_target = m_target
                break

        m = gen_m_target
        tiling["AL1_shape"] = [1, 1, 1, 1]
        tiling["BL1_shape"] = None
        tiling["AL0_matrix"] = [m, 1, 16, 16, 1, 1]
        tiling["BL0_matrix"] = [1, 2, 16, 16, 1, 1]
        tiling["CL0_matrix"] = [2, m, 16, 16, 1, 1]
        tiling["CUB_matrix"] = [2, m, 16, 16, 1, 1]
        tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                            'BL1_pbuffer': 1,
                                            'AL0_pbuffer': 1,
                                            'BL0_pbuffer': 1,
                                            'CL0_pbuffer': 1,
                                            'CUB_pbuffer': 1,
                                            'UBG_pbuffer': 1}
        tiling["AUB_channel_wise_flag"] = None
        tiling["BUB_channel_wise_flag"] = None
        tiling["A_overhead_opt_flag"] = 0
        tiling["B_overhead_opt_flag"] = 0
        tiling["batch_bef_group_flag"] = 0
        tiling["n_bef_batch_flag"] = 0
        tiling["n_bef_group_flag"] = 0
        tiling["block_dim"] = [1, 1, 1, 1]

        return tiling

    def _check_tiling(tiling):
        if tiling["AL0_matrix"][2] == 32 or not isinstance(tiling["AL1_shape"],
                                                           list):
            tiling = _default_tiling()
        return tiling

    tiling = _check_tiling(tiling)

    a_l1_tiling = tiling['AL1_shape']
    b_l1_tiling = tiling['BL1_shape']
    a_l0_tiling = tiling['AL0_matrix']
    b_l0_tiling = tiling['BL0_matrix']
    c_l0_tiling = tiling['CL0_matrix']
    c_ub_tiling = tiling['CUB_matrix']
    block_dim_tiling = tiling['block_dim']
    block_dim_tiling = _dequant_out_cg(mad_dtype, attrs_dict, block_dim_tiling)

    def _tiling_handle(a_l1_tiling, b_l1_tiling, b_l0_tiling, is_overload):
        if block_dim_tiling[1] > 1 or \
                (block_dim_tiling[2] > 1 and (stride_h < kh or stride_w < kw)):
            is_overload = True

        if a_l1_tiling == []:
            a_l1_tiling = [
                fmap_c0 * kw * kh,
                (ho * wo + (c_l0_tiling[1] * TILING_INT8_M) - 1) //
                (c_l0_tiling[1] * TILING_INT8_M), 1
            ]

        if b_l1_tiling == [] or b_l1_tiling is None:
            b_l1_tiling = [fmap_c0 * kw * kh, fmap_c0 // fmap_c0, 1]

        if b_l0_tiling == []:
            b_l0_tiling = [
                a_l0_tiling[1], 1, TILING_INT8_N, TILING_INT8_N, 1, a_l0_tiling[5]
            ]
        return is_overload, a_l1_tiling, b_l1_tiling, b_l0_tiling

    is_overload, a_l1_tiling, b_l1_tiling, b_l0_tiling = _tiling_handle(
        a_l1_tiling, b_l1_tiling, b_l0_tiling, is_overload)

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
    def _db_flag_handle(tiling):
        if "manual_pingpong_buffer" in tiling:
            double_buffer_flag = tiling["manual_pingpong_buffer"]
        return double_buffer_flag

    double_buffer_flag = _db_flag_handle(tiling)
    # L0C
    # batch
    mad_cc_bcut_o, mad_cc_bcut_ii = sch[mad_cc].split(mad_cc.op.axis[0],
                                                      factor=a_l0_tiling[4])

    # m
    mad_cc_mcut_o, mad_cc_mcut_ii = sch[mad_cc].split(mad_cc.op.axis[3],
                                                      factor=a_l0_tiling[0] *
                                                             a_l0_tiling[2])

    # n
    mad_cc_ncut_o, mad_cc_ncut_ii = sch[mad_cc].split(mad_cc.op.axis[2],
                                                      factor=b_l0_tiling[1])

    # k
    mad_cc_kcut_o, mad_cc_kcut_ii = sch[mad_cc].split(mad_cc.op.reduce_axis[0],
                                                      factor=b_l0_tiling[0])

    sch[mad_cc].reorder(mad_cc_bcut_o, mad_cc.op.axis[1], mad_cc_ncut_o,
                        mad_cc_mcut_o, mad_cc_kcut_o, mad_cc_bcut_ii,
                        mad_cc_ncut_ii, mad_cc_mcut_ii, mad_cc.op.axis[4],
                        mad_cc_kcut_ii, mad_cc.op.reduce_axis[1])
    sch[a_ca].compute_at(sch[mad_cc], mad_cc_kcut_o)
    sch = _set_a_cbuf_row_major(mad_dtype, a_cbuf_row_major, wo, sch)
    # batch
    res_cut_dict = {}
    res_bcut_o, res_bcut_i = sch[out].split(out.op.axis[0],
                                            factor=a_l1_tiling[2])
    res_bcut_io, res_bcut_ii = sch[out].split(res_bcut_i,
                                              factor=a_l0_tiling[4])
    res_bcut_iio, res_bcut_iii = sch[out].split(res_bcut_ii,
                                                factor=c_ub_tiling[4])
    res_cut_dict["res_bcut_o"] = res_bcut_o
    res_cut_dict["res_bcut_i"] = res_bcut_i
    res_cut_dict["res_bcut_io"] = res_bcut_io
    res_cut_dict["res_bcut_ii"] = res_bcut_ii
    res_cut_dict["res_bcut_iio"] = res_bcut_iio
    res_cut_dict["res_bcut_iii"] = res_bcut_iii

    if tensor_dict["flag_is_requant"]:
        requant_o, requant_i = sch[tensor_dict["data_transfer"]].split(
            tensor_dict["data_transfer"].op.axis[-1], factor=16)
        sch[tensor_dict["data_transfer"]].reorder(
            tensor_dict["data_transfer"].op.axis[0],
            tensor_dict["data_transfer"].op.axis[1],
            tensor_dict["data_transfer"].op.axis[2], requant_o,
            tensor_dict["data_transfer"].op.axis[-2], requant_i)

    # m
    res_mcut_o, res_mcut_i = sch[out].split(out.op.axis[3],
                                            factor=a_l1_tiling[1] *
                                                   c_l0_tiling[1] * a_l0_tiling[
                                                       2])
    res_mcut_io, res_mcut_ii = sch[out].split(res_mcut_i,
                                              factor=a_l0_tiling[0] *
                                                     a_l0_tiling[2])
    res_mcut_iio, res_mcut_iii = sch[out].split(res_mcut_ii,
                                                factor=c_ub_tiling[1] *
                                                       c_ub_tiling[2])
    res_cut_dict["res_mcut_o"] = res_mcut_o
    res_cut_dict["res_mcut_i"] = res_mcut_i
    res_cut_dict["res_mcut_io"] = res_mcut_io
    res_cut_dict["res_mcut_ii"] = res_mcut_ii
    res_cut_dict["res_mcut_iio"] = res_mcut_iio
    res_cut_dict["res_mcut_iii"] = res_mcut_iii
    # n
    res_ncut_o, res_ncut_i = sch[out].split(out.op.axis[2],
                                            factor=b_l1_tiling[1])
    res_ncut_io, res_ncut_ii = sch[out].split(res_ncut_i,
                                              factor=b_l0_tiling[1])
    res_ncut_iio, res_ncut_iii = sch[out].split(res_ncut_ii,
                                                factor=c_ub_tiling[0])
    res_cut_dict["res_ncut_o"] = res_ncut_o
    res_cut_dict["res_ncut_i"] = res_ncut_i
    res_cut_dict["res_ncut_io"] = res_ncut_io
    res_cut_dict["res_ncut_ii"] = res_ncut_ii
    res_cut_dict["res_ncut_iio"] = res_ncut_iio
    res_cut_dict["res_ncut_iii"] = res_ncut_iii

    sch[out].reorder(out.op.axis[1], res_bcut_o, res_ncut_o, res_mcut_o,
                     res_bcut_io, res_ncut_io, res_mcut_io, res_bcut_iio,
                     res_ncut_iio, res_mcut_iio, res_bcut_iii, res_ncut_iii,
                     res_mcut_iii, out.op.axis[4])
    res_bbcut_o, res_bbcut_i = sch[out].split(res_bcut_o,
                                              nparts=block_dim_tiling[0])
    res_nncut_o, res_nncut_i = sch[out].split(res_ncut_o,
                                              nparts=block_dim_tiling[1])
    res_mmcut_o, res_mmcut_i = sch[out].split(res_mcut_o,
                                              nparts=block_dim_tiling[2])
    res_cccut_o, res_cccut_i = sch[out].split(out.op.axis[1],
                                              nparts=block_dim_tiling[3])
    res_cut_dict["res_bbcut_o"] = res_bbcut_o
    res_cut_dict["res_bbcut_i"] = res_bbcut_i
    res_cut_dict["res_nncut_o"] = res_nncut_o
    res_cut_dict["res_nncut_i"] = res_nncut_i
    res_cut_dict["res_mmcut_o"] = res_mmcut_o
    res_cut_dict["res_mmcut_i"] = res_mmcut_i
    res_cut_dict["res_cccut_o"] = res_cccut_o
    res_cut_dict["res_cccut_i"] = res_cccut_i
    sch[out].reorder(res_cccut_o, res_bbcut_o, res_nncut_o, res_mmcut_o,
                     res_cccut_i, res_bbcut_i, res_nncut_i, res_mmcut_i)
    blocks = block_dim_tiling[0] * block_dim_tiling[1] * \
             block_dim_tiling[2] * block_dim_tiling[3]

    batch_cout_fused = sch[out].fuse(res_cccut_o, res_bbcut_o, res_nncut_o,
                                     res_mmcut_o)
    noo_true, _ = sch[out].split(batch_cout_fused, nparts=blocks)
    block = tvm.thread_axis("blockIdx.x")
    sch[out].bind(noo_true, block)
    sch[b_cbuf].compute_at(sch[out], res_cccut_i)

    def _spe_handle():
        if tiling['BL0_matrix'] == [] and howo_one_flag:
            sch[b_cb].compute_at(sch[out], res_cccut_i)
        else:
            sch[b_cb].compute_at(sch[mad_cc], mad_cc_kcut_o)

        if out.op.tag == "elewise_single_VS_min":
            sch[tensor_dict["bias_add"]].mem_unique()
            sch[tensor_dict["max_0"]].mem_unique()
            sch[attrs_dict["relu_ubuf"]].mem_unique()

        if tiling['BL1_shape'] is None:
            sch[b_cbuf].compute_inline()
        if tensor_dict["bias_flag"]:
            sch[attrs_dict["bias_ubuf"]].compute_at(sch[out], res_cccut_i)
            sch[tensor_dict["bias_add"]].compute_at(sch[out], res_mcut_iio)
        if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
            sch[pre_relu_ubuf].compute_at(sch[out], res_mmcut_i)
            sch[tensor_dict["relu_0"]].compute_at(sch[out], res_mmcut_i)
            sch[pre_relu_cbuf].compute_at(sch[out], res_mmcut_i)
        else:
            sch[a_cbuf_nc1hwc0].compute_at(sch[out], res_mmcut_i)
        sch[a_cbuf_row_major].compute_at(sch[out], res_mmcut_i)

        sch[mad_cc].compute_at(sch[out], res_mcut_io)
        return sch

    def _int32_spe_handle():
        if mad_dtype == "int32":
            if double_buffer_flag["AL0_pbuffer"] == 2:
                sch[a_cbuf_row_major].double_buffer()
            if "fmap" in tensor_dict and tensor_dict["fmap"].dtype == out.dtype:
                if double_buffer_flag["BL1_pbuffer"] == 2:
                    sch[b_cbuf].preload()
                if double_buffer_flag["AL0_pbuffer"] == 2:
                    sch[a_cbuf_row_major].preload()
                if "relu" not in tensor_dict["im2col_row_major"].op.input_tensors[
                    0].name:
                    if double_buffer_flag["AL1_pbuffer"] == 2:
                        sch[a_cbuf_nc1hwc0].preload()
            if tensor_dict["flag_is_quant_relu6_dequant"] or tensor_dict[
                "flag_is_dequant_sigmoid_mul"] or tensor_dict[
                "flag_is_dequant2_sigmoid_mul"]:
                sch[attrs_dict["deq_reg_ubuf"]].double_buffer()
                sch[attrs_dict["deq_reg_ubuf"]].preload()
                sch[attrs_dict["bias_ub"]].double_buffer()
                sch[attrs_dict["bias_ub"]].preload()
        return sch

    sch = _spe_handle()
    sch = _int32_spe_handle()

    if [tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"],
        tensor_dict["flag_is_dequant_quant"], tensor_dict["flag_is_requant"],
        tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant2_mul"],
        tensor_dict["flag_is_dequant_sigmoid_mul"],
        tensor_dict["flag_is_dequant2_sigmoid_mul"]] == [False] * 8:
        sch[tensor_dict["mad_ubuf"]].compute_at(sch[out], res_mcut_iio)
        sch[tensor_dict["mad_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                  (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))
    a2_axis, a3_axis, sch = _set_sch_int32_phase2(mad_dtype,
                                                  double_buffer_flag,
                                                  tensor_dict, attrs_dict,
                                                  res_cut_dict, sch)
    attrs_dict["a2_axis"] = a2_axis
    attrs_dict["a3_axis"] = a3_axis
    def _relu_mul_handle():

        if out.op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
            sch[attrs_dict["relu_ubuf"]].compute_at(sch[out], res_mcut_iio)
            sch[attrs_dict["relu_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]),
                                        (1, BLOCK_SIZE))
        if out.op.tag == "elewise_binary_mul":
            if not tensor_dict["flag_is_dequant_sigmoid_mul"] and not tensor_dict[
                "flag_is_dequant2_sigmoid_mul"]:
                if tensor_dict["flag_is_sigmoid_mul"]:
                    sch[tensor_dict["rec_7"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["rec_6"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["rec_5"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["rec_4"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["rec_3"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["muls"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["add_2"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                    sch[tensor_dict["exp"]].compute_at(
                        sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                else:
                    sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
                        sch[out], res_mcut_iio)
                sch[attrs_dict["mul_ubuf"]].compute_at(sch[out], res_mcut_iio)

                sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                         (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))

            elif not tensor_dict["flag_is_dequant_mul"] and not tensor_dict[
                "flag_is_dequant2_mul"]:
                sch[attrs_dict["mul_ubuf"]].compute_at(sch[out], res_mcut_iio)
                sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(
                    sch[out], res_mcut_iio)
                sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1),
                                                         (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))
        if out.op.tag == "elewise_single_VS_min":
            sch[tensor_dict["max_0"]].compute_at(sch[out], res_mcut_iio)
            sch[attrs_dict["relu_ubuf"]].compute_at(sch[out], res_mcut_iio)
            sch[attrs_dict["relu_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]),
                                        (1, BLOCK_SIZE))
        return sch

    sch = _relu_mul_handle()

    def _set_double_buffer(input_module, sch_to_deal):
        if input_module == 2:
            sch_to_deal.double_buffer()

    # al1
    if "relu" not in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
        _set_double_buffer(double_buffer_flag["AL1_pbuffer"],
                           sch[a_cbuf_nc1hwc0])
    _set_double_buffer(double_buffer_flag["BL1_pbuffer"], sch[b_cbuf])
    _set_double_buffer(double_buffer_flag["AL0_pbuffer"], sch[a_ca])
    _set_double_buffer(double_buffer_flag["BL0_pbuffer"], sch[b_cb])
    if [tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"],
        tensor_dict["flag_is_dequant_quant"], tensor_dict["flag_is_requant"],
        tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant2_mul"],
        tensor_dict["flag_is_dequant_sigmoid_mul"],
        tensor_dict["flag_is_dequant2_sigmoid_mul"]] == [False] * 8:
        _set_double_buffer(double_buffer_flag["CUB_pbuffer"],
                           sch[tensor_dict["mad_ubuf"]])
    setfmatrix_dict = {
        "conv_kernel_h": tensor_dict["mad_ubuf"].op.attrs['kernel_h'],
        "conv_kernel_w": tensor_dict["mad_ubuf"].op.attrs['kernel_w'],
        "conv_padding_top": tensor_dict["mad_ubuf"].op.attrs['padding'][0],
        "conv_padding_bottom": tensor_dict["mad_ubuf"].op.attrs['padding'][1],
        "conv_padding_left": tensor_dict["mad_ubuf"].op.attrs['padding'][2],
        "conv_padding_right": tensor_dict["mad_ubuf"].op.attrs['padding'][3],
        "conv_stride_h": tensor_dict["mad_ubuf"].op.attrs['stride'][0],
        "conv_stride_w": tensor_dict["mad_ubuf"].op.attrs['stride'][1],
        "conv_fm_c": fmap_c0,
        "conv_fm_h": fmap_h,
        "conv_fm_w": fmap_w
    }

    def _valid_shape_handle():
        if VALID_SHAPE:
            setfmatrix_dict["conv_fm_h"] = VALID_SHAPE[2]
            if INPUT_MEM_TYPE == 1:
                setfmatrix_dict["conv_fm_offset_h"] = OFFSET[2]
        return setfmatrix_dict

    setfmatrix_dict = _valid_shape_handle()

    # emit insn
    def _bias_relu_emit_insn(sch):
        if tensor_dict["bias_flag"]:
            sch[attrs_dict["bias_ubuf"]].emit_insn(sch[attrs_dict["bias_ubuf"]].op.axis[0], 'dma_copy')
            sch[tensor_dict["bias_add"]].emit_insn(
                sch[tensor_dict["bias_add"]].op.axis[0], 'vector_auto')

        if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
            sch[tensor_dict["relu_0"]].emit_insn(tensor_dict["relu_0"].op.axis[0],
                                                 'vector_auto')
            sch[pre_relu_ubuf].emit_insn(pre_relu_ubuf.op.axis[0], 'dma_copy')
            sch[pre_relu_cbuf].emit_insn(pre_relu_cbuf.op.axis[0], 'dma_copy')
        else:
            sch = _fmp_emit_insn(sch, a_cbuf_nc1hwc0)
        return sch

    sch = _bias_relu_emit_insn(sch)

    sch[a_cbuf_row_major].emit_insn(a_cbuf_row_major.op.axis[1], 'set_fmatrix',
                                    setfmatrix_dict)
    sch[a_ca].emit_insn(a_ca.op.axis[1], 'im2col')
    sch[b_cbuf].emit_insn(b_cbuf.op.axis[0], 'dma_copy')
    sch[b_cb].emit_insn(b_cb.op.axis[0], 'dma_copy')
    mad_dict = {
        "mad_pattern": cce_params.CONV_MODE,
        "k_outer": [mad_cc_kcut_o]
    }
    if ((True in [tensor_dict["flag_is_dequant2"],
                  tensor_dict["flag_is_dequant"],
                  tensor_dict["flag_is_requant"],
                  tensor_dict["flag_is_dequant_mul"],
                  tensor_dict["flag_is_dequant_sigmoid_mul"],
                  tensor_dict["flag_is_dequant2_mul"],
                  tensor_dict["flag_is_dequant2_sigmoid_mul"]] and
         tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1) or
            tensor_dict["flag_is_dequant_bias"]):
        mad_dict["init_bias"] = 1
        sch[tensor_dict["mad_bias"]].reused_by(tensor_dict["mad_after_bias"],
                                               mad_cc)
        if double_buffer_flag["CL0_pbuffer"] == 2:
            sch[tensor_dict["mad_bias"]].double_buffer()
            sch[mad_cc].double_buffer()
            sch[tensor_dict["mad_after_bias"]].double_buffer()
    sch[mad_cc].emit_insn(mad_cc_bcut_ii, 'mad', mad_dict)
    if [tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"],
        tensor_dict["flag_is_dequant_quant"], tensor_dict["flag_is_requant"],
        tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant2_mul"],
        tensor_dict["flag_is_dequant_sigmoid_mul"],
        tensor_dict["flag_is_dequant2_sigmoid_mul"]] == [False] * 8:
        sch[tensor_dict["mad_ubuf"]].emit_insn(
            sch[tensor_dict["mad_ubuf"]].op.axis[0], 'dma_copy')
    attrs_dict["out"] = out
    attrs_dict["mad_dtype"] = mad_dtype
    sch = _set_sch_int32_phase3(tensor_dict, sch, attrs_dict, res_cut_dict,
                                out)

    set_pragma_for_cache_read_mode(is_overload, sch[out], res_mmcut_i)

    return sch


def pragma_overload_filter(condition0, condition1, stage, first_axis):
    is_overload = False

    if condition0 > 1 or condition1 > 1:
        is_overload = True

    set_pragma_for_cache_read_mode(is_overload, stage, first_axis)


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=too-many-branches


def depthwise_conv2d_backprop_filter_d_schedule(depthwise_dfilter_res):
    """
    schedule of depthwise conv2d backprop filter

    dout->dout_transpose->dout_fractal--|
                                         |->mad_res->depthwise_dfilter
    fmap->fmap_transpose->feature_col--  /                   |
    ----->feature_col_pad---------------/          depthwise_dfilter_res
    reduce : K=NHoWo M=16 N=HwWw*16

    Parameters
    ----------
    depthwise_dfilter_res : tvm tensor
        output tensor(dfilter) in tvm.

    Returns
    -------
    s: tvm schedule
        the tensor of output.
    """
    s = create_schedule(depthwise_dfilter_res.op)

    # get tensors form input
    depthwise_dfilter = depthwise_dfilter_res.op.input_tensors[0]
    mad_res = depthwise_dfilter.op.input_tensors[0]
    dout_fractal = mad_res.op.input_tensors[0]
    dout_transpose = dout_fractal.op.input_tensors[0]
    dout = dout_transpose.op.input_tensors[0]
    feature_col_pad = mad_res.op.input_tensors[1]
    feature_col = feature_col_pad.op.input_tensors[0]
    fmap_transpose = feature_col.op.input_tensors[0]
    fmap = fmap_transpose.op.input_tensors[0]

    # define stages
    fmap_cbuf_nc1hwc0 = s.cache_write(fmap_transpose, cce_params.scope_cbuf)
    s[fmap_transpose].compute_inline()
    fmap_cbuf_row_major = s.cache_write(feature_col, cce_params.scope_cbuf)
    s[feature_col].compute_inline()
    fmap_cb = s.cache_write(feature_col_pad, cce_params.scope_cb)
    s[feature_col_pad].compute_inline()
    dout_cbuf = s.cache_write(dout_transpose, cce_params.scope_cbuf)
    s[dout_transpose].compute_inline()
    dout_ca = s.cache_write(dout_fractal, cce_params.scope_ca)
    s[dout_fractal].compute_inline()
    mad_ubuf = s.cache_write(mad_res, cce_params.scope_ubuf)
    s[mad_res].compute_inline()
    mad_cc = s.cache_write(mad_ubuf, cce_params.scope_cc)
    depthwise_dfilter_ubuf = s.cache_write(depthwise_dfilter,
                                           cce_params.scope_ubuf)
    s[depthwise_dfilter].compute_inline()

    # get shape values
    fmap_shape = [int(i.value) for i in fmap.shape]
    dout_shape = [int(i.value) for i in dout.shape]

    # NCgC1HiWiC0 (C1 = 1)
    fmap_n, fmap_cg, _, fmap_h, fmap_w, fmap_c0 = fmap_shape
    # NCgC1HoWoC0 (C1 = 1)
    dout_n, dout_cg, dout_c1, dout_h, dout_w, dout_c0 = dout_shape
    filter_h = depthwise_dfilter_res.op.attrs['kernel_h'].value
    filter_w = depthwise_dfilter_res.op.attrs['kernel_w'].value
    stride_h = int(depthwise_dfilter_res.op.attrs['stride'][0].value)
    stride_w = int(depthwise_dfilter_res.op.attrs['stride'][1].value)
    dilation_h = int(depthwise_dfilter_res.op.attrs['dilations'][0].value)
    dilation_w = int(depthwise_dfilter_res.op.attrs['dilations'][1].value)
    pad_top = int(depthwise_dfilter_res.op.attrs['padding'][0].value)
    pad_bottom = int(depthwise_dfilter_res.op.attrs['padding'][1].value)
    pad_left = int(depthwise_dfilter_res.op.attrs['padding'][2].value)
    pad_right = int(depthwise_dfilter_res.op.attrs['padding'][3].value)
    kernel_name = depthwise_dfilter_res.op.attrs['kernel_name']

    dout_shape_5hd = (dout_n, dout_cg, dout_h, dout_w, dout_c0)
    fmap_shape_5hd = (fmap_n, fmap_cg, fmap_h, fmap_w, fmap_c0)
    filter_5hd = (BLOCK_SIZE, dout_c1, filter_h, filter_w, BLOCK_SIZE)
    dout_shape_5hd = list(map(int, dout_shape_5hd))
    fmap_shape_5hd = list(map(int, fmap_shape_5hd))
    filter_5hd = list(map(int, filter_5hd))
    tiling = tiling_query(a_shape=dout_shape_5hd,
                          b_shape=fmap_shape_5hd,
                          c_shape=filter_5hd,
                          a_dtype=dout.dtype,
                          b_dtype=fmap.dtype,
                          c_dtype=depthwise_dfilter_res.dtype,
                          mad_dtype="float32",
                          padl=pad_left,
                          padr=pad_right,
                          padu=pad_top,
                          padd=pad_bottom,
                          strideh=stride_h,
                          stridew=stride_w,
                          strideh_expand=1,
                          stridew_expand=1,
                          dilationh=dilation_h,
                          dilationw=dilation_w,
                          group=1,
                          fused_double_operand_num=0,
                          bias_flag=0,
                          op_tag="depthwise_bp_filter",
                          kernel_name=kernel_name)
    _common_tiling_check(tiling)
    if tiling["BL1_shape"] is None:
        raise RuntimeError("BL1_shape can not be None in"
                           " depthwise_conv2d_backprop_filter")

    # Cg,C1HwWw,C1C0,C0 (zN in L0C, Cg, N1, M, N0)
    mad_cc_axis_cg, mad_cc_axis_n1, mad_cc_axis_m, \
    mad_cc_axis_n0 = mad_cc.op.axis
    n1_l0_factor = tiling["CL0_matrix"][0]  # tiling["BL0_matrix"][1]
    m0_l0_factor = tiling["CL0_matrix"][2]
    m1_l0_factor = tiling["CL0_matrix"][1]  # or tiling["AL0_matrix"][0]
    al1_shape_invalid = tiling["AL1_shape"] == [] or tiling["AL1_shape"] is None
    if al1_shape_invalid:
        m_l1_factor = BLOCK_SIZE
        k_hw_al1_factor = dout_h * dout_w
    else:
        m_l1_factor = tiling["AL1_shape"][1] * m1_l0_factor * m0_l0_factor
        k_hw_al1_factor = tiling["AL1_shape"][0]

    if not tiling["BL1_shape"]:
        n1_l1_factor = filter_h * filter_w * BLOCK_SIZE
        k_hw_bl1_factor = dout_h * dout_w
    else:
        n1_l1_factor = tiling["BL1_shape"][1] * n1_l0_factor
        k_hw_bl1_factor = tiling["BL1_shape"][0]

    if tiling["AL0_matrix"] == [] or tiling["AL0_matrix"] is None:
        k_hw_k0_factor = BLOCK_SIZE
        k_hw_l0_factor = (dout_h * dout_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        k_hw_k0_factor = tiling["AL0_matrix"][3]  # or tiling["BL0_matrix"][3]
        k_hw_l0_factor = tiling["AL0_matrix"][1]  # or tiling["BL0_matrix"][0]
    block_batch_nparts = tiling["block_dim"][0]
    block_n_nparts = tiling["block_dim"][1]
    block_m_nparts = tiling["block_dim"][2]
    block_cg_nparts = tiling["block_dim"][3]
    if "AUB_shape" not in tiling.keys() \
            or tiling["AUB_shape"] == [] or tiling["AUB_shape"] is None:
        block_h_nparts = 1
    else:
        block_h_nparts = tiling["AUB_shape"][0]
    if tiling["AL1_shape"] is None:
        s[dout_cbuf].compute_inline()
    # N
    mad_cc_n1_l1o, mad_cc_n1_l1i = s[mad_cc].split(mad_cc_axis_n1,
                                                   n1_l1_factor)
    mad_cc_n1_l0o, mad_cc_n1_l0i = s[mad_cc].split(mad_cc_n1_l1i, n1_l0_factor)
    # M
    mad_cc_m1_l1o, mad_cc_m_l1i = s[mad_cc].split(mad_cc_axis_m, m_l1_factor)
    mad_cc_m1, mad_cc_m0 = s[mad_cc].split(mad_cc_m_l1i, m0_l0_factor)
    mad_cc_m1_l0o, mad_cc_m1_l0i = s[mad_cc].split(mad_cc_m1, m1_l0_factor)
    # K
    mad_cc_kn_l1o = mad_cc.op.reduce_axis[0]
    block_batch_o, block_batch_i = s[mad_cc].split(mad_cc_kn_l1o,
                                                   nparts=block_batch_nparts)
    mad_cc_axis_k = mad_cc.op.reduce_axis[1]
    block_h_o, block_h_i = s[mad_cc].split(mad_cc_axis_k,
                                           nparts=block_h_nparts)

    if k_hw_al1_factor >= k_hw_bl1_factor:
        mad_cc_ak1_l1o, mad_cc_ak1_l1i = s[mad_cc].split(
            block_h_i, k_hw_al1_factor)
        mad_cc_bk1_l1o, mad_cc_k1_l1i = s[mad_cc].split(
            mad_cc_ak1_l1i, k_hw_bl1_factor)
        mad_cc_max_k1_l1o = mad_cc_ak1_l1o
        mad_cc_min_k1_l1o = mad_cc_bk1_l1o
    else:
        mad_cc_bk1_l1o, mad_cc_bk1_l1i = s[mad_cc].split(
            block_h_i, k_hw_bl1_factor)
        mad_cc_ak1_l1o, mad_cc_k1_l1i = s[mad_cc].split(
            mad_cc_bk1_l1i, k_hw_al1_factor)
        mad_cc_max_k1_l1o = mad_cc_bk1_l1o
        mad_cc_min_k1_l1o = mad_cc_ak1_l1o

    mad_cc_k1, mad_cc_k0 = s[mad_cc].split(mad_cc_k1_l1i, k_hw_k0_factor)
    mad_cc_k1_l0o, mad_cc_k1_l0i = s[mad_cc].split(mad_cc_k1, k_hw_l0_factor)

    s[mad_cc].reorder(mad_cc_axis_cg, block_batch_o, block_h_o, mad_cc_n1_l1o,
                      mad_cc_m1_l1o, block_batch_i, mad_cc_max_k1_l1o,
                      mad_cc_min_k1_l1o, mad_cc_n1_l0o, mad_cc_m1_l0o,
                      mad_cc_k1_l0o, mad_cc_m1_l0i, mad_cc_n1_l0i,
                      mad_cc_axis_n0, mad_cc_m0, mad_cc_k1_l0i, mad_cc_k0)
    s[dout_ca].compute_at(s[mad_cc], mad_cc_k1_l0o)
    s[fmap_cb].compute_at(s[mad_cc], mad_cc_k1_l0o)
    s[dout_cbuf].compute_at(s[mad_cc], mad_cc_ak1_l1o)
    s[fmap_cbuf_nc1hwc0].compute_at(s[mad_cc], mad_cc_bk1_l1o)
    s[fmap_cbuf_row_major].compute_at(s[mad_cc], mad_cc_bk1_l1o)

    dw_axis_cg, dw_axis_n1, dw_axis_m, dw_axis_n0 = s[
        depthwise_dfilter_res].op.axis
    n1_ub_factor = tiling["CUB_matrix"][0]
    m1_ub_factor = tiling["CUB_matrix"][1]
    # Block tiling
    block_cg_o, block_cg_i = s[depthwise_dfilter_res].split(
        dw_axis_cg, nparts=block_cg_nparts)
    block_n_o, block_n_i = s[depthwise_dfilter_res].split(
        dw_axis_n1, nparts=block_n_nparts)
    block_m_o, block_m_i = s[depthwise_dfilter_res].split(
        dw_axis_m, nparts=block_m_nparts)

    # N
    dw_n1_l0o, dw_n1_l0i = s[depthwise_dfilter_res].split(
        block_n_i, n1_l0_factor)
    dw_n1_ubo, dw_n1_ubi = s[depthwise_dfilter_res].split(
        dw_n1_l0i, n1_ub_factor)

    pragma_overload_filter(block_n_nparts, block_m_nparts,
                           s[depthwise_dfilter_res], dw_n1_ubi)

    # M
    dw_m1, dw_m0 = s[depthwise_dfilter_res].split(block_m_i, m0_l0_factor)
    dw_m1_l0o, dw_m1_l0i = s[depthwise_dfilter_res].split(dw_m1, m1_l0_factor)
    dw_m1_ubo, dw_m1_ubi = s[depthwise_dfilter_res].split(
        dw_m1_l0i, m1_ub_factor)
    s[depthwise_dfilter_res].reorder(block_cg_o, block_m_o, block_n_o,
                                     block_cg_i, dw_n1_l0o, dw_m1_l0o,
                                     dw_n1_ubo, dw_m1_ubo, dw_n1_ubi,
                                     dw_m1_ubi, dw_axis_n0, dw_m0)
    s[mad_cc].compute_at(s[depthwise_dfilter_res], dw_m1_l0o)
    s[mad_ubuf].compute_at(s[depthwise_dfilter_res], dw_m1_ubo)
    s[depthwise_dfilter_ubuf].compute_at(s[depthwise_dfilter_res], dw_m1_ubo)

    s[dout_cbuf].storage_align(s[dout_cbuf].op.axis[2], BLOCK_SIZE, 0)
    s[dout_ca].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE),
                            (1, BLOCK_SIZE))
    s[fmap_cb].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE),
                            (1, BLOCK_SIZE))
    s[fmap_cbuf_row_major].buffer_align((1, 1), (1, 1), (dout_w, dout_w),
                                        (1, 1), (filter_h, filter_h),
                                        (filter_w, filter_w), (1, BLOCK_SIZE))

    if tiling["manual_pingpong_buffer"]["AL1_pbuffer"] == DOUBLE_BUFFER:
        s[dout_cbuf].double_buffer()
    if tiling["manual_pingpong_buffer"]["BL1_pbuffer"] == DOUBLE_BUFFER:
        s[fmap_cbuf_nc1hwc0].double_buffer()
        s[fmap_cbuf_row_major].double_buffer()
    if tiling["manual_pingpong_buffer"]["AL0_pbuffer"] == DOUBLE_BUFFER:
        s[dout_ca].double_buffer()
    if tiling["manual_pingpong_buffer"]["BL0_pbuffer"] == DOUBLE_BUFFER:
        s[fmap_cb].double_buffer()
    if tiling["manual_pingpong_buffer"]["CL0_pbuffer"] == DOUBLE_BUFFER:
        s[mad_cc].double_buffer()
    if tiling["manual_pingpong_buffer"]["CUB_pbuffer"] == DOUBLE_BUFFER:
        s[mad_ubuf].double_buffer()
        s[depthwise_dfilter_ubuf].double_buffer()

    s[mad_ubuf].reused_by(depthwise_dfilter_ubuf)

    # emit insn
    s[fmap_cbuf_nc1hwc0].emit_insn(fmap_cbuf_nc1hwc0.op.axis[0], 'dma_copy')
    # emit convolution params.
    setfmatrix_dict = {
        "conv_kernel_h": depthwise_dfilter_res.op.attrs['kernel_h'],
        "conv_kernel_w": depthwise_dfilter_res.op.attrs['kernel_w'],
        "conv_padding_top": depthwise_dfilter_res.op.attrs['padding'][0],
        "conv_padding_bottom": depthwise_dfilter_res.op.attrs['padding'][1],
        "conv_padding_left": depthwise_dfilter_res.op.attrs['padding'][2],
        "conv_padding_right": depthwise_dfilter_res.op.attrs['padding'][3],
        "conv_stride_h": depthwise_dfilter_res.op.attrs['stride'][0],
        "conv_stride_w": depthwise_dfilter_res.op.attrs['stride'][1],
        "conv_fm_c": fmap.op.shape[2] * fmap.op.shape[5],
        "conv_fm_h": fmap.op.shape[3],
        "conv_fm_w": fmap.op.shape[4]
    }
    s[fmap_cbuf_row_major].emit_insn(fmap_cbuf_row_major.op.axis[1],
                                     'set_fmatrix', setfmatrix_dict)
    s[fmap_cb].emit_insn(fmap_cb.op.axis[1], 'im2col')
    s[dout_cbuf].emit_insn(dout_cbuf.op.axis[0], 'dma_copy')
    s[dout_ca].emit_insn(dout_ca.op.axis[0], 'dma_copy')

    s[mad_ubuf].emit_insn(mad_ubuf.op.axis[0], 'dma_copy')
    # mad_pattern value: 0 for gemm, 1 for gemv, 2 for convolution
    mad_dict = {
        "mad_pattern":
            cce_params.CONV_MODE,
        'k_outer': [
            block_batch_o, block_batch_i, block_h_o, mad_cc_ak1_l1o,
            mad_cc_bk1_l1o, mad_cc_k1_l0o
        ]
    }
    s[mad_cc].emit_insn(mad_cc_m1_l0i, 'mad', mad_dict)
    s[depthwise_dfilter_ubuf].reorder(depthwise_dfilter_ubuf.op.axis[0],
                                      depthwise_dfilter_ubuf.op.axis[2],
                                      depthwise_dfilter_ubuf.op.axis[1],
                                      depthwise_dfilter_ubuf.op.axis[3])
    s[depthwise_dfilter_ubuf].emit_insn(depthwise_dfilter_ubuf.op.axis[1],
                                        'elewise_single_diagonal')
    s[depthwise_dfilter_res].emit_insn(dw_n1_ubi, 'dma_copy')

    # for multi cores
    block = tvm.thread_axis("blockIdx.x")
    block_axis = s[depthwise_dfilter_res].fuse(block_cg_o, block_m_o,
                                               block_n_o)
    s[depthwise_dfilter_res].bind(block_axis, block)

    return s


# pylint: disable=locally-disabled,too-many-locals
def depthwise_conv2d_backprop_input_d_schedule(dx_res):
    """
    the schedule of depthwise_conv2d_backprop_input
    """
    s = create_schedule(dx_res.op)

    # get tensor info
    dx_cast = dx_res.op.input_tensors[0]
    mad_res = dx_cast.op.input_tensors[0]
    dout_col_pad = mad_res.op.input_tensors[0]
    weight_rotated = mad_res.op.input_tensors[1]
    weight = weight_rotated.op.input_tensors[0]
    dout_col = dout_col_pad.op.input_tensors[0]
    dout_dilated = dout_col.op.input_tensors[0]
    dout = dout_dilated.op.input_tensors[0]

    # set data flow
    dout_ubuf = s.cache_read(dout, cce_params.scope_ubuf, [dout_dilated])
    dout_cbuf_nc1hwc0 = s.cache_write(dout_dilated, cce_params.scope_cbuf)
    dout_dilated_ubuf = s.cache_write(dout_cbuf_nc1hwc0, cce_params.scope_ubuf)
    dout_cbuf_row_major = s.cache_write(dout_col, cce_params.scope_cbuf)
    s[dout_dilated].compute_inline()
    s[dout_col].compute_inline()
    dout_ca = s.cache_write(dout_col_pad, cce_params.scope_ca)
    s[dout_col_pad].compute_inline()

    weight_cbuf = s.cache_read(weight, cce_params.scope_cbuf, [weight_rotated])
    weight_cb = s.cache_write(weight_rotated, cce_params.scope_cb)
    s[weight_rotated].compute_inline()

    mad_cc = s.cache_write(mad_res, cce_params.scope_cc)
    mad_ubuf = s.cache_write(dx_cast, cce_params.scope_ubuf)
    s[mad_res].compute_inline()
    s[dx_cast].compute_inline()

    # compute shape value, out input img2col_padding
    block_size = dout.op.shape[len(dout.op.shape) - 1].value
    _, _, _, _, dout_dilated_w, _ = dout_dilated.shape
    fmap_w = dout_dilated_w.value + dx_res.op.attrs['dilated_pad'][
        2].value + dx_res.op.attrs['dilated_pad'][3].value - dx_res.op.attrs[
                 'weight_width'].value + 1
    stride = int(dout_dilated.op.attrs["strides"][0].value)

    # get shape value
    weight_shape = [int(i.value) for i in weight.shape]
    dout_shape = [int(i.value) for i in dout.shape]
    dout_dilated_shape = [int(i.value) for i in dout_dilated.shape]
    dout_col_shape = [int(i.value) for i in dout_col.shape]
    weight_height = dout_col_shape[4]

    def _tiling_fetch():
        """"tiling_fetch"""
        padding_top = int(dx_res.op.attrs['dilated_pad'][0])
        padding_bottom = int(dx_res.op.attrs['dilated_pad'][1])
        padding_left = int(dx_res.op.attrs['dilated_pad'][2])
        padding_right = int(dx_res.op.attrs['dilated_pad'][3])
        # after expand, full model sliding window, value must be 1
        strideH = int(dx_res.op.attrs['dilated_strides'][0])
        strideW = int(dx_res.op.attrs['dilated_strides'][1])
        kernel_h = int(dx_res.op.attrs['weight_height'])
        kernel_w = int(dx_res.op.attrs['weight_width'])
        kernel_name = dx_res.op.attrs['kernel_name']
        # expand stride equal ops interface parameter
        strideH_expand = stride
        strideW_expand = stride
        dilationH = 1
        dilationW = 1

        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = mad_cc.dtype
        mad_dtype = mad_cc.dtype
        groupNum = 1

        dout_shape_batch, dout_shape_output_c1, _, dout_shape_output_height, \
        dout_shape_output_width, dout_shape_block = dout_shape
        dout_shape_tiling = dout_shape_batch, dout_shape_output_c1, \
                            dout_shape_output_height, \
                            dout_shape_output_width, dout_shape_block
        weight_shape_C1, _, _, weight_shape_Co, \
        weight_shape_block = weight_shape
        weight_shape_tiling = weight_shape_Co, weight_shape_C1, \
                              kernel_h, kernel_w, weight_shape_block
        padding_top_tiling = (padding_top + abs(padding_top)) // 2
        padding_bottom_tiling = (padding_bottom + abs(padding_bottom)) // 2
        padding_left_tiling = (padding_left + abs(padding_left)) // 2
        padding_right_tiling = (padding_right + abs(padding_right)) // 2

        wd = dout_shape_output_width * stride - (stride - 1)
        hd = dout_shape_output_height * stride - (stride - 1)
        wi = (wd + padding_left + padding_right - kernel_w) // strideW + 1
        hi = (hd + padding_top + padding_bottom - kernel_h) // strideH + 1
        # shape format must be 5HD
        dout_shape_tiling = list(map(int, dout_shape_tiling))
        weight_shape_tiling = list(map(int, weight_shape_tiling))
        tiling_new = tiling_query(a_shape=dout_shape_tiling,
                                  b_shape=weight_shape_tiling,
                                  c_shape=None,
                                  a_dtype=in_dtype,
                                  b_dtype=w_dtype,
                                  c_dtype=res_dtype,
                                  mad_dtype=mad_dtype,
                                  padl=padding_left_tiling,
                                  padr=padding_right_tiling,
                                  padu=padding_top_tiling,
                                  padd=padding_bottom_tiling,
                                  strideh=strideH,
                                  stridew=strideW,
                                  strideh_expand=strideH_expand,
                                  stridew_expand=strideW_expand,
                                  dilationh=dilationH,
                                  dilationw=dilationW,
                                  group=groupNum,
                                  fused_double_operand_num=0,
                                  bias_flag=0,
                                  op_tag="depthwise_bp_input",
                                  kernel_name=kernel_name)

        # get tiling params
        AL1_tiling = tiling_new['AL1_shape']
        BL1_tiling = tiling_new['BL1_shape']
        AL0_tiling = tiling_new['AL0_matrix']
        BL0_tiling = tiling_new['BL0_matrix']
        CL0_tiling = tiling_new['CL0_matrix']
        CUB_tiling = tiling_new['CUB_matrix']
        AUB_tiling = tiling_new['AUB_shape']
        DOUBLE_BUFFER_tiling = tiling_new["manual_pingpong_buffer"]
        block_dim_tiling = tiling_new['block_dim']

        if AL1_tiling == []:
            AL1_tiling = [
                dout_shape_block * kernel_w * kernel_h,
                (hi * wi + (CL0_tiling[1] * 16) - 1) // (CL0_tiling[1] * 16),
                1, 1
            ]
        if BL1_tiling == [] or BL1_tiling is None:
            BL1_tiling = [
                dout_shape_block * kernel_w * kernel_h,
                dout_shape_block // dout_shape_block, 1, 1
            ]
        if BL0_tiling == []:
            BL0_tiling = [AL0_tiling[1], 1, 16, 16, 1, AL0_tiling[5]]
        return AL1_tiling, BL1_tiling, AL0_tiling, \
               BL0_tiling, CL0_tiling, CUB_tiling, \
               AUB_tiling, block_dim_tiling, DOUBLE_BUFFER_tiling

    def AutoTing():
        """"AutoTing"""
        LOC_factor_N = AL0_tiling[4]
        LOC_factor_m = AL0_tiling[0] * AL0_tiling[2]
        LOC_factor_n = BL0_tiling[1]
        LOC_factor_k = BL0_tiling[0]
        CUB_factor_N = CUB_tiling[4]
        CUB_factor_m = CUB_tiling[1] * CUB_tiling[2]
        CUB_factor_n = CUB_tiling[0]
        RES_L0C_factor_m = AL0_tiling[0] * AL0_tiling[2]
        L1_factor_N = AL1_tiling[2]
        L1_factor_m = AL1_tiling[1] * CL0_tiling[1] * 16

        # double buffer
        double_buffer_flag = {
            'AUB_pbuffer': False,
            'AL1_pbuffer': False,
            'BL1_pbuffer': False,
            'AL0_pbuffer': False,
            'BL0_pbuffer': False,
            'CL0_pbuffer': False,
            'CUB_pbuffer': False,
            'UBG_pbuffer': False,
        }
        double_buffer_flag = DOUBLE_BUFFER_tiling
        # muti core bind
        blocks = block_dim_tiling[0] * block_dim_tiling[3]
        mad_cc_axis_n, mad_cc_axis_cg, mad_cc_axis_co1, mad_cc_axis_howomad, \
        mad_cc_axis_co0 = mad_cc.op.axis
        mad_cc_Ncut_o, mad_cc_Ncut_i = s[mad_cc].split(mad_cc_axis_n,
                                                       factor=LOC_factor_N)
        mad_cc_mcut_o, mad_cc_mcut_i = s[mad_cc].split(mad_cc_axis_howomad,
                                                       factor=LOC_factor_m)
        mad_cc_kcut_o, mad_cc_kcut_i = s[mad_cc].split(
            mad_cc.op.reduce_axis[0], factor=LOC_factor_k)
        mad_cc_ncut_o, mad_cc_ncut_i = s[mad_cc].split(mad_cc_axis_co1,
                                                       factor=LOC_factor_n)
        s[mad_cc].reorder(mad_cc_Ncut_o, mad_cc_axis_cg, mad_cc_ncut_o,
                          mad_cc_mcut_o, mad_cc_kcut_o, mad_cc_Ncut_i,
                          mad_cc_ncut_i, mad_cc_mcut_i, mad_cc_axis_co0,
                          mad_cc_kcut_i, mad_cc.op.reduce_axis[1])
        s[dout_ca].compute_at(s[mad_cc], mad_cc_kcut_o)
        s[weight_cb].compute_at(s[mad_cc], mad_cc_kcut_o)

        mad_ubuf_axis_n, mad_ubuf_axis_cg, mad_ubuf_axis_co1, \
        mad_ubuf_axis_howomad, mad_ubuf_axis_co0 = mad_ubuf.op.axis
        mad_ubuf_Ncut_o, mad_ubuf_Ncut_i = s[mad_ubuf].split(
            mad_ubuf_axis_n, factor=CUB_factor_N)
        mad_ubuf_mcut_o, mad_ubuf_mcut_i = s[mad_ubuf].split(
            mad_ubuf_axis_howomad, factor=CUB_factor_m)
        mad_ubuf_ncut_o, mad_ubuf_ncut_i = s[mad_ubuf].split(
            mad_ubuf_axis_co1, factor=CUB_factor_n)
        s[mad_ubuf].reorder(mad_ubuf_Ncut_o, mad_ubuf_axis_cg, mad_ubuf_ncut_o,
                            mad_ubuf_mcut_o, mad_ubuf_Ncut_i, mad_ubuf_ncut_i,
                            mad_ubuf_mcut_i, mad_ubuf_axis_co0)
        s[mad_cc].compute_at(s[mad_ubuf], mad_ubuf_mcut_o)

        conv_Ncut_o, conv_Ncut_i = s[dx_res].split(dx_res.op.axis[0],
                                                   factor=L1_factor_N)
        conv_hcut_o, conv_hcut_i = s[dx_res].split(dx_res.op.axis[3],
                                                   factor=L1_factor_m)
        conv_mcut_o, conv_mcut_i = s[dx_res].split(conv_hcut_i,
                                                   factor=RES_L0C_factor_m)
        s[dx_res].reorder(conv_Ncut_o, dx_res.op.axis[1], conv_hcut_o,
                          conv_mcut_o, conv_Ncut_i, dx_res.op.axis[2],
                          conv_mcut_i, dx_res.op.axis[4])
        s[mad_ubuf].buffer_align((1, 1), (1, 1), (1, 1), (1, block_size),
                                 (1, block_size))
        s[mad_ubuf].compute_at(s[dx_res], conv_mcut_o)
        s[dout_cbuf_row_major].buffer_align((1, 1), (1, 1), (fmap_w, fmap_w),
                                            (1, 1), (1, 1), (1, 1),
                                            (1, block_size))
        s[dout_cbuf_row_major].compute_at(s[dx_res], conv_hcut_o)
        s[dout_cbuf_nc1hwc0].compute_at(s[dx_res], conv_hcut_o)
        s[weight_cbuf].compute_at(s[dx_res], conv_hcut_o)

        if stride > 1:
            AUB_factor_m = AUB_tiling[1]
            ub_l1hcut_o, ub_l1hcut_i = s[dout_cbuf_nc1hwc0].split(
                dout_cbuf_nc1hwc0.op.axis[3], factor=AUB_factor_m)
            s[dout_dilated_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
            s[dout_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
            s[dout_cbuf_nc1hwc0].emit_insn(ub_l1hcut_i, 'dma_copy')
            s[dout_dilated_ubuf].emit_insn(dout_dilated_ubuf.op.axis[5],
                                           'dma_padding')
            s[dout_ubuf].emit_insn(dout_ubuf.op.axis[0], 'dma_copy')
            # aub double buffer
            if double_buffer_flag["AUB_pbuffer"] == 2:
                s[dout_dilated_ubuf].double_buffer()
        else:
            s[dout_dilated_ubuf].compute_inline()
            s[dout_ubuf].compute_inline()
            s[dout_cbuf_nc1hwc0].emit_insn(dout_cbuf_nc1hwc0.op.axis[0],
                                           'dma_copy')

        # emit convolution params.
        setfmatrix_dict = {
            "conv_kernel_h": dx_res.op.attrs['weight_height'],
            "conv_kernel_w": dx_res.op.attrs['weight_width'],
            "conv_padding_top": dx_res.op.attrs['dilated_pad'][0],
            "conv_padding_bottom": dx_res.op.attrs['dilated_pad'][1],
            "conv_padding_left": dx_res.op.attrs['dilated_pad'][2],
            "conv_padding_right": dx_res.op.attrs['dilated_pad'][3],
            "conv_stride_h": dx_res.op.attrs['dilated_strides'][0],
            "conv_stride_w": dx_res.op.attrs['dilated_strides'][1],
            "conv_fm_c": dout_dilated.shape[2] * dout_dilated.shape[5],
            "conv_fm_h": dout_dilated.shape[3],
            "conv_fm_w": dout_dilated.shape[4]
        }
        is_overload = False
        strideH = int(dx_res.op.attrs['dilated_strides'][0])
        strideW = int(dx_res.op.attrs['dilated_strides'][1])
        kernel_h = int(dx_res.op.attrs['weight_height'])
        kernel_w = int(dx_res.op.attrs['weight_width'])
        if block_dim_tiling[1] > 1 or (block_dim_tiling[2] > 1 and
                                       (strideH < kernel_h
                                        or strideW < kernel_w)):
            is_overload = True
        set_pragma_for_cache_read_mode(is_overload, s[dx_res], conv_Ncut_i)

        s[dout_cbuf_row_major].emit_insn(dout_cbuf_row_major.op.axis[1],
                                         'set_fmatrix', setfmatrix_dict)
        s[dout_ca].emit_insn(dout_ca.op.axis[1], 'im2col')
        s[weight_cbuf].emit_insn(weight_cbuf.op.axis[0], 'dma_copy')
        s[weight_cb].emit_insn(weight_cb.op.axis[3], 'dma_copy')
        s[mad_ubuf].emit_insn(mad_ubuf_Ncut_i, 'dma_copy')
        mad_dict = {
            "mad_pattern": cce_params.CONV_MODE,
            "k_outer": mad_cc_kcut_o
        }
        s[mad_cc].emit_insn(mad_cc_Ncut_i, 'mad', mad_dict)
        s[dx_res].emit_insn(conv_Ncut_i, 'dma_copy')

        # turn on dubole buffer
        # al1
        if double_buffer_flag["AL1_pbuffer"] == 2:
            s[dout_cbuf_nc1hwc0].double_buffer()
        # bl1
        if double_buffer_flag["BL1_pbuffer"] == 2:
            s[weight_cbuf].double_buffer()
        # l0a
        if double_buffer_flag["AL0_pbuffer"] == 2:
            s[dout_ca].double_buffer()
        # l0b
        if double_buffer_flag["BL0_pbuffer"] == 2:
            s[weight_cb].double_buffer()
        # L0C
        if double_buffer_flag["CL0_pbuffer"] == 2:
            s[mad_cc].double_buffer()
        # CUB
        if double_buffer_flag["CUB_pbuffer"] == 2:
            s[mad_ubuf].double_buffer()
        s[dx_res].reorder(conv_Ncut_o, dx_res.op.axis[1], conv_hcut_o,
                          conv_mcut_o, conv_Ncut_i, dx_res.op.axis[2],
                          conv_mcut_i, dx_res.op.axis[4])

        # bind muti core
        if blocks != 1:
            res_NNCut_o, res_NNCut_i = s[dx_res].split(
                conv_Ncut_o, nparts=block_dim_tiling[0])
            res_ccCut_o, res_ccCut_i = s[dx_res].split(
                dx_res.op.axis[1], nparts=block_dim_tiling[3])
            s[dx_res].reorder(res_NNCut_o, res_ccCut_o, res_NNCut_i,
                              res_ccCut_i)
            out_fused = s[dx_res].fuse(res_NNCut_o, res_ccCut_o)
            out_fused_out, _ = s[dx_res].split(out_fused, nparts=blocks)
            bind_out, _ = s[dx_res].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            s[dx_res].bind(bind_out, blockidx)

    def _get_tiling_plan(dst_w, filter_shape, filter_height, dout_shape,
                         stride):
        """
        tiling plan, with a cut of batch and ci;
                     with a cut of output height and weight;
                     with a cut of m , k, n
        dst_h: fmap_h
        dst_w: fmap_w
        filter_shape: C1, Hf*Wf, 1, C0, C0
        dout_shape: N, Co1, 1, Ho, Wo, C0
        stride: strideH, strideW
        """
        # float16
        data_size = 2
        l1_size = cce_conf.get_soc_spec(cce_conf.L1_SIZE)
        l0a_size = cce_conf.get_soc_spec(cce_conf.L0A_SIZE)
        ub_size = cce_conf.get_soc_spec(cce_conf.UB_SIZE)

        hf_wf = filter_shape[1]
        ho = dout_shape[3]
        wo = dout_shape[4]
        # compute dilation shape
        wd = wo * stride - (stride - 1)
        hd = ho * stride - (stride - 1)
        max_h_in_l1 = (l1_size - hf_wf * BLOCK_SIZE * BLOCK_SIZE * data_size) \
                      // (data_size * wd * BLOCK_SIZE) - (filter_height - 1)
        if max_h_in_l1 < BLOCK_SIZE:
            raise RuntimeError("tile_hd must be 16x!")
        tile_hd = hd if hd <= max_h_in_l1 else (max_h_in_l1 // BLOCK_SIZE *
                                                BLOCK_SIZE)

        tile_k = 1
        tile_n = 1
        # 2 is for double buffer
        max_l0a_m = l0a_size // (data_size * BLOCK_SIZE * 2)
        tile_fm_h = tile_hd + dx_res.op.attrs['dilated_pad'][
            0].value + dx_res.op.attrs['dilated_pad'][
                        1].value - dx_res.op.attrs['weight_height'].value + 1
        tile_m = min(max_l0a_m, _ceil((tile_fm_h) * dst_w))

        # UB : dout : dout_n, dout_cgroup, dout_c1, dout_h, dout_w, dout_c0
        #      dilate : input_shape[0], input_shape[1], input_shape[2],
        #               dilated_h, dilated_w, input_shape[5]
        #      cast : dout_n, dout_cgroup, dout_c1, input_h*input_w, dout_c0
        max_h_in_ub = ((ub_size // data_size -
                        (tile_m * BLOCK_SIZE)) // BLOCK_SIZE +
                       (stride - 1) * wd) // (wo + stride * wd)
        tile_ho = (tile_hd + (stride - 1)) // stride
        tile_h_ub = min(tile_ho, max_h_in_ub)
        # dilated_h
        tile_h_ub = tile_h_ub * stride - (stride - 1)

        return tile_h_ub, tile_hd, tile_m, tile_k, tile_n

    def NoAutoTing():
        """"NoAutoTing"""
        tile_h_ub, tile_h, tile_m, tile_k, tile_n = _get_tiling_plan(
            fmap_w, weight_shape, weight_height, dout_shape, stride)
        mad_cc_axis_n, mad_cc_axis_cg, mad_cc_axis_co1, mad_cc_axis_howomad, \
        mad_cc_axis_co0 = mad_cc.op.axis

        mad_cc_Ncut_o, mad_cc_Ncut_i = s[mad_cc].split(mad_cc_axis_n, factor=1)
        mad_cc_mcut_o, mad_cc_mcut_i = s[mad_cc].split(mad_cc_axis_howomad,
                                                       factor=tile_m)
        mad_cc_kcut_o, mad_cc_kcut_i = s[mad_cc].split(
            mad_cc.op.reduce_axis[0], factor=tile_k)
        mad_cc_ncut_o, mad_cc_ncut_i = s[mad_cc].split(mad_cc_axis_co1,
                                                       factor=tile_n)
        s[mad_cc].reorder(mad_cc_Ncut_o, mad_cc_axis_cg, mad_cc_ncut_o,
                          mad_cc_mcut_o, mad_cc_kcut_o, mad_cc_Ncut_i,
                          mad_cc_ncut_i, mad_cc_mcut_i, mad_cc_axis_co0,
                          mad_cc_kcut_i, mad_cc.op.reduce_axis[1])
        s[dout_ca].compute_at(s[mad_cc], mad_cc_kcut_o)
        s[weight_cb].compute_at(s[mad_cc], mad_cc_kcut_o)

        mad_ubuf_axis_n, mad_ubuf_axis_cg, mad_ubuf_axis_co1, \
        mad_ubuf_axis_howomad, mad_ubuf_axis_co0 = mad_ubuf.op.axis
        mad_ubuf_Ncut_o, mad_ubuf_Ncut_i = s[mad_ubuf].split(mad_ubuf_axis_n,
                                                             factor=1)
        mad_ubuf_mcut_o, mad_ubuf_mcut_i = s[mad_ubuf].split(
            mad_ubuf_axis_howomad, factor=tile_m)
        mad_ubuf_ncut_o, mad_ubuf_ncut_i = s[mad_ubuf].split(mad_ubuf_axis_co1,
                                                             factor=tile_n)
        s[mad_ubuf].reorder(mad_ubuf_axis_cg, mad_ubuf_Ncut_o, mad_ubuf_ncut_o,
                            mad_ubuf_mcut_o, mad_ubuf_Ncut_i, mad_ubuf_ncut_i,
                            mad_ubuf_mcut_i, mad_ubuf_axis_co0)
        s[mad_cc].compute_at(s[mad_ubuf], mad_ubuf_mcut_o)

        conv_Ncut_o, conv_Ncut_i = s[dx_res].split(dx_res.op.axis[0], factor=1)
        tile_h_fm = tile_h + dx_res.op.attrs['dilated_pad'][0].value \
                    + dx_res.op.attrs['dilated_pad'][1].value \
                    - dx_res.op.attrs['weight_height'].value + 1
        conv_hcut_o, conv_hcut_i = s[dx_res].split(dx_res.op.axis[3],
                                                   factor=tile_h_fm * fmap_w)
        conv_mcut_o, conv_mcut_i = s[dx_res].split(conv_hcut_i, factor=tile_m)
        s[dx_res].reorder(conv_Ncut_o, dx_res.op.axis[1], conv_hcut_o,
                          conv_mcut_o, conv_Ncut_i, dx_res.op.axis[2],
                          conv_mcut_i, dx_res.op.axis[4])
        s[mad_ubuf].buffer_align((1, 1), (1, 1), (1, 1), (1, block_size),
                                 (1, block_size))
        s[mad_ubuf].compute_at(s[dx_res], conv_mcut_o)
        s[dout_cbuf_row_major].buffer_align((1, 1), (1, 1), (1, fmap_w),
                                            (1, 1), (1, 1), (1, 1),
                                            (1, block_size))
        s[dout_cbuf_row_major].compute_at(s[dx_res], conv_hcut_o)
        s[dout_cbuf_nc1hwc0].compute_at(s[dx_res], conv_hcut_o)
        s[weight_cbuf].compute_at(s[dx_res], dx_res.op.axis[1])

        if stride > 1:
            dout_dilated_w = dout_dilated_shape[3]
            tile_dilated_h = tile_h

            s[dout_cbuf_nc1hwc0].buffer_tile(
                (conv_Ncut_o.var, 1), (dx_res.op.axis[1].var, 1), (0, 1),
                (conv_hcut_o.var * tile_dilated_h, tile_dilated_h),
                (0, dout_dilated_w), (0, 16))

            ub_l1hcut_o, ub_l1hcut_i = s[dout_cbuf_nc1hwc0].split(
                dout_cbuf_nc1hwc0.op.axis[3], factor=tile_h_ub)
            s[dout_dilated_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
            s[dout_ubuf].compute_at(s[dout_cbuf_nc1hwc0], ub_l1hcut_o)
            s[dout_cbuf_nc1hwc0].emit_insn(ub_l1hcut_i, 'dma_copy')

            s[dout_dilated_ubuf].emit_insn(dout_dilated_ubuf.op.axis[5],
                                           'dma_padding')

            s[dout_ubuf].emit_insn(dout_ubuf.op.axis[0], 'dma_copy')
        else:
            s[dout_dilated_ubuf].compute_inline()
            s[dout_ubuf].compute_inline()

            s[dout_cbuf_nc1hwc0].emit_insn(dout_cbuf_nc1hwc0.op.axis[0],
                                           'dma_copy')

        # emit convolution params.
        setfmatrix_dict = {
            "conv_kernel_h": dx_res.op.attrs['weight_height'],
            "conv_kernel_w": dx_res.op.attrs['weight_width'],
            "conv_padding_top": dx_res.op.attrs['dilated_pad'][0],
            "conv_padding_bottom": dx_res.op.attrs['dilated_pad'][1],
            "conv_padding_left": dx_res.op.attrs['dilated_pad'][2],
            "conv_padding_right": dx_res.op.attrs['dilated_pad'][3],
            "conv_stride_h": dx_res.op.attrs['dilated_strides'][0],
            "conv_stride_w": dx_res.op.attrs['dilated_strides'][1],
            "conv_fm_c": dout_dilated.shape[2] * dout_dilated.shape[5],
            "conv_fm_h": dout_dilated.shape[3],
            "conv_fm_w": dout_dilated.shape[4]
        }

        s[dout_cbuf_row_major].emit_insn(dout_cbuf_row_major.op.axis[1],
                                         'set_fmatrix', setfmatrix_dict)
        s[dout_ca].emit_insn(dout_ca.op.axis[1], 'im2col')
        s[weight_cbuf].emit_insn(weight_cbuf.op.axis[0], 'dma_copy')
        s[weight_cb].emit_insn(weight_cb.op.axis[3], 'dma_copy')
        s[mad_ubuf].emit_insn(mad_ubuf_Ncut_i, 'dma_copy')
        mad_dict = {
            "mad_pattern": cce_params.CONV_MODE,
            "k_outer": mad_cc_kcut_o
        }
        is_overload = False
        strideH = int(dx_res.op.attrs['dilated_strides'][0])
        strideW = int(dx_res.op.attrs['dilated_strides'][1])
        kernel_h = int(dx_res.op.attrs['weight_height'])
        kernel_w = int(dx_res.op.attrs['weight_width'])
        if block_dim_tiling[1] > 1 or (block_dim_tiling[2] > 1 and
                                       (strideH < kernel_h
                                        or strideW < kernel_w)):
            is_overload = True
        set_pragma_for_cache_read_mode(is_overload, s[dx_res], conv_Ncut_i)
        s[mad_cc].emit_insn(mad_cc_Ncut_i, 'mad', mad_dict)
        s[dx_res].emit_insn(conv_Ncut_i, 'dma_copy')

        s[dout_ca].double_buffer()
        s[weight_cb].double_buffer()
        s[mad_cc].double_buffer()

        # for multi cores
        block = tvm.thread_axis("blockIdx.x")
        s[dx_res].bind(conv_Ncut_o, block)

    AL1_tiling, _, AL0_tiling, BL0_tiling, CL0_tiling, \
    CUB_tiling, AUB_tiling, block_dim_tiling, \
    DOUBLE_BUFFER_tiling = _tiling_fetch()
    if AL0_tiling[2] == 32:
        NoAutoTing()
    else:
        AutoTing()

    return s
