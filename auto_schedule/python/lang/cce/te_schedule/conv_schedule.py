"""Copyright 2019 Huawei Technologies Co., Ltd

Schedule of conv2d.
"""
import re
from enum import Enum
from te import platform as cce
from te.lang.cce import ConvParam
from te.lang.cce import MaxPoolParam
from te.lang.cce import is_support_v200
from te.platform import cce_conf
from te.platform import CUBE_MKN
from te.platform import conv_buffer_ex
from te.platform.cce_buffer import cur_cce_product_params as cce_product
from te.domain.tiling.tiling_query import tiling_query
from te.domain.tiling.get_tiling import get_tiling
from te.domain.tiling.tiling_helper import TILING_INSTANCE
from te import tvm
import te.lang.cce
from te.platform import cce_util
from te.platform import get_soc_spec
from te.platform.cce_policy import get_L1_info
from te.platform.fusion_manager import get_fusion_build_cfg
from te.utils.error_manager import error_manager_conv2d as err_man
from . import util


# tiling check
TILING_AL1_SHAPWE_DIM = 4
TILING_BL1_SHAPWE_DIM = 4
TILING_AL0_MATRIX_DIM = 6
TILING_BL0_MATRIX_DIM = 6
TILING_CL0_MATRIX_DIM = 6
TILING_CUB_MATRIX_DIM = 6
TILING_BLOCK_DIM_DIM = 4
TILING_FLOAT16_M = 16
TILING_FLOAT16_K_C04 = 4
TILING_FLOAT16_K = 16
TILING_FLOAT16_N = 16
TILING_INT8_M = 16
TILING_INT8_K = 32
TILING_INT8_N = 16
CONV_OP_NUM = 5
SMALL_GRAPH_OP_NUM = 16
L1_FUSION_SCOPE = 1
DDR_SCOPE = 0

# the pooling nums
POOLING_STRIDE = 2
POOLING_WINDOW = 3

def get_srctensor(tensor):
    """
    fetch input_tensor tensor if it exists

    Parameters
    ----------
    outs : the input_tensor tensor

    Returns
    -------

    """
    if tensor.op.input_tensors:
        return tensor.op.input_tensors[0]
    return None

def get_read_select_srctensor(tensor, tensor_map):
    """
    fetch input_tensor of read_select

    Parameters
    ----------
    outs : the tensor_map

    Returns
    -------

    """
    output_ub_4d = tensor
    tensor_map["output_ub_4d"] = output_ub_4d
    output_ub_5d = get_srctensor(output_ub_4d)
    tensor_map["output_ub_5d"] = output_ub_5d
    return tensor_map


def check_conv_bn1(outs):
    """
    check conv + bn1

    Parameters
    ----------
    outs : the outputs of op

    Returns
    -------

    """
    if isinstance(outs, tvm.tensor.Tensor) or not isinstance(outs, list) or \
            len(outs) != 3:
        return False

    conv_out, reduce_0, reduce_1 = outs
    if "convolution_" not in conv_out.op.tag or \
            ("reduce_sum" not in reduce_0.op.tag) or \
            ("reduce_sum" not in reduce_1.op.tag):
        return False

    # check conv_type is fp16
    if conv_out.dtype != "float16":
        return False
    # check cast->reduce_0
    reduce0_src = get_srctensor(reduce_0)
    reduce0_src_pre = get_srctensor(reduce0_src)

    if ("elewise_single_cast" not in reduce0_src.op.tag) or \
            (reduce0_src_pre != conv_out):
        return False
    # check reduce_axis
    if (conv_out.op.axis[1].dom.extent.value !=
            reduce_0.op.axis[1].dom.extent.value) or \
        (conv_out.op.axis[1].dom.extent.value !=
         reduce_1.op.axis[1].dom.extent.value):
        return False

    return True


def check_doubleout_quant_v200(outs):
    """
    check v200 quant fuse for double out

    Parameters
    ----------
    outs : the outputs of op

    Returns
    -------

    """
    if isinstance(outs, tvm.tensor.Tensor) or not isinstance(outs, list) or \
            len(outs) != 2:
        return False

    # conv+dequant+requant fusion pattern
    #   conv
    #     |
    # dequant_s16
    #      |
    #  requant_s16
    #    / \
    #   /   \
    #  s8  s16
    out_s8, relu_s16 = outs

    if ("requant_s16_vadd" not in relu_s16.op.tag) or \
            ("requant_s16_data_transfer" not in out_s8.op.tag):
        return False

    # check dtype is s16 and s8
    if (relu_s16.dtype != "int16") or (out_s8.dtype != "int8"):
        return False

    deqs16_rmpad = None
    for i in relu_s16.op.input_tensors:
        if i.op.tag == "dequant_s16_remove_pad":
            deqs16_rmpad = i
    if deqs16_rmpad is None:
        return False

    deqs16_res = get_srctensor(deqs16_rmpad)
    if deqs16_res is None:
        return False

    return True


def check_doubleout_dequant_v100(outs):
    """
    checkout conv + dequant + quant double output

    Parameters
    ----------
    outs : true when check_dequant_addrelu_quant_db_v100

    Returns

    -------
    """

    if isinstance(outs, tvm.tensor.Tensor):
        return False

    if not isinstance(outs, list):
        return False

    if len(outs) != 2:
        return False

    out_fp16, out_s8 = outs

    ori_out_fp16 = out_fp16.op.input_tensors[0] \
        if "write_select" in out_fp16.op.name else out_fp16
    ori_out_s8 = out_s8.op.input_tensors[0] \
        if "write_select" in out_s8.op.name else out_s8

    if "quant" not in ori_out_s8.op.tag:
        return False

    if ori_out_fp16.dtype != "float16" or ori_out_s8.dtype != "int8":
        return False

    return True


def reget_tensor_list(outs):
    """
    redefine the tensor_map for conv + bn1

    redefine the tensor map for conv + dequant + add + relu +quant

    Parameters
    ----------
    outs : the outputs of op

    Returns
    -------
    outputs

    """
    def _fcombine(arg_0, arg_1):
        return arg_0[0] + arg_1[0], arg_0[1] + arg_1[1]

    def _fidentity(t_0, t_1):
        return tvm.const(0, t_0), tvm.const(0, t_1)

    ConvParam.convbn1_flag = False
    if check_conv_bn1(outs):
        conv_out, _, _ = outs
        conv_res_shape = tuple(conv_out.shape)
        reduce_shape = (conv_res_shape[1], conv_res_shape[3])
        k_0 = tvm.reduce_axis((0, conv_res_shape[0]), name='k_0')
        k_1 = tvm.reduce_axis((0, conv_res_shape[2]), name='k_1')
        cub = conv_out.op.input_tensors[0]
        c_col = cub.op.input_tensors[0]
        c_ub = tvm.compute(cub.shape,
                           lambda n, i, j, k: c_col(n, i, j, k),
                           name='c_ub',
                           tag="convolution_" + "c_ub",
                           attrs=cub.op.attrs)
        ConvParam.TENSOR_MAP["c_ub"] = c_ub
        c_res = tvm.compute(conv_res_shape,
                            lambda *indice: c_ub(*indice),
                            name="C",
                            tag="convolution_" + "C")
        ConvParam.TENSOR_MAP["C"] = c_res
        cast_0_ub = tvm.compute(conv_res_shape,
                                lambda *indice:
                                c_res(*indice).astype("float16"),
                                name="cast_0_ub")
        cast_0 = tvm.compute(conv_res_shape,
                             lambda *indice: cast_0_ub(*indice),
                             name="cast_0")
        cast_1 = tvm.compute(conv_res_shape,
                             lambda *indice: cast_0(*indice).astype("float32"),
                             name="cast_1")
        ConvParam.TENSOR_MAP["cast_1"] = cast_1
        mul_0 = te.lang.cce.vmul(cast_1, cast_1)

        tuple_reduce = tvm.comm_reducer(_fcombine,
                                        _fidentity,
                                        name='tuple_reduce')
        mean_out, _ = tvm.compute(reduce_shape,
                                  lambda c1, c0:
                                  tuple_reduce((cast_1[k_0, c1, k_1, c0],
                                                mul_0[k_0, c1, k_1, c0]),
                                               axis=[k_0, k_1]),
                                  name="mean_out")
        outputs = [cast_0, mean_out]
        ConvParam.convbn1_flag = True
    elif check_doubleout_quant_v200(outs):
        out_s8, relu_s16 = outs
        res_remove_pad_u8 = tvm.compute(out_s8.shape,
                                        lambda i, j, k, l: out_s8(i, j, k, l),
                                        name='res_remove_pad_u8',
                                        tag='res_remove_pad_u8')
        res_remove_pad_s16 = tvm.compute(relu_s16.shape,
                                         lambda i, j, k, l:
                                         relu_s16(i, j, k, l),
                                         name='res_remove_pad_s16',
                                         tag='res_remove_pad_s16')
        ConvParam.TENSOR_MAP["res_remove_pad_u8"] = res_remove_pad_u8
        ConvParam.TENSOR_MAP["res_remove_pad_s16"] = res_remove_pad_s16

        virtual_res = tvm.compute(out_s8.shape,
                                  lambda i, j, k, l:
                                  res_remove_pad_u8(i, j, k, l) +
                                  res_remove_pad_s16(i, (j*32 + l) // 16, k,
                                                     (j*32 + l) % 16),
                                  name='virtual_res',
                                  tag="conv_virtual_res")
        ConvParam.TENSOR_MAP["virtual_res"] = virtual_res

        ConvParam.TENSOR_MAP["c_double_output_s16"] = relu_s16
        ConvParam.TENSOR_MAP["c_ub_reform"] = out_s8

        ConvParam.conv_deq_req_double_out = True
        outputs = [virtual_res, res_remove_pad_u8, res_remove_pad_s16]
    elif check_doubleout_dequant_v100(outs):
        out_f16, out_s8 = outs

        if "write_select" in out_f16.op.name:
            res_out_fp16 = out_f16
        else:
            if "addr_type" in out_f16.op.attrs:
                out_fp16_addr = out_f16.op.attrs["addr_type"].value
            else:
                out_fp16_addr = DDR_SCOPE

            res_out_fp16 = tvm.compute(out_f16.shape,
                                       lambda i, j, k, l: out_f16(i, j, k, l),
                                       name='res_out_fp16',
                                       tag='res_out_fp16',
                                       attrs={"addr_type": out_fp16_addr})

        ConvParam.TENSOR_MAP["res_out_fp16"] = res_out_fp16
        ConvParam.TENSOR_MAP["res_out_s8"] = out_s8

        virtual_res = tvm.compute(out_s8.shape,
                                  lambda i, j, k, l:
                                  out_s8(i, j, k, l) +
                                  res_out_fp16(i, (j*32 + l) // 16, k,
                                               (j*32 + l) % 16),
                                  name='conv_virtual_res',
                                  tag="conv_virtual_res",
                                  )
        ConvParam.TENSOR_MAP["virtual_res"] = virtual_res

        outputs = [virtual_res, res_out_fp16, out_s8]
        ConvParam.conv_deq_req_double_out = True
    else:
        outputs = outs

    return outputs


def check_quantfuse_doubleout(tensor_list, sch):
    """
    checkout if deuquant requant(quant) double out or not

    Parameters
    ----------
    tensor_list : the tensor of output

    sch : tvm.schedule
        schedule to build or to print lower code

    Returns
    -------
    tensor_list

    """
    if hasattr(ConvParam, "conv_deq_req_double_out"):
        if ConvParam.conv_deq_req_double_out:
            tensor_list = tensor_list[:-2]
            tensor_list.append(sch.cce_special['real_out_tensor'][1])
            tensor_list.append(sch.cce_special['real_out_tensor'][2])
            ConvParam.conv_deq_req_double_out = False

    for tensor in tensor_list:
        if "conv_virtual_res" in tensor.op.name:
            tensor_list.remove(tensor)

    return tensor_list


def reset_mask_insn(i_b, type_, bits=128, mask_func=None):
    """
    caculate the mask, and set vector mask

    Parameters
    ----------
    param i_b: ir builder

    param type_: the type of mask dst

    param bits: the bit of mask, default : 128

    Returns
    -------
    """
    # argmin/argmax has his own set_mask func
    if mask_func is not None:
        mask1, mask2 = mask_func(bits)
    else:
        mask1, mask2 = cce_util.set_mask(bits)

    i_b.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))


def check_feature_map(tiling_new, al1_factor):
    """
    check whether feature_map is overload

    Parameters
    ----------
    tiling_new : tiling result

    al1_factor: al1_factor[0] == 1 means AL1 k is full load,
                al1_factor[1] == 1 means AL1 m is full load

    Returns
    -------
    true for overload, false for not overload
    """
    if tiling_new["block_dim"][1] > 1:
        return True
    if al1_factor[0] != 1:
        return True
    if ((ConvParam.stride_h < ConvParam.filter_h or \
        ConvParam.stride_w < ConvParam.filter_w) and (al1_factor[1] != 1)):
        return True

    return False


class CceConvOp:

    """class of cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    Returns
    -------
    cceop_instance : instance of cceop

    """

    def __init__(self):
        self._schedule = None
        self._max_pool_tensor_map = MaxPoolParam.tensor_map
        self._dim_map = ConvParam.dim_map
        self._res_tensor = None
        self._op_graph = None
        self._fused_double_operand_num = 0
        self._aipp_fuse_flag = False
        self._m_part_nums = 1
        self._convbn1_flag = False
        self._input_memory_type = []
        self._output_memory_type = []
        self._valid_shape = []
        self._l1_fusion_type = -1
        self._fmap_l1_addr_flag = "nothing"
        self._fmap_l1_valid_size = 0
        self._vector_read_select = False
        self._write_select = False
        self._v200_data_flow_type = None
        self._lhisi_data_flow_type = None
        self.overload_flag = False
        self.conv_pool_fused_flag = False
        self._conv_quant_fused_flag = False
        self._conv1d_split_w_flag = False
        self._l0b_first_flag = False
        self._lhisi_dequant_quant_para = {'deq_sqrt': False,
                                          'deq_relu': False,
                                          'deq_vector': False,
                                          'quant_round': None,
                                          'quant_padding': False}
        self._emit_insn_map = {"elewise_single_relu": "vector_relu",
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
                               "elewise_binary_cmpsel_gt": "vector_select_gt",
                               "elewise_binary_cmpsel_ge": "vector_select_ge",
                               "elewise_binary_cmpsel_lt": "vector_select_lt",
                               "elewise_binary_cmpsel_le": "vector_select_le",
                               "elewise_binary_cmpsel_eq": "vector_select_eq",
                               "elewise_binary_cmpsel_ne": "vector_select_ne",
                               "elewise_binary_cmpsel": "vector_cmpsel",
                               "elewise_binary_add": "vector_add",
                               "elewise_binary_sub": "vector_sub",
                               "elewise_binary_mul": "vector_mul",
                               "elewise_binary_min": "vector_min",
                               "elewise_binary_max": "vector_max",
                               "elewise_binary_or": "vector_or",
                               "elewise_binary_and": "vector_and",
                               "elewise_single_relu": "vector_relu",
                               "elewise_single_lrelu": "vector_lrelu",
                               "elewise_binary_addrelu": "vector_addrelu",
                               "elewise_binary_subrelu": "vector_subrelu",
                               "elewise_multiple_sel": "vector_select_bool"}
        self._l1_size = cce_conf.get_soc_spec("L1_SIZE")
        self._corenum = cce_conf.get_soc_spec("CORE_NUM")
        self._ub_size = cce_conf.get_soc_spec("UB_SIZE")
        self.unzip_parameters = {"weight_zip_flag": False,
                                 "max_block_size": 32 * 1024,
                                 "compact_mode_index_size": 8,
                                 "uncompact_mode_index_size": 2,
                                 "compress_flag": 2,
                                 "compress_tiling": [0, 0, 0, 0]}
        self._var_map = ConvParam.var_map
        self._dynamic_mode = ConvParam.dynamic_mode
        self._tiling_case = None
        self._var_range = None
        self._fused_flag = False
        self._compute_at_buffer = None
        self._compute_at_axis = None

    def schedule(self, res, spec_node_list, sch_list, \
                 convbn1_flag=False, tiling_case=None, var_range=None):
        """
        auto_schedule for cce AI-CORE. For now, only one convolution operation
        is supported.

        Parameters
        ----------
        res: tvm.tensor

        spec_node_list: same as other template in cce_schedule

        sch_list: use sch_list[0] to return conv schedule

        Returns
        -------
        True for sucess, False for no schedule
        """
        def int_ceil_div(num_a, num_b):
            """
            upper division
            """
            if num_b == 0:
                err_man.raise_err_specific("conv2d", "division by zero")
            return (num_a + num_b - 1) // num_b

        def ceil(num_a, num_b):
            """
            upper align
            """
            if num_b == 0:
                err_man.raise_err_specific("conv2d", "division by zero")
            return (num_a + num_b - 1) // num_b * num_b

        def __lcm(wout, factor):
            # get least common multiple of wout and factor
            tmp = wout * factor
            while wout % factor != 0:
                wout, factor = factor, (wout % factor)
            return tmp // factor
        def int_ceil_div_tvm(num_a, num_b):
            return tvm.floordiv((num_a + num_b - 1), num_b)

        def check_tiling(tiling, w_dtype, fmap_shape_nc1hwc0):
            """
            default tiling check

            Returns
            -------
            true for auto tiling, false for default tiling
            """
            def check_tiling_m_k_fp16(tiling):
                if tiling["AL0_matrix"][2] != TILING_FLOAT16_M:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['AL0_matrix'][2]", str(TILING_FLOAT16_M), \
                        "when w_dtype is float16")
                if tiling["AL0_matrix"][3] not in (TILING_FLOAT16_K,
                                                   TILING_FLOAT16_K_C04):
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['AL0_matrix'][3]", str(TILING_FLOAT16_K) + \
                        "or" + \
                        str(TILING_FLOAT16_K_C04), "when w_dtype is float16")
                if tiling["BL0_matrix"] != []:
                    if tiling["BL0_matrix"][2] != TILING_FLOAT16_N:
                        err_man.raise_err_value_or_format_invalid("conv2d", \
                            "tiling['BL0_matrix'][2]", str(TILING_FLOAT16_N), \
                            "when w_dtype is float16")
                    if tiling["BL0_matrix"][3] not in (TILING_FLOAT16_K,
                                                       TILING_FLOAT16_K_C04):
                        err_man.raise_err_value_or_format_invalid("conv2d", \
                            "tiling['BL0_matrix'][3]", \
                            str(TILING_FLOAT16_K) + "or" + \
                            str(TILING_FLOAT16_K_C04), \
                            "when w_dtype is float16")
                if tiling["CL0_matrix"][2] != TILING_FLOAT16_M:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['CL0_matrix'][2]", str(TILING_FLOAT16_M), \
                        "when w_dtype is float16")
                if tiling["CL0_matrix"][3] != TILING_FLOAT16_N:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['CL0_matrix'][3]", str(TILING_FLOAT16_N), \
                        "when w_dtype is float16")

            def check_tiling_m_k_int8(tiling):
                if tiling["AL0_matrix"][2] != TILING_INT8_M:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['AL0_matrix'][2]", str(TILING_INT8_M), \
                        "when w_dtype is float16")
                if tiling["AL0_matrix"][3] != TILING_INT8_K:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['AL0_matrix'][3]", str(TILING_INT8_K), \
                        "when w_dtype is float16")
                if tiling["BL0_matrix"] != []:
                    if tiling["BL0_matrix"][2] != TILING_INT8_N:
                        err_man.raise_err_value_or_format_invalid("conv2d", \
                            "tiling['BL0_matrix'][2]", str(TILING_INT8_N), \
                            "when w_dtype is float16")
                    if tiling["BL0_matrix"][3] != TILING_INT8_K:
                        err_man.raise_err_value_or_format_invalid("conv2d", \
                            "tiling['BL0_matrix'][3]", str(TILING_INT8_K), \
                            "when w_dtype is float16")
                if tiling["CL0_matrix"][2] != TILING_INT8_M:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['CL0_matrix'][2]", str(TILING_INT8_M), \
                        "when w_dtype is float16")
                if tiling["CL0_matrix"][3] != TILING_INT8_N:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "tiling['CL0_matrix'][3]", str(TILING_INT8_N), \
                        "when w_dtype is float16")

            def check_default_tiling():
                if tiling["AL0_matrix"][2] == 32:
                    return True
                if not isinstance(tiling["AL1_shape"], list):
                    return True
                return False

            def handle_l1_fusion():
                if self._l1_fusion_type == 1:
                    if tiling["AL1_shape"] and tiling["AL1_shape"][0] != []:
                        err_man.raise_err_value_or_format_invalid("conv2d", \
                            "tiling['AL1_shape'][0]", str([]), \
                            "when L1_fusion_type is breadth fusion")

                if self._l1_fusion_type in [0, 1]:
                    if tiling["block_dim"] != [1, 1, 1, 1]:
                        err_man.raise_err_specific("conv2d", \
                            "only support one core tiling "\
                            + "in L1 Fusion situation")

            def handle_inner_batch():
                if self._dynamic_mode == "dynamic_batch":
                    return
                batch = int(fmap_shape_nc1hwc0[0])
                if len(tiling["AL1_shape"]) > 2:
                    if tiling["AL1_shape"][2] > 1:
                        if tiling["AL1_shape"][2] != batch and \
                        (not self._convbn1_flag):
                            err_man.raise_err_equal_invalid("conv2d", \
                                "al1_batch", "batch")

            if tiling is None:
                err_man.raise_err_specific("conv2d", "tiling is None")

            if check_default_tiling():
                return False

            handle_l1_fusion()

            if tiling["AL1_shape"] != []:
                if len(tiling["AL1_shape"]) != TILING_AL1_SHAPWE_DIM:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "AL1_shape dim", str(TILING_AL1_SHAPWE_DIM), "")

            if (tiling["BL1_shape"] != []) and \
                    (tiling["BL1_shape"] is not None):
                if len(tiling["BL1_shape"]) != TILING_BL1_SHAPWE_DIM:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "BL1_shape dim", str(TILING_BL1_SHAPWE_DIM), "")

            if len(tiling["AL0_matrix"]) != TILING_AL0_MATRIX_DIM:
                err_man.raise_err_value_or_format_invalid("conv2d", \
                    "AL0_matrix dim", str(TILING_AL0_MATRIX_DIM), "")

            if tiling["BL0_matrix"] != []:
                if len(tiling["BL0_matrix"]) != TILING_BL0_MATRIX_DIM:
                    err_man.raise_err_value_or_format_invalid("conv2d", \
                        "BL0_matrix dim", str(TILING_BL0_MATRIX_DIM), "")

            if len(tiling["CL0_matrix"]) != TILING_CL0_MATRIX_DIM:
                err_man.raise_err_value_or_format_invalid("conv2d", \
                    "CL0_matrix dim", str(TILING_CL0_MATRIX_DIM), "")

            if len(tiling["CUB_matrix"]) != TILING_CUB_MATRIX_DIM:
                err_man.raise_err_value_or_format_invalid("conv2d", \
                    "CUB_matrix dim", str(TILING_CUB_MATRIX_DIM), "")

            if len(tiling["block_dim"]) != TILING_BLOCK_DIM_DIM:
                err_man.raise_err_value_or_format_invalid("conv2d", \
                    "block_dim dim", str(TILING_BLOCK_DIM_DIM), "")

            if not isinstance(tiling["manual_pingpong_buffer"], dict):
                err_man.raise_err_value_or_format_invalid("conv2d", \
                    "manual_pingpong_buffer", "dict", "")

            if tiling["AL0_matrix"][0] != tiling["CL0_matrix"][1]:
                err_man.raise_err_equal_invalid("conv2d", "mA", "mC")

            if tiling["BL0_matrix"] != []:
                if tiling["AL0_matrix"][1] != tiling["BL0_matrix"][0]:
                    err_man.raise_err_equal_invalid("conv2d", "kA", "kB")
                if tiling["BL0_matrix"][1] != tiling["CL0_matrix"][0]:
                    err_man.raise_err_equal_invalid("conv2d", "nB", "nC")

            if w_dtype == "float16":
                check_tiling_m_k_fp16(tiling)
            elif w_dtype == "int8":
                check_tiling_m_k_int8(tiling)
            else:
                err_man.raise_err_value_or_format_invalid("conv2d", \
                    "weight", "float16 or int8", "")

            handle_inner_batch()

            return True


        def tiling_fetch():
            """
            get tiling.

            Returns
            -------
            tiling
            """
            def get_fused_ub_cl0():
                double_num_cl0 = 0

                def _checkout_input_ops(lop):
                    if self._lhisi_data_flow_type or self._v200_data_flow_type:
                        return True
                    if "convolution" in lop["op"] or ( \
                        ("bias_tensor" in lop["op"]) and ( \
                        "bias" in tensor_map.keys())):
                        return True
                    if ("bias_tensor" in lop["op"]) and \
                            ("fp16_bias" in tensor_map.keys()):
                        return True
                    if "max_pooling_pad_" in lop["op"]:
                        return True
                    if self.conv_pool_fused_flag:
                        return True
                    if ("dma_copy" in lop["next_op"][0][ \
                        "dst_buffer"].op.attrs) or ( \
                        "fusion_fmap_select" in lop["next_op"][0]["op"]):
                        return True
                    if lop["dst_buffer"].name == 'compress_index' or lop[ \
                        "dst_buffer"].name == "Filter":
                        return True
                    return False

                for lop in self._op_graph.input_ops:
                    if _checkout_input_ops(lop):
                        continue
                    double_num_cl0 += 1

                return double_num_cl0

            def get_compress_tiling_shape(tiling_ok_flag):
                if not tiling_ok_flag:
                    self.unzip_parameters["compress_tiling"] = \
                            tiling["BL0_matrix"][0:4]
                else:
                    weight = ConvParam.TENSOR_MAP["filter"]
                    weight_shape = list(weight.shape)
                    weight_dtype = weight.dtype
                    if tiling_new["BL1_shape"] == []:
                        self.unzip_parameters["compress_tiling"] = weight_shape
                    elif tiling_new["BL1_shape"] is None:
                        if tiling_new["BL0_matrix"] == []:
                            self.unzip_parameters["compress_tiling"] = weight_shape
                        else:
                            self.unzip_parameters["compress_tiling"] = \
                                tiling_new["BL0_matrix"][0:4]
                    else:
                        self.unzip_parameters["compress_tiling"][3] = \
                        CUBE_MKN[w_dtype]['mac'][1]
                        self.unzip_parameters["compress_tiling"][2] = \
                        CUBE_MKN[w_dtype]['mac'][0]
                        self.unzip_parameters["compress_tiling"][1] = \
                        tiling_new["BL1_shape"][1] * tiling_new["CL0_matrix"][0]
                        self.unzip_parameters["compress_tiling"][0] = \
                        tiling_new["BL1_shape"][0] // CUBE_MKN[w_dtype]['mac'][1]

            def tiling_l1_shape_get():
                """
                get L1_shape for tiling.

                """
                if tiling_new["AL1_shape"] == []:
                    tiling["AL1_shape"] = []
                else:
                    tiling["AL1_shape"] = tiling_new["AL1_shape"][0:4]
                    tiling["AL1_shape"][0] = int(tiling["AL1_shape"][0] / \
                        (((shape_w_nc1hwc0[2] - 1)*ConvParam.dilate_h + 1)*((\
                        shape_w_nc1hwc0[3] - 1)*ConvParam.dilate_w + 1) * \
                        CUBE_MKN[w_dtype]['mac'][1]))
                    if tiling["AL1_shape"][0] == 0:
                        tiling["AL1_shape"][0] = 1
                    if c0_optim_flg:
                        tiling["AL1_shape"][0] = 1

                if tiling_new["BL1_shape"] == [] or \
                        tiling_new["BL1_shape"] is None:
                    tiling["BL1_shape"] = tiling_new["BL1_shape"]
                else:
                    tiling["BL1_shape"] = tiling_new["BL1_shape"][0:2]
                    tiling["BL1_shape"][0] = int(tiling["BL1_shape"][0] / \
                        (shape_w_nc1hwc0[2]*shape_w_nc1hwc0[3]*CUBE_MKN[
                            w_dtype]['mac'][1]))
                    if c0_optim_flg:
                        tiling["BL1_shape"][0] = 1

            def handle_v100_tiling():
                # if the scene is weight compress, the high 8 bit
                # of fusion_type is 2
                if self.unzip_parameters.get("weight_zip_flag") and \
                cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES"):
                    compress_fusion_flag = self.unzip_parameters.get(
                        "compress_flag")
                    compress_fusion_flag = compress_fusion_flag << 8
                    fusion_type_new = fusion_type + compress_fusion_flag
                else:
                    fusion_type_new = fusion_type
                in_mem = list(map(int, self._input_memory_type))
                out_mem = list(map(int, self._output_memory_type))
                pooling_shape = [0, 0]
                pooling_stride = [0, 0]
                if self.conv_pool_fused_flag:
                    pooling_shape = [POOLING_WINDOW, POOLING_WINDOW]
                    pooling_stride = [POOLING_STRIDE, POOLING_STRIDE]
                info_dict = {"op_type": 'conv2d',
                             "a_shape": fmap_shape_nc1hwc0,
                             "b_shape": shape_w_nc1hwc0,
                             "c_shape": c_shape,
                             "a_dtype": in_dtype,
                             "b_dtype": w_dtype,
                             "c_dtype": res_dtype,
                             "mad_dtype": mad_dtype,
                             "pad": [ConvParam.pad_w[0], ConvParam.pad_w[1], \
                             ConvParam.pad_h[0], ConvParam.pad_h[1]],
                             "stride": [ConvParam.stride_h, \
                             ConvParam.stride_w],
                             "dilation": [ConvParam.dilate_h, \
                             ConvParam.dilate_w],
                             "group": 1,
                             "bias_flag": bias_flag,
                             "fused_coefficient": \
                                [0, 0, self._fused_double_operand_num],
                             "fused_channel_wise": [0, 0, 0],
                             "in_fm_memory_type": in_mem,
                             "out_fm_memory_type": out_mem,
                             "l1_fusion_type": self._l1_fusion_type,
                             "fusion_type": fusion_type_new,
                             "reserved_ub": reserved_ub,
                             "fused_ub_cl0": fused_ub_cl0,
                             "kernel_name": ConvParam.kernel_name,
                             "pooling_shape": pooling_shape,
                             "pooling_stride": pooling_stride}

                tiling_new = get_tiling(info_dict)
                return tiling_new

            def get_default_tiling():

                def fmap2_read_select(m_target):
                    elewise_read_select_flg = False
                    for lop in self._op_graph.body_ops:
                        if "output_ub_4d" in lop["op"]:
                            elewise_read_select_flg = True
                    if elewise_read_select_flg:
                        tiling_n = 2
                        for m_ub_target in range(32, 0, -1):
                            hw_4d = (m_ub_target * m_bit_length['float16'])
                            bias_ub = 2 * tiling_n * m_bit_length["float16"] \
                            * m_bit_ratio["int32"]
                            deq_vector = tiling_n * m_bit_length["float16"] * \
                            m_bit_ratio["float16"]
                            #ub size for c_ub when min tiling
                            ub_4d = tiling_n * m_bit_length["float16"] * \
                            hw_4d * m_bit_ratio["float16"]
                            #max ub size for fmap2 read select in 5HD shape
                            ub_5d_reverved = tiling_n * \
                            m_bit_length["float16"] * w_out * 2 * \
                            m_bit_ratio["float16"]
                            #double out fused_coefficient is 5
                            max_ub_feature_map = 5 * ub_4d + ub_5d_reverved + \
                            deq_vector + bias_ub
                            if max_ub_feature_map <= self._ub_size:
                                break
                            if m_ub_target == 1 and max_ub_feature_map > \
                            self._ub_size:
                                err_man.raise_err_specific("conv2d", \
                                    "Min tiling still exceed ub buffer, "\
                                    + "when fmap2 read select in")
                    else:
                        m_ub_target = 32
                    return m_target if m_ub_target > m_target else m_ub_target

                def _handle_block_dim():
                    if self._convbn1_flag:
                        tiling["block_dim"] = tiling_new["block_dim"]
                    else:
                        tiling["block_dim"] = [1, 1, 1]
                    device_core_num = self._corenum
                    if (ConvParam.batch > 1) and (device_core_num > 1):
                        if ConvParam.batch <= device_core_num:
                            tiling["block_dim"][0] = ConvParam.batch
                        else:
                            for i in range(device_core_num, 0, -1):
                                if ConvParam.batch % i == 0:
                                    break
                            tiling["block_dim"][0] = i
                    else:
                        tiling["block_dim"][0] = 1
                    if self._l1_fusion_type in (0, 1):
                        tiling["block_dim"] = [1, 1, 1]

                tiling = {}
                config = CUBE_MKN[w_dtype]
                ci0 = config['mac'][1]
                l1_buffer_size = self._l1_size
                m_bit_length = {"float32": 32, "float16": 16,
                                "uint8": 8, "int8": 8, "uint4": 4, "int4": 4}
                m_bit_ratio = {"int32": 4, "float32": 4, "float16": 2,
                               "uint8": 1, "int8": 1,
                               "uint4": 1.0 / 2, "int4": 1.0 / 2}
                input_data_type = in_dtype
                w_out = ConvParam.w_out

                for m_target in range(32, 0, -1):
                    tmp1 = ((m_target*m_bit_length['float16']) + \
                        w_out - 1) // w_out
                    tmp2 = ((tmp1*ConvParam.stride_h) + \
                        kh_dilate)*ConvParam.w_in
                    max_feature_map = 1*ci0*tmp2*2*m_bit_ratio[input_data_type]
                    if max_feature_map < l1_buffer_size:
                        break
                tiling_m = fmap2_read_select(m_target)
                tiling_k = 1
                tiling_n = 2
                tiling["AL1_shape"] = [1]
                if self.unzip_parameters.get("weight_zip_flag") and \
                cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES"):
                    tiling["BL1_shape"] = [1]
                else:
                    tiling["BL1_shape"] = None

                if w_dtype == "int8":
                    c_0 = 32
                else:
                    c_0 = 16
                tiling["AL0_matrix"] = [tiling_m, tiling_k, 16, c_0]
                tiling["BL0_matrix"] = [tiling_k, tiling_n, 16, c_0]
                tiling["CL0_matrix"] = [tiling_n, tiling_m, 16, 16]
                tiling["CUB_matrix"] = [tiling_n, tiling_m, 16, 16]
                tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                                    'BL1_pbuffer': 1,
                                                    'AL0_pbuffer': 1,
                                                    'BL0_pbuffer': 1,
                                                    'CL0_pbuffer': 1,
                                                    'CUB_pbuffer': 1,
                                                    'UBG_pbuffer': 1}
                tiling["A_overhead_opt_flag"] = False
                tiling["B_overhead_opt_flag"] = False
                tiling["scale_drq_split_flag"] = True
                tiling["bias_split_flag"] = True
                if self._l1_fusion_type == 1:
                    tiling["AL1_shape"] = []
                if self._input_memory_type[0] == 1:
                    tiling["AL1_shape"] = []
                _handle_block_dim()
                return tiling

            def get_reserved_ub():
                reserved_ub = 0
                tiling_type = TILING_INSTANCE.get_tiling_type()
                elewise_read_select_flg = False
                for lop in self._op_graph.body_ops:
                    if "output_ub_4d" in lop["op"]:
                        elewise_read_select_flg = True
                        break
                if elewise_read_select_flg and \
                (tiling_type != "atc_tuning_tiling" or \
                    tiling_type != "tuning_tiling"):
                    #reserved ub for lx fusion when element read select in
                    reserved_ub = reserved_ub + 2*c_shape[3]*c_shape[1]*2*16
                return reserved_ub

            def _config_tiling(tiling):
                if tiling_ok_flag and \
                    not ConvParam.tiling_query_param.get("default_tiling"):
                    tiling["AL0_matrix"] = tiling_new["AL0_matrix"][0:4]
                    tiling["CL0_matrix"] = tiling_new["CL0_matrix"][0:5]
                    tiling["CUB_matrix"] = tiling_new["CUB_matrix"][0:4]
                    tiling["A_overhead_opt_flag"] = \
                        tiling_new["A_overhead_opt_flag"]
                    tiling["B_overhead_opt_flag"] = \
                        tiling_new["B_overhead_opt_flag"]

                    if tiling_new["BL0_matrix"] == []:
                        tiling["BL0_matrix"] = []
                    else:
                        tiling["BL0_matrix"] = tiling_new["BL0_matrix"][0:4]

                    tiling["manual_pingpong_buffer"] = tiling_new[
                        "manual_pingpong_buffer"]
                    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]

                    tiling_l1_shape_get()

                    tiling["block_dim"] = tiling_new["block_dim"]
                    tiling["scale_drq_split_flag"] = False
                    tiling["bias_split_flag"] = False
                    tiling['CUB_channel_wise_flag'] = tiling_new[
                        'CUB_channel_wise_flag'] if 'CUB_channel_wise_flag' in \
                                                 tiling_new else None
                else:
                    tiling = get_default_tiling()

                if self._l1_fusion_type == 1 or self._l1_fusion_type == 0:
                    tiling["A_overhead_opt_flag"] = False
                    tiling["B_overhead_opt_flag"] = False

                return tiling

            fmap_shape_nc1hwc0 = ConvParam.tiling_query_param[
                "fmap_shape_nc1hwc0"]
            shape_w_nc1hwc0 = ConvParam.tiling_query_param["shape_w_nc1hwc0"]
            if self._dynamic_mode:
                fmap_shape_nc1hwc0 = list(fmap_shape_nc1hwc0)
                shape_w_nc1hwc0 = list(shape_w_nc1hwc0)
            else:
                fmap_shape_nc1hwc0 = list(map(int, fmap_shape_nc1hwc0))
                shape_w_nc1hwc0 = list(map(int, shape_w_nc1hwc0))
            in_dtype = ConvParam.tiling_query_param["in_dtype"]
            w_dtype = ConvParam.tiling_query_param["w_dtype"]
            res_dtype = ConvParam.tiling_query_param["res_dtype"]
            mad_dtype = ConvParam.tiling_query_param["mad_dtype"]
            bias_flag = ConvParam.tiling_query_param["bias_flag"]
            fusion_type = self._op_graph.fusion_type
            w_out = ConvParam.w_out
            h_out = ConvParam.h_out

            c_ub_shape = list(tensor_map["c_ub"].shape)
            c_shape = [c_ub_shape[0], c_ub_shape[1], \
                       h_out, w_out, c_ub_shape[3]]
            if self._dynamic_mode:
                c_shape = list(c_shape)
            else:
                c_shape = list(map(int, c_shape))

            reserved_ub = get_reserved_ub()
            fused_ub_cl0 = get_fused_ub_cl0()

            if is_support_v200() and ConvParam.res_dtype == "int32":
                def coeff(dtype, base):
                    pattern = re.compile(r'[a-z]*(\d+)')
                    base_res = pattern.match(base)
                    dtype_res = pattern.match(dtype)
                    if not base_res:
                        err_man.raise_err_specific("conv2d", \
                            ("base(%s) of coeff not match pattern" % base))
                    if not dtype_res:
                        err_man.raise_err_specific("conv2d", \
                            ("x(%s) of coeff not match pattern" % dtype))
                    return int_ceil_div(int(dtype_res.group(1)),
                                        int(base_res.group(1)))

                def calc_coeff_l0c_to_ub():
                    if self._v200_data_flow_type == \
                       DataFlowType.S16EltWiseS8S16:
                        # cannot be reused
                        return coeff('float16', 'int8')
                    if self._v200_data_flow_type == DataFlowType.S16EltWiseS8:
                        # compensation(reuse part)
                        return coeff('float16', 'int8') - 1
                    if self._v200_data_flow_type in (DataFlowType.S32toS8,
                                                     DataFlowType.S32toS16):
                        pass
                    return 0

                def handle_v200_tiling(fused_coefficient, fused_channel_wise, \
                        reserved_ub):
                    # if the scene is weight compress, the high 8 bit
                    # of fusion_type is 2
                    if self.unzip_parameters.get("weight_zip_flag") and \
                    cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES"):
                        compress_fusion_flag = self.unzip_parameters.get(
                            "compress_flag")
                        compress_fusion_flag = compress_fusion_flag << 8
                        fusion_type_new = fusion_type + compress_fusion_flag
                    else:
                        fusion_type_new = fusion_type
                    in_mem = list(map(int, self._input_memory_type))
                    out_mem = list(map(int, self._output_memory_type))

                    if res.op.tag == "conv_virtual_res":
                        c_dtype = "int8"
                    else:
                        c_dtype = res.dtype

                    if self._v200_data_flow_type == \
                            DataFlowType.V200_General_Fusion:
                        fused_coefficient = [
                            0, 0, self._fused_double_operand_num]
                        fused_channel_wise = [0, 0, 0]

                    if "bias_s16_flag" in tensor_map:
                        bias_s16_size = shape_w_nc1hwc0[0]*2
                    else:
                        bias_s16_size = 0

                    # v200 deq and req full load.
                    if self._v200_data_flow_type in (
                            DataFlowType.S16EltWiseS8,
                            DataFlowType.S16EltWiseS8S16):
                        # deqs16_scale + reqs16_scale + bias_s16
                        reserved_ub += shape_w_nc1hwc0[0]*16 + bias_s16_size
                    elif self._v200_data_flow_type == DataFlowType.S32toS16:
                        # deqs16_scale + bias_s16
                        reserved_ub += shape_w_nc1hwc0[0]*8 + bias_s16_size
                    else:
                        reserved_ub += shape_w_nc1hwc0[0]*8

                    info_dict = {"op_type": 'conv2d',
                                 "a_shape": fmap_shape_nc1hwc0,
                                 "b_shape": shape_w_nc1hwc0,
                                 "c_shape": c_shape,
                                 "a_dtype": in_dtype,
                                 "b_dtype": w_dtype,
                                 "c_dtype": c_dtype,
                                 "mad_dtype": mad_dtype,
                                 "pad": [ConvParam.pad_w[0], \
                                 ConvParam.pad_w[1], ConvParam.pad_h[0], \
                                 ConvParam.pad_h[1]],
                                 "stride": [ConvParam.stride_h, \
                                 ConvParam.stride_w],
                                 "dilation": [ConvParam.dilate_h, \
                                 ConvParam.dilate_w],
                                 "group": 1,
                                 "bias_flag": bias_flag,
                                 "fused_coefficient": fused_coefficient,
                                 "fused_channel_wise": fused_channel_wise,
                                 "in_fm_memory_type": in_mem,
                                 "out_fm_memory_type": out_mem,
                                 "l1_fusion_type": self._l1_fusion_type,
                                 "fusion_type": fusion_type_new,
                                 "reserved_ub": reserved_ub,
                                 "fused_ub_cl0": fused_ub_cl0,
                                 "kernel_name": ConvParam.kernel_name}
                    tiling_new = get_tiling(info_dict)

                    return tiling_new

                c_fused_coefficient = 0
                cub_channel_coefficient = 0
                c_dtype = res_dtype
                if self._v200_data_flow_type in (DataFlowType.S16EltWiseS8,
                                                 DataFlowType.S32toS8,
                                                 DataFlowType.S16EltWiseS8S16):
                    c_dtype = 'int8'
                elif self._v200_data_flow_type in (DataFlowType.S32toS16,):
                    c_dtype = 'float16'

                if self._v200_data_flow_type in \
                        (DataFlowType.S16EltWiseS8,
                         DataFlowType.S16EltWiseS8S16):
                    for buffer_fm2 in v200_fm2_cache_buffer:
                        fm2_dtype = buffer_fm2.dtype
                        c_fused_coefficient += coeff(fm2_dtype, c_dtype)

                if self._v200_data_flow_type in ( \
                                DataFlowType.S32toS16,
                                DataFlowType.S32toS8,
                                DataFlowType.S16EltWiseS8,
                                DataFlowType.S16EltWiseS8S16):
                    for buffer_data in v200_cache_buffer:
                        bias_or_scale_dtype = buffer_data.dtype
                        cub_channel_coefficient += coeff(bias_or_scale_dtype,
                                                         c_dtype)

                c_fused_coefficient += calc_coeff_l0c_to_ub()
                fused_coefficient = [0, 0, c_fused_coefficient]
                fused_channel_wise = [0, 0, cub_channel_coefficient]
                tiling_new = handle_v200_tiling(fused_coefficient,
                                                fused_channel_wise,
                                                reserved_ub)
            elif self._dynamic_mode in ("dynamic_hw", "dynamic_batch"):
                tiling_new = self._tiling_case
            else:
                tiling_new = handle_v100_tiling()

            tiling_ok_flag = check_tiling(tiling_new, w_dtype, \
                fmap_shape_nc1hwc0)
            tiling = {}
            tiling = _config_tiling(tiling)
            get_compress_tiling_shape(tiling_ok_flag)
            return tiling

        def double_operand_num_fetch():
            self._fused_double_operand_num = \
            self._op_graph.fused_double_operand_num

            if self._fused_flag and (not self.conv_pool_fused_flag):
                if len(self._op_graph.body_ops) < SMALL_GRAPH_OP_NUM:
                    analyze_data_dependence()
            else:
                self._fused_double_operand_num = 0
                if self._lhisi_data_flow_type:
                    cal_double_opprand_num()
                if self.conv_pool_fused_flag:
                    self._fused_double_operand_num += 2

        def _tiling_of_pooling():
            """
            change the tiling of maxpool
            """

            tiling['A_overhead_opt_flag'] = 0
            tiling['B_overhead_opt_flag'] = 0
            tiling['n_bef_batch_flag'] = 0
            tiling['CUB_channel_wise_flag'] = False
            if tiling["AL1_shape"] == []:
                tiling["AL1_shape"] = [1, pooling_out[0], 1, 1]
            al1_facter_pooling \
                = int_ceil_div(pooling_out[0], tiling["AL1_shape"][1])
            al1_facter_pooling \
                = int_ceil_div(pooling_out[0], al1_facter_pooling)
            m_l0c = int(int_ceil_div(POOLING_WINDOW * conv_w, 16))
            self._m_part_nums = tiling["CUB_matrix"][1]
            cube_m = tiling["CUB_matrix"][1]
            tiling["CUB_matrix"][1] = m_l0c
            tiling["CL0_matrix"][1] = m_l0c
            tiling["AL0_matrix"][0] = m_l0c
            # when padding, then only 2 block tile in m
            c_out = dim_map["filter_matrix_dim"][1]
            if pooling_padding[0] > 0 and tiling["block_dim"][2] > 2:
                block_dim[2] = 2
            if tiling["CL0_matrix"][0] * tiling["block_dim"][1] < c_out:
                tiling["block_dim"][1] = c_out // tiling["CL0_matrix"][0]
            return cube_m, al1_facter_pooling

        def analyze_data_dependence():
            """
            analyze data dependence in conv fusion.

            Returns
            -------
            """
            canot_reuse_map = {}
            for lop in self._op_graph.body_ops:
                if ("convolution" not in lop["op"]) or \
                        (self._fused_flag and (lop["op"] == "convolution_C")):
                    if len(lop["prev_op"]) > 1:
                        for tensor_list in lop["prev_op"]:
                            tensor = tensor_list["dst_buffer"]
                            tmp = list(map(lambda x: x["dst_buffer"],
                                           lop["prev_op"]))
                            tmp.remove(tensor)
                            canot_reuse_set = set(tmp)
                            if tensor in canot_reuse_map.keys():
                                canot_reuse_map[tensor].update(canot_reuse_set)
                            else:
                                canot_reuse_map[tensor] = canot_reuse_set
            buffer_reuse_set_list = []
            for lop in self._op_graph.body_ops + self._op_graph.input_ops:
                if lop["dst_buffer"] in canot_reuse_map:
                    if not buffer_reuse_set_list:
                        buffer_reuse_set = {lop["dst_buffer"]}
                        buffer_reuse_set_list.append(buffer_reuse_set)
                    else:
                        for tensor_list in buffer_reuse_set_list:
                            if not canot_reuse_map[
                                    lop["dst_buffer"]] & tensor_list:
                                tensor_list.add(lop["dst_buffer"])
                                if lop["prev_op"] == []:
                                    tensor_list.remove(lop["dst_buffer"])

                                break
                        else:
                            buffer_reuse_set = {lop["dst_buffer"]}
                            buffer_reuse_set_list.append(buffer_reuse_set)

            if self._convbn1_flag:
                self._fused_double_operand_num = len(buffer_reuse_set_list)
            else:
                self._fused_double_operand_num = len(buffer_reuse_set_list) + 1

        def handle_max_pooling():
            """
             handle compute of pooling
            """
            input_5d_data = self._max_pool_tensor_map["input_5d_data"]
            trans_line_data = self._max_pool_tensor_map["trans_line_data"]
            trans_vn_node = self._max_pool_tensor_map["trans_vn_node"]
            sch[trans_line_data].compute_at(sch[res_c], m_outer_inner_outer)
            sch[trans_vn_node].compute_at(sch[res_c], m_outer_inner_outer)
            sch[input_5d_data].reused_by(trans_line_data)
            sch[trans_vn_node].reused_by(\
                self._max_pool_tensor_map["ub_reshape"])
            sch[trans_line_data].emit_insn(
                trans_line_data.op.axis[0], "dma_copy")
            sch[trans_vn_node].emit_insn(
                trans_vn_node.op.axis[0], "phony_insn")
            if double_buffer_flag["CUB_pbuffer"] == 2:
                sch[trans_vn_node].double_buffer()
            al1_nparts = int_ceil_div(pooling_out[0], al1_facter_pooling)
            offset_bound \
                = (block_tile * int_ceil_div(al1_nparts, block_dim[2])
                   * al1_facter_pooling
                   + m_outer_outer_outer_inner * al1_facter_pooling
                   + m_outer_outer_inner + 1) * POOLING_STRIDE \
                   - pooling_padding[0]

            input_5d_data_offset \
                = (block_tile * int_ceil_div(al1_nparts, block_dim[2])
                   * al1_facter_pooling
                   + m_outer_outer_outer_inner.var * al1_facter_pooling
                   + m_outer_outer_inner.var) * POOLING_STRIDE\
                  - pooling_padding[0]
            input_5d_data_offset_condition = tvm.any(
                tvm.all(m_outer_outer_outer_inner.var == 0,
                        m_outer_outer_inner.var == 0),
                tvm.all(m_outer_outer_outer_inner.var + \
                        m_outer_outer_inner.var != 0,
                        input_5d_data.op.axis[2] >
                        input_5d_data_offset))
            sch[input_5d_data].set_store_predicate(
                input_5d_data_offset_condition, partition=True)

            #  redefine tensor scope
            sch[trans_line_data].buffer_tile(
                (None, None),
                (None, None),
                (offset_bound, 1),
                (0, conv_w),
                (None, None),
            )
            sch[input_5d_data].buffer_tile(
                (None, None),
                (None, None),
                (None, POOLING_WINDOW),
                (None, None),
                (None, None),
            )

            max_pool_tensors = self._max_pool_tensor_map["max_pool_tensors"]
            sch[input_5d_data].compute_at(
                sch[res_c], m_outer_inner_outer)
            co_outer, co_inner = sch[input_5d_data].split( \
                input_5d_data.op.axis[-1], 16)
            sch[input_5d_data].reorder(co_outer,
                                       input_5d_data.op.axis[-3],
                                       input_5d_data.op.axis[-2],
                                       co_inner)
            sch[input_5d_data].emit_insn(
                input_5d_data.op.axis[0], 'vector_auto')
            for pooling_tensor in max_pool_tensors:
                sch[pooling_tensor].compute_at(
                    sch[res_c], m_outer_inner_outer)
                sch[pooling_tensor].emit_insn(
                    pooling_tensor.op.axis[0], 'vector_max')
                sch[pooling_tensor].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 16),
                    (1, 1),
                )
            ub_reshape = self._max_pool_tensor_map["ub_reshape"]
            sch[ub_reshape].compute_at(
                sch[res_c], m_outer_inner_outer)
            if double_buffer_flag["CUB_pbuffer"] == 2:
                sch[ub_reshape].double_buffer()
            sch[ub_reshape].emit_insn(
                ub_reshape.op.axis[0], 'vector_auto')

            def handle_v100_padding():
                pad_data = self._max_pool_tensor_map["max_pooling_pad_data"]
                pad_zero = pad_data.op.input_tensors[1]
                for tensor in (pad_data, pad_zero):
                    sch[tensor].storage_align(tensor.op.axis[3], 16, 0)
                    sch[tensor].buffer_align(
                        (1, 1),
                        (1, 1),
                        (1, 1),
                        (1, 16),
                        (1, 1),
                    )
                    sch[tensor].set_scope(cce.scope_ubuf)
                    sch[tensor].compute_at(sch[res_c], m_outer_inner_outer)

                sch[pad_data].emit_insn(pad_data.op.axis[0], \
                    'dma_copy', {"split_select": 1})
                sch[pad_zero].emit_insn(pad_zero.op.axis[0], \
                                        'vector_dup')
                sch[pad_data].reused_by(pad_zero)

            def handle_v200_padding():
                pad_data = self._max_pool_tensor_map["max_pooling_pad_data"]
                pad_top = self._max_pool_tensor_map["max_pooling_pad_top"]
                pad_bottom \
                    = self._max_pool_tensor_map["max_pooling_pad_bottom"]
                pad_left = self._max_pool_tensor_map["max_pooling_pad_left"]
                pad_right = self._max_pool_tensor_map["max_pooling_pad_right"]
                pad_vn = self._max_pool_tensor_map["max_pooling_pad_vn"]
                for tensor in (pad_data, pad_top, pad_bottom,
                               pad_left, pad_right, pad_vn):
                    sch[tensor].storage_align(tensor.op.axis[3], 16, 0)
                    sch[tensor].buffer_align(
                        (1, 1),
                        (1, 1),
                        (1, 1),
                        (1, 16),
                        (1, 1),
                    )
                    sch[tensor].set_scope(cce.scope_ubuf)
                    sch[tensor].compute_at(sch[res_c], m_outer_inner_outer)

                sch[pad_data].emit_insn(pad_data.op.axis[0],
                                        'dma_copy', {"split_select": 1})
                for stage in (pad_top, pad_bottom, pad_left, pad_right):
                    sch[stage].emit_insn(stage.op.axis[0],
                                         'vector_dup', {"split_select": 1})
                sch[pad_vn].emit_insn(pad_vn.op.axis[0], 'phony_insn')

                sch[pad_top].reused_by(pad_data)
                sch[pad_bottom].reused_by(pad_data)
                sch[pad_left].reused_by(pad_data)
                sch[pad_right].reused_by(pad_data)
                sch[pad_vn].reused_by(pad_data)

            if "max_pooling_pad_data" in self._max_pool_tensor_map.keys():
                if is_support_v200():
                    handle_v200_padding()
                else:
                    handle_v100_padding()


        def double_buffer():
            """
            double buffer.
            """
            def _l1_double_buffer():
                # al1
                if double_buffer_flag["AL1_pbuffer"] == 2:
                    sch[al1].double_buffer()
                    if self.conv_pool_fused_flag:
                        sch[al1].preload()
                # bl1
                if double_buffer_flag["BL1_pbuffer"] == 2:
                    sch[bl1].double_buffer()

            def _l0_double_buffer():
                # l0a
                if double_buffer_flag["AL0_pbuffer"] == 2:
                    sch[fmap_col].double_buffer()
                # l0b
                if double_buffer_flag["BL0_pbuffer"] == 2:
                    sch[bl0].double_buffer()
                # L0C
                if double_buffer_flag["CL0_pbuffer"] == 2:
                    sch[c_col].double_buffer()
                    if bias_preload_flag:
                        sch[bias_l0c].double_buffer()
                        sch[c_col_bias].double_buffer()
                        sch[bias_l0c].preload()
                        if bias_optimize_flag:
                            sch[bias_ub_brc].double_buffer()
                            sch[bias_ub_brc].preload()

            def _resub_double_buffer():
                # resUB
                if ConvParam.swrite_flag and \
                        double_buffer_flag["UBG_pbuffer"] == 2:
                    if swrite_onlyconv_flag and has_vector_flag:
                        sch[self._res_tensor.op.input_tensors[
                            0].op.input_tensors[0]].double_buffer()
                    else:
                        sch[self._res_tensor.op.input_tensors[
                            0]].double_buffer()
                if has_vector_flag and not ConvParam.swrite_flag:
                    if double_buffer_flag["UBG_pbuffer"] == 2:
                        if self._fused_flag:
                            sch[res_ub].double_buffer()
                        else:
                            sch[self._res_tensor.op.input_tensors[
                                0]].double_buffer()

            _l1_double_buffer()
            _l0_double_buffer()

            if self._dynamic_mode and self._fused_flag:
                convolution_C = [lop["dst_buffer"] for lop in self._op_graph.body_ops \
                                     if lop["op"] == "convolution_C"][0]
                res_ub_dynamic = self._op_graph.output_ops[0]["dst_buffer"]
                # c_ub reused by convolution_C res_ub
                sch[c_ub].reused_by(res_ub_dynamic, convolution_C)
                if double_buffer_flag["CUB_pbuffer"] == 2:
                    sch[res_ub_dynamic].double_buffer()
                    sch[convolution_C].double_buffer()
                if double_buffer_flag["UBG_pbuffer"] == 2:
                    sch[c_ub].double_buffer()
                    sch[convolution_C].double_buffer()
            if double_buffer_flag["CUB_pbuffer"] == 2:
                sch[c_ub].double_buffer()

            _resub_double_buffer()

            # conv_bn1
            if double_buffer_flag["CUB_pbuffer"] == 2 and self._convbn1_flag:
                sch[d_pad].double_buffer()
                sch[tensor_map['cast_1']].double_buffer()

        def intrin_mapping(weight, tiling):
            """
            intrin mapping.
            """

            def bias_intrin_mapping():
                """
                bias intrin mapping.
                """
                if "bias" in tensor_map.keys():
                    if bias_preload_flag:
                        sch[bias_l0c].reused_by(c_col_bias, c_col)
                        sch[c_col_bias].emit_insn(c_col_bias.op.axis[0],
                                                  'phony_insn')
                    else:
                        sch[bias_l0c].pragma(bias_l0c.op.axis[0],
                                             'reuse_output', 1)
                        sch[c_col_bias].pragma(c_col_bias.op.axis[0],
                                               'replace_output', 1)
                        sch[c_col_bias].pragma(c_col_bias.op.axis[0], 'empty')
                        sch[c_col].pragma(k_outer_inner, 'replace_output', 1)

                    if bias_optimize_flag:
                        _, _ = sch[bias_l0c].split(\
                            bias_l0c.op.axis[2], 16)
                        sch[bias_l0c].emit_insn(\
                            bias_l0c.op.axis[1], 'dma_copy')
                        sch[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
                        sch[bias_ub_brc].emit_insn(bias_ub_brc.op.axis[0],
                                                   'vector_auto')
                    else:
                        sch[bias_l0c].emit_insn(\
                            bias_l0c.op.axis[1], 'dma_copy')
                        sch[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')

            def config_setfmatrix(setfmatrix_dict):
                def handle_valid_shape(setfmatrix_dict):
                    if self._valid_shape:
                        setfmatrix_dict["conv_fm_h"] = self._valid_shape[2]
                        if self._input_memory_type[0] == 1:
                            setfmatrix_dict["conv_fm_offset_h"] = \
                            ConvParam.fusion_para.get("slice_offset")[2]
                    elif self._aipp_fuse_flag:
                        setfmatrix_dict["conv_fm_h"] = al1.shape[2].value
                    else:
                        setfmatrix_dict["conv_fm_h"] = fmap.shape[2]

                    return setfmatrix_dict

                def al1_emit_insn():
                    if self._input_memory_type[0] == 1:
                        sch[al1].emit_insn(al1.op.axis[0], 'phony_insn')
                    elif self._aipp_fuse_flag:
                        aipp_map = al1.op.attrs
                        aipp_map['spr_0'] = al1.op.axis[0]
                        aipp_map_res = {"spr_0": al1.op.axis[0],
                                     "spr_1": aipp_map["spr_1"],
                                     "spr_2": aipp_map["spr_2"],
                                     "spr_3": aipp_map["spr_3"],
                                     "spr_4": aipp_map["spr_4"],
                                     "spr_5": aipp_map["spr_5"],
                                     "spr_6": aipp_map["spr_6"],
                                     "spr_7": aipp_map["spr_7"],
                                     "spr_8": aipp_map["spr_8"],
                                     "spr_9": aipp_map["spr_9"],
                                     "src_image_h": aipp_map["src_image_h"],
                                     "src_image_w": aipp_map["src_image_w"],
                                     "input_format": aipp_map["input_format"],
                                     "load_start_pos_h": \
                                        aipp_map["load_start_pos_h"],
                                     "load_start_pos_w": \
                                        aipp_map["load_start_pos_w"],
                                     "crop_size_h": aipp_map["crop_size_h"],
                                     "crop_size_w": aipp_map["crop_size_w"]}
                        sch[al1].emit_insn(al1.op.axis[1], \
                                           "load_image_to_cbuf", aipp_map_res)
                    else:
                        sch[al1].emit_insn(al1.op.axis[0], \
                            'dma_copy', {"mem_align": 1})
                        if self._l1_fusion_type in (0, 1):
                            sch[al1].pragma(al1.op.axis[0], 'jump_data',  1)

                def update_conv_fm_w(setfmatrix_dict):
                    """
                    for aipp+conv fuse, because of the crop function of loadimage
                    the true image size for conv is the size after crop
                    """
                    if self._aipp_fuse_flag:
                        setfmatrix_dict["conv_fm_w"] = al1.shape[3].value
                    else:
                        setfmatrix_dict["conv_fm_w"] = fmap.shape[3]

                    return setfmatrix_dict

                def _dynamic_im2col_v2():
                    if strideh_opti_flag:
                        strideh_update = 1
                    else:
                        strideh_update = c_ub.op.attrs['stride'][0]
                    im2col_attr = {
                        'set_fmatrix': 1,
                        'conv_kernel_h': c_ub.op.attrs['kernel_h'],
                        'conv_kernel_w': c_ub.op.attrs['kernel_w'],
                        'conv_padding_top': c_ub.op.attrs['padding'][0],
                        'conv_padding_bottom': c_ub.op.attrs['padding'][1],
                        'conv_padding_left': c_ub.op.attrs['padding'][2],
                        'conv_padding_right': c_ub.op.attrs['padding'][3],
                        'conv_stride_h': strideh_update,
                        'conv_stride_w': c_ub.op.attrs['stride'][1],
                        'conv_fm_c': fmap.shape[4]*fmap.shape[1],
                        'conv_fm_c1': fmap.shape[1],
                        'conv_fm_h': fmap.shape[2],
                        'conv_fm_w': fmap.shape[3],
                        'conv_fm_c0': fmap.shape[4],
                    }
                    if l0a_load2d_flag:
                        sch[al1].emit_insn(al1.op.axis[0], 'dma_copy')
                        sch[fmap_col].emit_insn(new_fmap_col_axis[3], 'dma_copy')
                    else:
                        sch[al1].emit_insn(al1.op.axis[0], 'dma_copy', im2col_attr)
                        sch[fmap_col].emit_insn(new_fmap_col_axis[3], 'im2col_v2', im2col_attr)

                def add_pragma_for_aipp_fuse(setfmatrix_dict):
                    """
                    in aipp+conv fusion:
                    for YUV format, pragma conv_yuv_align is needed for height
                    offset revision
                    for YUV format and C0=4(1951), another pragma is used for
                    pad top offset revison
                    """
                    if self._aipp_fuse_flag \
                        and al1.op.attrs["input_format"] == "YUV420SP_U8":
                        setfmatrix_dict["conv_yuv_align"] = yuv_align
                        width_out = dim_map.get("out_img_height_width")[1]
                        c0_4 = \
                            ConvParam.tiling_query_param["fmap_shape_nc1hwc0"][-1]
                        if is_support_v200() and c0_4 <= 4 \
                            and __lcm(width_out, 16) > width_out:
                            setfmatrix_dict["conv_yuv_pad_align"] = yuv_pad_align
                    return setfmatrix_dict

                def _im2col_with_row_major(setfmatrix_dict):
                    setfmatrix_dict = {
                        "conv_kernel_h": ConvParam.filter_h,
                        "conv_kernel_w": ConvParam.filter_w,
                        "conv_padding_top": ConvParam.padding[0],
                        "conv_padding_bottom": ConvParam.padding[1],
                        "conv_padding_left": ConvParam.padding[2],
                        "conv_padding_right": ConvParam.padding[3],
                        "conv_stride_h": ConvParam.stride_h,
                        "conv_stride_w": ConvParam.stride_w,
                        "conv_dilation_h": ConvParam.dilate_h,
                        "conv_dilation_w": ConvParam.dilate_w}

                    if "offset_pad" in tensor_map.keys():
                        if tensor_map["offset_pad"] is not None:
                            sch[res_c].emit_insn(noo, 'set_padding_ex')

                    if strided_read_flag or sread_strideh_flag or \
                        self._aipp_fuse_flag:
                        # AL1 is the true feature map while
                        # fmap in tensor_map is the input of strided_read.
                        setfmatrix_dict[
                            "conv_fm_c"] = al1.shape[1]*al1.shape[4]
                    elif sread_load2d_flag:
                        setfmatrix_dict[
                            "conv_fm_c"] = al1.shape[1]*al1.shape[3]
                    else:
                        setfmatrix_dict[
                            "conv_fm_c"] = fmap.shape[1]*fmap.shape[4]

                    setfmatrix_dict = handle_valid_shape(setfmatrix_dict)
                    setfmatrix_dict = update_conv_fm_w(setfmatrix_dict)
                    al1_emit_insn()

                    setfmatrix_dict = add_pragma_for_aipp_fuse(setfmatrix_dict)

                    return setfmatrix_dict

                if self._dynamic_mode:
                    _dynamic_im2col_v2()
                else:
                    setfmatrix_dict = _im2col_with_row_major(setfmatrix_dict)

                return setfmatrix_dict


            def _lhisi_intrin():
                if self._lhisi_dequant_quant_para['deq_vector']:
                    sch[c_ub].pragma(c_ub.op.axis[2], 'deq_scale', 'vector')
                else:
                    sch[c_ub].pragma(c_ub.op.axis[0], 'deq_scale', 'scalar')

                if self._lhisi_data_flow_type == DataFlowTypeLhisi.S32toFp16:
                    sch[res_c].emit_insn(c_pragma_axis, 'dma_copy')
                if "c_ub_res" in tensor_map:
                    c_ub_res = tensor_map["c_ub_res"]
                    sch[c_ub_res].emit_insn(c_ub_res.op.axis[0], 'vector_auto')

            def get_compress_parameters():
                """
                function: get compress parameters from different chip
                C200(1951_PG1): unzip engines=4;unzip_channels=2;
                                unzip_max_ratios=8X(64);unzip_is_tight=1
                H200(lhisi_ES): unzip engines=1;unzip_channels=4;
                                unzip_max_ratios=8X(64);unzip_is_tight=1
                these parameters as follows:
                unzip_engines: number of  decompression engines of chip
                unzip_max_ratios: (8x: 64 4x: 32)
                unzip_channels: numbers of decompress engines
                unzip_is_tight: compress model, tightly or not
                :return: unzip_engines, unzip_max_ratios,
                         unzip_channels,unzip_is_tight
                """
                unzip_engines, unzip_max_ratios, \
                unzip_channels, unzip_is_tight = get_soc_spec("UNZIP")

                return unzip_engines, unzip_channels, \
                       unzip_max_ratios, unzip_is_tight

            def get_byte_size(weight_dtype, weight_shape):
                """
                function: calculate  byte size, according to weight dtype
                :param weight_dtype: the dtype of weight
                :param weight_shape:
                :return: if weight dtype is float16, size * 2;
                         if weight dtype is int 8, byte size * 1.
                weight_shape_size:
                weight_compress_tiling_size:
                """
                weight_compress_tiling_size = int( \
                    self.unzip_parameters["compress_tiling"][0] * \
                    self.unzip_parameters["compress_tiling"][1] * \
                    self.unzip_parameters["compress_tiling"][2] * \
                    self.unzip_parameters["compress_tiling"][3])
                weight_shape_size = int( \
                    weight_shape[0] *
                    weight_shape[1] *
                    weight_shape[2] *
                    weight_shape[3])
                if weight_dtype == "int8":
                    return weight_compress_tiling_size, weight_shape_size
                elif weight_dtype == "float16":
                    return weight_compress_tiling_size*2, weight_shape_size*2
                else:
                    raise RuntimeError(
                        "weight compress only support int8 and float16")

            def get_all_factor(k_or_n_number):
                """
                function: Calculate the common divisor of the k-axis or n-axis
                :param k_or_n_number: number of bl0_matrix[0] or bl0_matrix[1]
                :return: k_or_n_number_list
                """
                k_or_n_number_list = []
                for i in range(1, k_or_n_number + 1):
                    if k_or_n_number % i == 0:
                        k_or_n_number_list.append(i)
                return k_or_n_number_list
            def get_proper_factor(k_or_n_number, block_size_last):
                """
                function: get proper number of k_or_n_number,
                to calculate proper basic compressed block size
                :param k_or_n_number: number of bl0_matrix[0] or bl0_matrix[1]
                :param block_size_last: all size of bl0_matrix or last 3 of it
                :return: proper number k or n
                """
                k_or_n_number_list = get_all_factor(k_or_n_number)
                for number in k_or_n_number_list:
                    if int(number * block_size_last) <= \
                            self.unzip_parameters.get("max_block_size"):
                        cur_number = number
                return cur_number

            def get_compress_index_size(
                weight_shape, mode_index_size, weight_shape_size, block_size
            ):
                compress_tiling_k = self.unzip_parameters["compress_tiling"][0]
                compress_tiling_n = self.unzip_parameters["compress_tiling"][1]
                n_remainders = weight_shape[1] % compress_tiling_n
                compress_tiling_k_frac = weight_shape[0] // compress_tiling_k
                compress_index_size = 0
                if int(n_remainders) == 0:
                    compress_index_size = weight_shape_size // \
                        block_size*mode_index_size
                else:
                    first_weight_size = weight_shape[0]*(
                        weight_shape[1] - n_remainders
                        )*weight_shape[2]*weight_shape[3]
                    compress_index_size += first_weight_size // \
                        block_size*mode_index_size
                    compress_index_size += compress_tiling_k_frac*int_ceil_div(
                        n_remainders * compress_tiling_k * weight_shape[2] * \
                        weight_shape[3], block_size) * mode_index_size
                return compress_index_size

            def calculate_compress_index(weight, mode_index_size):
                weight_shape = weight.shape
                weight_dtype = weight.dtype
                weight_compress_tiling_size, weight_shape_size = \
                    get_byte_size(weight_dtype, weight_shape)
                if weight_compress_tiling_size <= self.unzip_parameters.get(
                        "max_block_size"):
                    compress_index_size = get_compress_index_size(
                        weight_shape, mode_index_size, weight_shape_size,
                        weight_compress_tiling_size)
                    block_size = weight_compress_tiling_size
                    tiling_block_frac = 1
                else:
                    if weight_compress_tiling_size % self.unzip_parameters.get(
                            "max_block_size") == 0:
                        compress_index_size = get_compress_index_size(
                            weight_shape, mode_index_size, weight_shape_size,
                             self.unzip_parameters.get("max_block_size"))
                        block_size = self.unzip_parameters.get(
                            "max_block_size")
                        tiling_block_frac = \
                            weight_compress_tiling_size // block_size
                    else:
                        block_size_all = weight_compress_tiling_size
                        block_size_last3 = int(block_size_all // \
                        self.unzip_parameters["compress_tiling"][0])
                        if block_size_last3 <= int(
                                self.unzip_parameters["max_block_size"]):
                            kb_value = get_proper_factor( \
                            int(self.unzip_parameters["compress_tiling"][0]),
                            block_size_last3)
                            block_size = kb_value * block_size_last3
                            tiling_block_frac = \
                                weight_compress_tiling_size // block_size
                            compress_index_size = get_compress_index_size(
                                weight_shape, mode_index_size,
                                weight_shape_size, block_size)
                        else:
                            block_size_last2 = int(block_size_all // ( \
                            self.unzip_parameters["compress_tiling"][0] * \
                            self.unzip_parameters["compress_tiling"][1]))
                            nb_value = get_proper_factor( \
                            int(self.unzip_parameters["compress_tiling"][1]),
                            block_size_last2)
                            block_size = nb_value * block_size_last2
                            tiling_block_frac = \
                                weight_compress_tiling_size // block_size
                            compress_index_size = get_compress_index_size(
                                weight_shape, mode_index_size,
                                weight_shape_size, block_size)
                return block_size, compress_index_size, tiling_block_frac

            def get_basic_compress_block(weight, unzip_is_tight):
                """
                function: Calculate the block size and weight index size,
                          according to weight shape and tiling
                :param weight: weight shape, such as [18, 8, 16, 32]
                :param unzip_is_tight: compress model, tightly or not
                :return:
                block_size: basic block size required by compression algorithm
                compress_index_size: Index size of compressed data
                index: unzip command, command axis on bl1 or bl0
                """
                if unzip_is_tight:
                    block_size, compress_index_size, tiling_block_frac = \
                    calculate_compress_index(weight, \
                    self.unzip_parameters.get("compact_mode_index_size"))

                else:
                    block_size, compress_index_size, tiling_block_frac = \
                    calculate_compress_index(weight, \
                    self.unzip_parameters.get("uncompact_mode_index_size"))
                return block_size, compress_index_size, tiling_block_frac

            def emit_unzip(weight, tiling):
                """
                function: unzip emit_insn
                :param weight: weight
                :param tiling: tiling
                :return:
                """
                weight_shape = weight.shape
                compress_tiling_k = self.unzip_parameters["compress_tiling"][0]
                compress_tiling_n = self.unzip_parameters["compress_tiling"][1]
                compress_tiling_n_frac = weight_shape[0] // compress_tiling_k
                block_num = dim_map['filter_matrix_dim'][0] * \
                            dim_map['filter_matrix_dim'][1] * \
                            dim_map['filter_matrix_dim'][2] * \
                            dim_map['filter_matrix_dim'][3]
                if bl1_factor[0] != 1 and dim_map['out_img_shape'][2] > \
                   tiling["CL0_matrix"][1]*16:
                    block_num = block_num // \
                    int_ceil_div(dim_map['out_img_shape'][1], tiling["CL0_matrix"][0])
                if tiling["BL1_shape"] is None:
                    unzip_engines, unzip_channels, unzip_max_ratios, \
                    unzip_is_tight = get_compress_parameters()
                    block_size, compress_index_size, tiling_block_frac = \
                    get_basic_compress_block(weight, unzip_is_tight)
                    conflict = tvm.make.Call("int32", "tvm_tuple", ( \
                               int(block_size),
                               int(compress_index_size),
                               unzip_is_tight,
                               unzip_engines,
                               unzip_channels,
                               unzip_max_ratios,
                               compress_tiling_k,
                               compress_tiling_n), \
                               tvm.expr.Call.Intrinsic, None, 0)
                    sch[bl0].pragma(bl0.op.axis[0],
                                    "json_info_compress_parameters", conflict)
                    sch.set_var_range(ConvParam.compress_index_shape, \
                                      int(compress_index_size),
                                      int(compress_index_size))
                    sch.set_var_range(ConvParam.compress_tiling_n,
                                      int(compress_tiling_n),
                                      int(compress_tiling_n))
                    sch.set_var_range(ConvParam.compress_tiling_k,
                                      int(compress_tiling_k),
                                      int(compress_tiling_k))
                    sch.set_var_range(ConvParam.compress_tiling_n_frac,
                                      int(compress_tiling_n_frac),
                                      int(compress_tiling_n_frac))
                    sch.set_var_range(ConvParam.compress_tiling_frac,
                                      int(tiling_block_frac),
                                      int(tiling_block_frac))
                    sch[bl0].emit_insn(bl0.op.axis[0], "unzip",
                                       {"block_size": block_size, \
                                        "compress_mode": unzip_is_tight,
                                        "hoist_axis": out_extract_axis,
                                        "block_num": block_num // block_size})
                elif tiling["BL1_shape"] == []:
                    unzip_engines, unzip_channels, unzip_max_ratios, \
                    unzip_is_tight = get_compress_parameters()
                    block_size, compress_index_size, tiling_block_frac = \
                    get_basic_compress_block(weight, unzip_is_tight)
                    conflict = tvm.make.Call("int32", "tvm_tuple", ( \
                        int(block_size),
                        int(compress_index_size),
                        unzip_is_tight,
                        unzip_engines,
                        unzip_channels,
                        unzip_max_ratios,
                        compress_tiling_k,
                        compress_tiling_n), \
                        tvm.expr.Call.Intrinsic, None, 0)
                    sch[bl1].pragma(bl1.op.axis[0],
                                    "json_info_compress_parameters", conflict)
                    sch.set_var_range(ConvParam.compress_index_shape, \
                                      int(compress_index_size),
                                      int(compress_index_size))
                    sch.set_var_range(ConvParam.compress_tiling_n,
                                      int(compress_tiling_n),
                                      int(compress_tiling_n))
                    sch.set_var_range(ConvParam.compress_tiling_k,
                                      int(compress_tiling_k),
                                      int(compress_tiling_k))
                    sch.set_var_range(ConvParam.compress_tiling_n_frac,
                                      int(compress_tiling_n_frac),
                                      int(compress_tiling_n_frac))
                    sch.set_var_range(ConvParam.compress_tiling_frac,
                                      int(tiling_block_frac),
                                      int(tiling_block_frac))
                    sch[bl1].emit_insn(bl1.op.axis[0], "unzip",
                                       {"block_size": block_size,
                                        "compress_mode": unzip_is_tight,
                                        "hoist_axis": out_extract_axis,
                                        "block_num": block_num // block_size})
                    sch[bl0].emit_insn(bl0.op.axis[0], 'dma_copy')
                else:
                    unzip_engines, unzip_channels, unzip_max_ratios, \
                    unzip_is_tight = get_compress_parameters()
                    block_size, compress_index_size, tiling_block_frac = \
                    get_basic_compress_block(weight, unzip_is_tight)
                    conflict = tvm.make.Call("int32", "tvm_tuple", ( \
                        int(block_size),
                        int(compress_index_size),
                        unzip_is_tight,
                        unzip_engines,
                        unzip_channels,
                        unzip_max_ratios,
                        compress_tiling_k,
                        compress_tiling_n), \
                        tvm.expr.Call.Intrinsic, None, 0)
                    sch[bl1].pragma(bl1.op.axis[0],
                                    "json_info_compress_parameters", conflict)
                    sch.set_var_range(ConvParam.compress_index_shape, \
                                      int(compress_index_size),
                                      int(compress_index_size))
                    sch.set_var_range(ConvParam.compress_tiling_n,
                                      int(compress_tiling_n),
                                      int(compress_tiling_n))
                    sch.set_var_range(ConvParam.compress_tiling_k,
                                      int(compress_tiling_k),
                                      int(compress_tiling_k))
                    sch.set_var_range(ConvParam.compress_tiling_n_frac,
                                      int(compress_tiling_n_frac),
                                      int(compress_tiling_n_frac))
                    sch.set_var_range(ConvParam.compress_tiling_frac,
                                      int(tiling_block_frac),
                                      int(tiling_block_frac))
                    sch[bl1].emit_insn(bl1.op.axis[0], "unzip",
                                       {"block_size": block_size,
                                        "compress_mode": unzip_is_tight,
                                        "hoist_axis": out_extract_axis,
                                        "block_num": block_num // block_size})
                    sch[bl0].emit_insn(bl0.op.axis[0], 'dma_copy')

            def pragma_v100_quant():
                if not self._v200_data_flow_type and \
                                self._lhisi_data_flow_type is None:
                    sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
                else:
                    # for v200 dequant, performance will be better
                    # when emit_insn on axis[2]
                    if is_support_v200():
                        sch[c_ub].emit_insn(c_ub.op.axis[2], 'dma_copy')

            def handle_l0a_emit_insn():
                if l0a_load2d_flag:
                    sch[fmap_col].emit_insn(new_fmap_col_axis[3], 'dma_copy')
                else:
                    if strideh_opti_flag:
                        setfmatrix_dict["conv_stride_h"] = 1
                    sch[fmap_col_before].emit_insn(
                        fmap_col_before.op.axis[0], \
                            'set_fmatrix', setfmatrix_dict)
                    sch[fmap_col].emit_insn(new_fmap_col_axis[3], 'im2col')

            def get_weight_repeat_number():
                weight_repeat_load_num = 1
                if tiling["BL1_shape"] is None:
                    if tiling["BL0_matrix"] == []:
                        weight_repeat_load_num = 1
                    else:
                        weight_repeat_load_num = al1_factor[1]
                elif tiling["BL1_shape"] == []:
                    weight_repeat_load_num = 1
                else:
                    weight_repeat_load_num = al1_factor[1]
                sch[res_c].pragma(bido, "json_info_weight_repeat",
                                weight_repeat_load_num)

            def _handle_v200_emit_insn():
                if self._v200_data_flow_type == DataFlowType.S32toS8:
                    sch[c_ub_reform].emit_insn(c_ub_reform.op.axis[2], 'dma_copy')
                elif self._v200_data_flow_type in [DataFlowType.S16EltWiseS8,
                                                   DataFlowType.S16EltWiseS8S16]:
                    if "requant_s16_vaddrelu" in tensor_map:
                        relu_s16 = tensor_map["requant_s16_vaddrelu"]
                        sch[relu_s16].emit_insn(
                            relu_s16.op.axis[1], 'vector_addrelu')
                    elif "requant_s16_vadd" in tensor_map:
                        s16 = tensor_map["requant_s16_vadd"]
                        sch[s16].emit_insn(s16.op.axis[1], 'vector_add')
                    sch[c_ub_reform].emit_insn(c_ub_reform.op.axis[2],
                                               'vector_conv_vdeq')
                    if self._v200_data_flow_type == DataFlowType.S16EltWiseS8S16:
                        sch[res_remove_pad_u8].emit_insn(
                            res_remove_pad_u8.op.axis[0], 'dma_copy')
                        sch[res_remove_pad_s16].emit_insn(
                            res_remove_pad_s16.op.axis[0], 'dma_copy')
                        if not self._l0b_first_flag:
                            sch[res_c].emit_insn(c_outer_inner_inner, 'phony_insn')
                        else:
                            sch[res_c].emit_insn(batch_inner_inner, 'phony_insn')

            bias_intrin_mapping()
            setfmatrix_dict = {}
            setfmatrix_dict = config_setfmatrix(setfmatrix_dict)

            if self._dynamic_mode is None:
                handle_l0a_emit_insn()
                get_weight_repeat_number()

            if self.unzip_parameters.get("weight_zip_flag"):
                emit_unzip(weight, tiling)
            else:
                if tiling["BL1_shape"] is not None:
                    sch[bl1].emit_insn(bl1.op.axis[0], 'dma_copy')
                sch[bl0].emit_insn(bl0.op.axis[0], 'dma_copy')

            # pragma for v100 quant
            pragma_v100_quant()

            _handle_v200_emit_insn()

            if self._lhisi_data_flow_type:
                _lhisi_intrin()

            if has_vector_flag and self._fused_flag or \
            self._v200_data_flow_type in (DataFlowType.V200_General_Fusion,) \
            and res_c.op.tag != "quant":
                if res_c.op.name != 'conv_virtual_res':
                    sch[res_c].emit_insn(c_pragma_axis, 'dma_copy')

            if swrite_onlyconv_flag:
                sch[res_c].emit_insn(c_pragma_axis, 'dma_copy')

            mad_dict = {"mad_pattern": 2,
                        "k_outer": [k_outer_outer_outer_outer,
                                    k_outer_outer_outer_inner,
                                    k_outer_outer_inner]}
            if "bias" in tensor_map.keys():
                # only use this pragma in bias case
                mad_dict["init_bias"] = 1
            sch[c_col].emit_insn(cn_axis, 'mad', mad_dict)

        def _handle_deq_and_bias(c_result_ub_deq, c_ub_reform=None):
            if c_ub_reform is not None:
                sch[c_result_ub_deq].compute_inline()
                reform_outer, reform_inner = sch[c_ub_reform].split(
                    c_ub_reform.op.axis[3], nparts=2)
                sch[c_ub_reform].compute_at(sch[res_c], m_outer_inner_outer)
                sch[c_ub_reform].reorder(c_ub_reform.op.axis[0],
                                         c_ub_reform.op.axis[1],
                                         reform_outer,
                                         c_ub_reform.op.axis[2],
                                         reform_inner)
                sch[c_ub_reform].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, CUBE_MKN[c_result_ub_deq.dtype]["mac"][0]),
                    (1, CUBE_MKN[c_result_ub_deq.dtype]["mac"][2]))

        def checkout_quant_dequant(res, tensor_map):
            """
            checkout the fuse dataflow of conv + dequant + quant

            Returns
            -------
            the tensor_map with fuse dataflow
            """
            def check_dataflow(temp_tensor):
                if temp_tensor.op.tag == "conv_virtual_res":
                    self._lhisi_data_flow_type = DataFlowTypeLhisi.S32toFp16S8
                elif "dequant1" in temp_tensor.op.tag:
                    if not self._lhisi_data_flow_type:
                        self._lhisi_data_flow_type = \
                            DataFlowTypeLhisi.S32toFp16
                    if temp_tensor.op.tag == "dequant1_vector":
                        self._lhisi_dequant_quant_para['deq_vector'] = True
                elif "dequant2" in temp_tensor.op.tag:
                    self._lhisi_dequant_quant_para['deq_sqrt'] = True

                elif temp_tensor.op.tag == "dequant_relu":
                    self._lhisi_dequant_quant_para['deq_relu'] = True
                elif temp_tensor.op.tag == "quant":
                    if self._lhisi_data_flow_type != \
                            DataFlowTypeLhisi.S32toFp16S8:
                        if tensor_map['c_col'].dtype != "int32":
                            self._conv_quant_fused_flag = True
                        else:
                            self._lhisi_data_flow_type = \
                                DataFlowTypeLhisi.S32toS8
                    self._lhisi_dequant_quant_para['quant_round'] = \
                        temp_tensor.op.attrs["round_mode"]

            def handle_tensor(temp_tensor):
                nonlocal double_num
                nonlocal tensor_map
                if temp_tensor in tensor_visit_set:
                    return
                tensor_visit_set.add(temp_tensor)
                check_dataflow(temp_tensor)
                if len(temp_tensor.op.input_tensors) == 2 \
                        and "elewise" in temp_tensor.op.tag:
                    double_num = double_num + 1
                    if temp_tensor.op.input_tensors[0].op.name == \
                            "output_ub_4d" or \
                            temp_tensor.op.input_tensors[1].op.name == \
                            "output_ub_4d":
                        self._vector_read_select = True
                        double_num += 1
                        if temp_tensor.op.input_tensors[0].op.name == \
                                "output_ub_4d":
                            read_select_tensor = \
                                temp_tensor.op.input_tensors[0]
                            tensor_map = get_read_select_srctensor(
                                read_select_tensor, tensor_map)
                            temp_tensor = temp_tensor.op.input_tensors[1]
                        else:
                            read_select_tensor = \
                                temp_tensor.op.input_tensors[1]
                            tensor_map = get_read_select_srctensor(
                                read_select_tensor, tensor_map)
                    else:
                        input_tensor = temp_tensor.op.input_tensors[0]
                        if not input_tensor.op.input_tensors:
                            temp_tensor = temp_tensor.op.input_tensors[1]
                for tensor in  temp_tensor.op.input_tensors:
                    handle_tensor(tensor)

            double_num = 0
            temp_tensor = res
            tensor_visit_set = set()
            handle_tensor(temp_tensor)
            return double_num, tensor_map

        def handle_lhisi_fuse_res():
            if self._lhisi_data_flow_type == DataFlowTypeLhisi.S32toFp16:
                if "dequant_remove_pad" not in res.op.tag and \
                        "write_select" not in res.op.name and \
                        not ConvParam.swrite_dequant_flag:
                    c_ub_res = sch.cache_write(res_c, cce.scope_ubuf)
                    tensor_map["c_ub_res"] = c_ub_res

        def handle_lhisi_fuse_compute_at():
            for lop in self._op_graph.body_ops:
                if (("convolution" not in lop["op"]) or \
                    ("convolution_A" in lop["op"])) or \
                       (self._fused_flag and (lop["op"] == "convolution_C")):
                    if lop['dst_buffer'] == res or \
                            lop["op"] == "conv_vector_remove_pad":
                        continue
                    elif lop["op"] == "input_ub":
                        quant_input = tensor_map["quant_input"]
                        if quant_input.op.attrs["c_out"].value % 2 == 1:
                            self._schedule[lop["dst_buffer"]].compute_at(
                                self._schedule[res_c], m_outer_inner_outer)
                            self._lhisi_dequant_quant_para["quant_padding"] = \
                                True
                        else:
                            self._schedule[lop["dst_buffer"]].compute_inline()
                    elif lop["op"] == "dequant_remove_pad":
                        self._schedule[lop["dst_buffer"]].compute_inline()
                    else:
                        if lop["dst_buffer"].op.tag != "strided_read":
                            self._schedule[lop["dst_buffer"]].compute_at(
                                self._schedule[res_c], m_outer_inner_outer)
            if "c_ub_res" in tensor_map:
                c_ub_res = tensor_map["c_ub_res"]
                sch[c_ub_res].compute_at(sch[res_c], m_outer_inner_outer)

        def handle_v100_quant_input():

            def get_flag_and_read_map():
                elewise_flag = False
                dequant_flag = False
                tmp_read_map = []
                for nop in lop["next_op"]:
                    tmp_read_map.append(nop["dst_buffer"])
                    if "elewise" in nop["op"]:
                        elewise_flag = True
                    if "dequant1" in nop["op"] or "dequant2" in nop["op"]:
                        dequant_flag = True
                    if "output_ub_5d" in nop["op"]:
                        elewise_flag = True
                        output_ub_5d = nop["dst_buffer"]
                        #for index of operand in dma copy is nonlinear
                        sch[output_ub_5d].storage_align(\
                            sch[output_ub_5d].op.axis[2], \
                            output_ub_5d.shape[3].value * \
                            output_ub_5d.shape[4].value, 0)
                return elewise_flag, dequant_flag, tmp_read_map

            v100_cache_buffer = []
            for lop in self._op_graph.input_ops:
                elewise_flag, dequant_flag, tmp_read_map = \
                get_flag_and_read_map()
                if elewise_flag:
                    if "addr_type" in lop["dst_buffer"].op.attrs:
                        self._input_memory_type.append(
                            lop["dst_buffer"].op.attrs["addr_type"].value)
                        if lop["dst_buffer"].op.attrs["addr_type"].value \
                        == L1_FUSION_SCOPE:
                            sch[lop["dst_buffer"]].set_scope( \
                                cce.scope_cbuf_fusion)
                    else:
                        self._input_memory_type.append(DDR_SCOPE)
                    if not self._vector_read_select:
                        tmp_cache_buffer = self._schedule.cache_read(
                            lop["dst_buffer"],
                            cce.scope_ubuf,
                            list(set(tmp_read_map)))
                        lop["cache_buffer"] = tmp_cache_buffer
                        v100_cache_buffer.append(tmp_cache_buffer)
                    continue

                if dequant_flag:
                    tmp_cache_buffer = self._schedule.cache_read(
                        lop["dst_buffer"],
                        cce.scope_ubuf,
                        list(set(tmp_read_map)))
                    lop["cache_buffer"] = tmp_cache_buffer
                    scale_ub = tmp_cache_buffer
                    continue
                handle_v100_quant_input_continue_flag = "convolution" in lop["op"] \
                or "convolution_bias_ub_brc" in tmp_read_map[0].op.name or \
                "fusion_fmap_select" in tmp_read_map[0].op.name or \
                "fmap_l1" in tmp_read_map[0].op.name or \
                "aipp_res_convolution" in tmp_read_map[0].op.tag or \
                "weight_unzip" in tmp_read_map[0].op.name or\
                "strided_read" in tmp_read_map[0].op.tag or \
                lop["dst_buffer"].name == 'compress_index' or \
                lop["dst_buffer"].name == "Filter"
                if handle_v100_quant_input_continue_flag:
                    continue
                tmp_cache_buffer = self._schedule.cache_read(
                    lop["dst_buffer"],
                    cce.scope_ubuf,
                    list(set(tmp_read_map)))
                lop["cache_buffer"] = tmp_cache_buffer
                v100_cache_buffer.append(tmp_cache_buffer)

            return v100_cache_buffer, scale_ub

        def get_multiout_ub2(multi_out):
            multiout_ub2 = {}
            if multi_out and self._convbn1_flag:
                for out_op in multi_out:
                    for lop in self._op_graph.body_ops:
                        if lop["dst_buffer"].name.split('.')[0] == \
                                out_op.op.name:
                            tmp_read_map = []
                            for nop in lop["next_op"]:
                                if nop["dst_buffer"] not in tmp_read_map:
                                    tmp_read_map.append(nop["dst_buffer"])
                            multiout_ub2[out_op.name] = sch.cache_read(
                                lop["dst_buffer"],
                                cce.scope_ubuf,
                                tmp_read_map)
            return multiout_ub2

        def cal_double_opprand_num():
            if self._lhisi_dequant_quant_para["deq_sqrt"] or \
                    self._lhisi_dequant_quant_para["deq_relu"]:
                self._fused_double_operand_num += 1
            if self._lhisi_data_flow_type in (
                    DataFlowTypeLhisi.S32toFp16S8,
                    DataFlowTypeLhisi.S32toS8):
                self._fused_double_operand_num += 2
            self._fused_double_operand_num += double_num

        def get_c_ub_reform():
            if self._v200_data_flow_type == DataFlowType.S32toS8:
                c_ub_reform = ConvParam.TENSOR_MAP["data_transfer"]
            elif self._v200_data_flow_type == DataFlowType.S16EltWiseS8:
                c_ub_reform = sch.cache_write(res, cce.scope_ubuf)
            elif self._v200_data_flow_type == DataFlowType.S16EltWiseS8S16:
                c_ub_reform = tensor_map["c_ub_reform"]
            else:
                c_ub_reform = None
            return c_ub_reform

        def c04_row_major_reshape_compute(tensor_map):
            if self._dynamic_mode is None and \
                    is_support_v200():
                row_major_reshape = tensor_map["row_major_reshape_res"]
                sch[row_major_reshape].compute_inline()

        def c04_row_major_buffer_tile():
            if c0_optim_flg:
                kernel_h = c_ub.op.attrs['kernel_h']
                sch[fmap_col_before].buffer_tile( \
                (None, None), \
                (None, None), \
                (None, None), \
                (None, kernel_h), \
                (None, None), \
                (None, None))

        def bl0_attach():
            """
            do BL0 tensor compute_at
            """
            if self.conv_pool_fused_flag:
                sch[bl0].compute_at(sch[c_col], coio)
                if blocks != 1:
                    sch[bl0].allocate_at(sch[res], bidi, \
                        run_once_axes=[m_outer_outer_inner, \
                        m_outer_outer_outer_inner, boo])
                else:
                    sch[bl0].allocate_at(sch[res], m_outer_outer_outer_outer, \
                        run_once_axes=[m_outer_outer_inner, \
                        m_outer_outer_outer_inner, boo])
                sch[bl0].pragma(bl0.op.axis[0], "filter_reshape", 1)
            elif tiling["BL0_matrix"]:
                sch[bl0].compute_at(sch[c_col], coo)
            else:
                if blocks == 1:
                    sch[bl0].compute_at(sch[res_c], batch_outer)
                else:
                    sch[bl0].compute_at(sch[res_c], bido)

        def _align_fmap_col_before(l0a_load2d_flag, w_out):
            if l0a_load2d_flag:
                sch[al1].storage_align(sch[al1].op.axis[1], 256, 0)
            elif not l0a_load2d_flag and self._dynamic_mode is None:
                channel_0 = \
                    ConvParam.tiling_query_param["fmap_shape_nc1hwc0"][-1]
                if is_support_v200() and channel_0 <= 4:
                    sch[fmap_col_before].buffer_align(
                        (1, 1),
                        (__lcm(w_out, 16), __lcm(w_out, 16)),
                        (1, 1), (1, 1), (1, 1),
                        (1, CUBE_MKN[fmap_col_before.dtype]["mac"][1]))
                elif self._conv1d_split_w_flag:
                    sch[fmap_col_before].buffer_align(
                        (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),
                        (1, CUBE_MKN[fmap_col_before.dtype]["mac"][1]))
                else:
                    sch[fmap_col_before].buffer_align(
                        (1, 1), (w_out, w_out), (1, 1), (1, 1), (1, 1),
                        (1, CUBE_MKN[fmap_col_before.dtype]["mac"][1]))

        def _non_convolution_body_set_scope():
            for lop in self._op_graph.body_ops:
                if (("convolution" not in lop["op"]) or \
                    (self._fused_flag and (lop["op"] == "convolution_C")))  \
                    and (lop['dst_buffer'] not in spec_node_list):
                    self._schedule[lop["dst_buffer"]].set_scope(cce.scope_ubuf)

        def _quant_bias_set_scope():
            bias = tensor_map["bias"]
            bias_l0c = tensor_map["bias_l0c"]
            c_col_bias = tensor_map["c_col_bias"]
            bias_ub_brc = None
            sch[bias_l0c].set_scope(cce.scope_cc)
            sch[c_col_bias].set_scope(cce.scope_cc)
            if bias_optimize_flag:
                bias_ub_brc = tensor_map["bias_ub_brc"]
                sch[bias_ub_brc].set_scope(cce.scope_ubuf)
                bias_ub = sch.cache_read(bias, cce.scope_ubuf, [bias_ub_brc])
            else:
                bias_ub = sch.cache_read(bias, cce.scope_ubuf, [bias_l0c])
            has_bias_ub = True

            return bias, bias_l0c, c_col_bias, \
                   bias_ub_brc, bias_ub, has_bias_ub
            # for multi core pass must assert all buffer in one for Loop

        def set_output_memory_type():
            if self._output_memory_type == "fuse_flag":
                if self._lhisi_data_flow_type == DataFlowTypeLhisi.S32toFp16S8:
                    res_out_fp16 = tensor_map["res_out_fp16"]
                    res_out_s8 = tensor_map["res_out_s8"]
                    if "addr_type" in res_out_fp16.op.attrs:
                        fp16_addr_type = \
                        res_out_fp16.op.attrs["addr_type"].value
                    else:
                        fp16_addr_type = DDR_SCOPE
                    if "addr_type" in res_out_s8.op.attrs:
                        s8_addr_type = res_out_s8.op.attrs["addr_type"].value
                    else:
                        s8_addr_type = DDR_SCOPE
                    self._output_memory_type = [fp16_addr_type, s8_addr_type]

                    if fp16_addr_type == L1_FUSION_SCOPE:
                        sch[res_out_fp16].set_scope(cce.scope_cbuf_fusion)
                    if s8_addr_type == L1_FUSION_SCOPE:
                        sch[res_out_s8].set_scope(cce.scope_cbuf_fusion)
                else:
                    if "addr_type" in res_c.op.attrs:
                        res_addr_type = res_c.op.attrs["addr_type"].value
                    else:
                        res_addr_type = DDR_SCOPE
                    self._output_memory_type = [res_addr_type]
                    if res_addr_type == L1_FUSION_SCOPE:
                        sch[res_c].set_scope(cce.scope_cbuf_fusion)
            else:
                if self._output_memory_type == L1_FUSION_SCOPE:
                    sch[res_c].set_scope(cce.scope_cbuf_fusion)
                self._output_memory_type = [self._output_memory_type]

        def _mini_or_hisi_checkout_quant_dequant(tensor_map):
            if "write_select" in res.op.name:
                self._write_select = True
                double_num, tensor_map = checkout_quant_dequant(
                    res.op.input_tensors[0], tensor_map)
            else:
                double_num, tensor_map = checkout_quant_dequant(
                    res, tensor_map)
            return double_num, tensor_map

        def _input_cache_read():

            def _cache_read_continue(lop):

                def handle_dma_copy():
                    output_ub_5d = lop["next_op"][0]["dst_buffer"]
                    #for index of operand in dma copy is nonlinear
                    sch[output_ub_5d].storage_align(\
                        sch[output_ub_5d].op.axis[2], \
                        output_ub_5d.shape[3].value * \
                        output_ub_5d.shape[4].value, 0)
                    temp = lop["dst_buffer"]
                    if "addr_type" in temp.op.attrs and \
                    int(temp.op.attrs["addr_type"]) == L1_FUSION_SCOPE:
                        sch[temp].set_scope(cce.scope_cbuf_fusion)
                    if "addr_type" in lop["dst_buffer"].op.attrs:
                        self._input_memory_type.append(
                            lop["dst_buffer"].op.attrs["addr_type"].value)

                if self._lhisi_data_flow_type:
                    return True
                if "convolution" in lop["op"]:
                    return True
                if "max_pooling_pad_" in lop["op"]:
                    return True
                if lop["dst_buffer"] == tensor_map.get("bias"):
                    return True
                if ("bias_tensor" in lop["dst_buffer"].name) \
                     and ("bias" in tensor_map.keys()):
                    return True
                if "dma_copy" in lop["next_op"][0]["dst_buffer"].op.attrs:
                    handle_dma_copy()
                    return True
                if "fusion_fmap_select" in lop["next_op"][0]["op"]:
                    return True
                if lop["dst_buffer"].name == 'compress_index' or lop[
                        "dst_buffer"].name == "Filter":
                    return True

                return False

            def handle_l1_fusion():
                if "addr_type" in lop["dst_buffer"].op.attrs and \
                int(lop["dst_buffer"].op.attrs["addr_type"]) \
                == L1_FUSION_SCOPE:
                    sch[lop["dst_buffer"]].set_scope(cce.scope_cbuf_fusion)

            for lop in self._op_graph.input_ops:
                if _cache_read_continue(lop):
                    continue

                if "addr_type" in lop["dst_buffer"].op.attrs and \
                        "conv_vector_bias_add" not in lop["next_op"][0]["op"]:
                    self._input_memory_type.append(
                        lop["dst_buffer"].op.attrs["addr_type"].value)
                handle_l1_fusion()
                fm2_flag = False
                tmp_read_map = []
                for nop in lop["next_op"]:
                    tmp_read_map.append(nop["dst_buffer"])
                    if "requant_s16_vaddrelu" in nop["op"] or \
                            "requant_s16_vadd" in nop["op"]:
                        fm2_flag = True

                # Fmap.local.UB should not exist when strided read
                # weight_unzip input handle
                if strided_read_flag and lop[
                        "dst_buffer"] == tensor_map["fmap"] or \
                        (tmp_read_map != [] and "weight_unzip" \
                            in tmp_read_map[0].op.name):
                    continue

                tmp_cache_buffer = self._schedule.cache_read(
                    lop["dst_buffer"], cce.scope_ubuf, list(set(tmp_read_map)))
                lop["cache_buffer"] = tmp_cache_buffer

                if fm2_flag:
                    v200_fm2_cache_buffer.append(tmp_cache_buffer)
                else:
                    v200_cache_buffer.append(tmp_cache_buffer)
                    if tmp_cache_buffer.dtype == "int16":
                        tensor_map["bias_s16_flag"] = True

        def _body_ops_compute_at():

            def _fuse_body_ops_compute_at():
                if has_vector_flag and self._fused_flag \
                        and not ConvParam.swrite_flag:
                    self._schedule[res_ub].compute_at(
                        self._schedule[self._compute_at_buffer[1]], \
                        self._compute_at_axis[1])
                else:
                    pass

            def _buffertile_pooling(lop):
                offset_bound_first = int_ceil_div(al1_nparts, block_dim[2]) \
                                     * al1_facter_pooling \
                                     * POOLING_STRIDE * block_tile
                offset_bound = (m_outer_outer_outer_inner
                                * al1_facter_pooling
                                + m_outer_outer_inner) \
                               * POOLING_STRIDE + 1
                offset_bound += (offset_bound_first - pooling_padding[0])
                offset_bound_first -= (pooling_padding[0] * block_tile)

                extend_first = POOLING_WINDOW * conv_w
                extend = POOLING_STRIDE * conv_w
                if conv_w % 16 != 0:
                    extend_first = int_ceil_div(extend_first, 16) * 16
                    extend = int_ceil_div(extend, 16) * 16
                self._schedule[lop["dst_buffer"]].buffer_tile(
                    (None, None),
                    (None, None),
                    (tvm.select(
                        m_outer_outer_outer_inner.var == 0,
                        tvm.select(m_outer_outer_inner.var == 0,
                                   offset_bound_first * conv_w,
                                   offset_bound * conv_w),
                        offset_bound * conv_w),
                     tvm.select(
                         m_outer_outer_outer_inner.var == 0,
                         tvm.select(m_outer_outer_inner.var == 0,
                                    extend_first,
                                    extend),
                         extend)),
                    (None, None),
                )

            def _pooling_reused(pooling_bias_add, pooling_relu):
                if self.conv_pool_fused_flag and conv_w % 16 != 0:
                    if pooling_bias_add is not None \
                            and pooling_relu is not None:
                        sch[pooling_bias_add].compute_inline()
                        sch[pooling_relu].reused_by(ConvParam.TENSOR_MAP["C"])


            pooling_bias_add = None
            pooling_relu = None
            for lop in self._op_graph.body_ops:
                if (("convolution" not in lop["op"]) or \
                    ("convolution_A" in lop["op"])) or \
                        (self._fused_flag and (lop["op"] == "convolution_C")):
                    continue_flg = self._lhisi_data_flow_type or \
                    (lop["op"] == "conv_vector_remove_pad") or \
                    (self._v200_data_flow_type is not None) or \
                    (lop["op"] == "mean_out" and self._convbn1_flag) or \
                    ("pooling2d_max" in lop["op"])

                    read_write_select_flg = \
                    ("fusion_fmap_select" in lop["op"]) or \
                    ("write_select" in lop["op"] and not ConvParam.swrite_flag)

                    if continue_flg or read_write_select_flg:
                        continue

                    compute_at_flag = lop["dst_buffer"].op.name != "fmap_l1" \
                        and lop["dst_buffer"].op.tag not in \
                        ("strided_read", "strided_write") \
                        and lop["dst_buffer"].op.tag != "aipp_res"

                    if compute_at_flag:
                        self._schedule[lop["dst_buffer"]].compute_at(
                            self._schedule[self._compute_at_buffer[1]],
                            self._compute_at_axis[1])
                    if self.conv_pool_fused_flag \
                            and len(lop["dst_buffer"].shape) == 4:
                        al1_nparts \
                            = int_ceil_div(pooling_out[0], al1_facter_pooling)
                        _buffertile_pooling(lop)
                        if lop["dst_buffer"].op.tag == "elewise_single_relu":
                            pooling_relu = lop["dst_buffer"]
                        if lop["dst_buffer"].op.tag == "conv_vector_bias_add":
                            pooling_bias_add = lop["dst_buffer"]
                    if not self._fused_flag:
                        if compute_at_flag:
                            self._schedule[lop["dst_buffer"]].buffer_align(
                                (1, 1),
                                (1, 1),
                                (1, tiling[
                                    "CL0_matrix"][1]*tiling["CL0_matrix"][2]),
                                (1, 1))

                _fuse_body_ops_compute_at()
            _pooling_reused(pooling_bias_add, pooling_relu)

        def _body_ops_convbn1_flag(multiout_ub2):
            if multi_out and self._convbn1_flag:
                for out_op in multi_out:
                    for lop in self._op_graph.body_ops:
                        if lop["dst_buffer"].name.split('.')[0] \
                            == out_op.op.name:
                            out_tensor = multiout_ub2[out_op.op.name]
                            self._schedule[out_tensor].compute_at(
                                self._schedule[self._compute_at_buffer[1]],
                                self._compute_at_axis[1])
                            sch[out_tensor].emit_insn(
                                out_tensor.op.axis[0], "phony_insn")

        def _input_ops_compute_at():
            def _checkout_input_ops_compute_at(lop):
                if self._lhisi_data_flow_type or self._v200_data_flow_type:
                    return True
                if "convolution" in lop["op"] or ( \
                    ("bias_tensor" in lop["op"]) and ( \
                    "bias" in tensor_map.keys())):
                    return True
                if ("bias_tensor" in lop["op"]) and \
                        ("fp16_bias" in tensor_map.keys()):
                    sch[lop['cache_buffer']].compute_at(sch[res_c], bido)
                    return True
                if "max_pooling_pad_" in lop["op"]:
                    return True
                if self.conv_pool_fused_flag:
                    sch[lop['cache_buffer']].compute_at(sch[res_c], bido)
                    return True
                if ("dma_copy" in lop["next_op"][0][ \
                    "dst_buffer"].op.attrs) or ( \
                    "fusion_fmap_select" in lop["next_op"][0]["op"]):
                    # in body_op
                    return True
                if lop["dst_buffer"].name == 'compress_index' or lop[ \
                    "dst_buffer"].name == "Filter":
                    return True
                return False

            for lop in self._op_graph.input_ops:
                if _checkout_input_ops_compute_at(lop):
                    continue
                if "cache_buffer" in lop:
                    self._schedule[lop["cache_buffer"]].compute_at(
                        self._schedule[self._compute_at_buffer[0]], \
                                       self._compute_at_axis[0])

        def _conv_pooling_optm():

            def _cub_buffer_tile(al1_facter_pooling):
                offset_bound_first \
                    = (block_tile * int_ceil_div(al1_nparts, block_dim[2])
                       * al1_facter_pooling * POOLING_STRIDE) * conv_w
                offset_bound \
                    = ((m_outer_outer_outer_inner * al1_facter_pooling
                        + m_outer_outer_inner) * POOLING_STRIDE + 1) * conv_w
                offset_bound \
                    += (offset_bound_first - pooling_padding[0] * conv_w)
                offset_bound_first \
                    -= (block_tile * pooling_padding[0] * conv_w)
                first_time_row = POOLING_WINDOW * conv_w
                other_time_row = POOLING_STRIDE * conv_w

                if conv_w % 16 != 0:
                    first_time_row \
                        = int_ceil_div(first_time_row, 16) * 16 + 16
                    other_time_row \
                        = int_ceil_div(other_time_row, 16) * 16 + 16
                    offset_bound_first = offset_bound_first // 16 * 16
                    offset_bound = offset_bound // 16 * 16

                self._schedule[c_ub].buffer_tile(
                    (None, None),
                    (None, None),
                    (tvm.select(m_outer_outer_outer_inner.var == 0,
                                tvm.select(m_outer_outer_inner.var == 0,
                                           offset_bound_first,
                                           offset_bound),
                                offset_bound),
                     tvm.select(m_outer_outer_outer_inner.var == 0,
                                tvm.select(m_outer_outer_inner.var == 0,
                                           first_time_row,
                                           other_time_row),
                                other_time_row)),
                    (None, None),
                )
                sch[c_col].buffer_tile(
                    (None, None),
                    (None, None),
                    (tvm.select(m_outer_outer_outer_inner.var == 0,
                                tvm.select(m_outer_outer_inner.var == 0,
                                           offset_bound_first,
                                           offset_bound),
                                offset_bound),
                     tvm.select(
                         m_outer_outer_outer_inner.var == 0,
                         tvm.select(m_outer_outer_inner.var == 0,
                                    first_time_row,
                                    other_time_row),
                         other_time_row)),
                    (None, None),
                    (None, None),
                    (None, None),
                )

            def _al0_buffer_tile(al1_facter_pooling):
                fmap_col_offset_bound_first\
                    = int_ceil_div(al1_nparts, block_dim[2])\
                      * al1_facter_pooling * POOLING_STRIDE * conv_w
                fmap_col_offset_bound_first \
                    = block_tile * fmap_col_offset_bound_first

                fmap_col_offset_bound \
                    = (m_outer_outer_outer_inner * al1_facter_pooling
                       + m_outer_outer_inner) * POOLING_STRIDE * conv_w
                fmap_col_offset_bound \
                    += (conv_w + fmap_col_offset_bound_first
                        - pooling_padding[0] * conv_w)
                fmap_col_offset_bound_first \
                    -= block_tile * pooling_padding[0] * conv_w
                fmap_col_offset_bound_first //= 16
                fmap_col_offset_bound //= 16

                sch[fmap_col].buffer_tile(
                    (None, None),
                    (tvm.select(m_outer_outer_outer_inner.var == 0,
                                tvm.select(m_outer_outer_inner.var == 0,
                                           fmap_col_offset_bound_first
                                           + boo*cube_m,
                                           fmap_col_offset_bound
                                           + boo*cube_m),
                                fmap_col_offset_bound + boo * cube_m),
                     cube_m),
                    (None, None),
                    (None, None),
                    (None, None),
                )

            def _row_major_buffer_tile(al1_facter_pooling):
                fcol_before_offset_bound_first \
                    = block_tile * int_ceil_div(al1_nparts, block_dim[2]) \
                      * al1_facter_pooling * POOLING_STRIDE * conv_w
                fcol_before_offset_bound \
                    = m_outer_outer_outer_inner * al1_facter_pooling \
                      * POOLING_STRIDE * conv_w + conv_w
                fcol_before_offset_bound \
                    += (fcol_before_offset_bound_first
                        - pooling_padding[0] * conv_w)
                fcol_before_offset_bound_first \
                    -= block_tile * pooling_padding[0] * conv_w
                extend_fmap_col_before_first = \
                    int(conv_w * ((al1_facter_pooling - 1)
                                  * POOLING_STRIDE + POOLING_WINDOW))
                extend_fmap_col_before = \
                    int(conv_w * al1_facter_pooling * POOLING_STRIDE)

                if conv_w % 16 != 0:
                    extend_fmap_col_before += (align_16_nums * conv_w)
                    extend_fmap_col_before_first += (align_16_nums * conv_w)
                    fcol_before_offset_bound -= align_16_nums * conv_w
                    fcol_before_offset_bound_first -= align_16_nums * conv_w

                sch[fmap_col_before].buffer_tile(
                    (None, None),
                    (tvm.select(m_outer_outer_outer_inner.var == 0,
                                fcol_before_offset_bound_first,
                                fcol_before_offset_bound),
                     tvm.select(m_outer_outer_outer_inner.var == 0,
                                extend_fmap_col_before_first,
                                extend_fmap_col_before)),
                    (None, None),
                    (None, None),
                    (None, None),
                    (None, None),
                )

            def _al1_buffer_tile(al1_facter_pooling):
                al1_before_offset_bound_first \
                    = block_tile * int_ceil_div(al1_nparts, block_dim[2]) \
                      * al1_facter_pooling * POOLING_STRIDE * stride_h \
                      - ConvParam.padding[0]
                al1_before_offset_bound \
                    = al1_before_offset_bound_first + \
                      m_outer_outer_outer_inner * al1_facter_pooling \
                      * POOLING_STRIDE * stride_h + stride_h \
                      - stride_h * pooling_padding[0]
                extend_al1_first = int(kernel_h + al1_facter_pooling
                                       * POOLING_STRIDE * stride_h)
                extend_al1 = int(kernel_h + al1_facter_pooling
                                 * POOLING_STRIDE * stride_h) - stride_h
                al1_before_offset_bound_first \
                    -= stride_h * pooling_padding[0] * block_tile
                if conv_w % 16 != 0:
                    al1_before_offset_bound -= (align_16_nums * stride_h)
                    al1_before_offset_bound_first \
                        -= (align_16_nums * stride_h)
                    extend_al1 += (align_16_nums * stride_h)
                    extend_al1_first += (align_16_nums * stride_h)

                sch[al1].buffer_tile(
                    (None, None),
                    (None, None),
                    (tvm.select(m_outer_outer_outer_inner.var == 0,
                                al1_before_offset_bound_first,
                                al1_before_offset_bound),
                     tvm.select(m_outer_outer_outer_inner.var == 0,
                                extend_al1_first,
                                extend_al1)),
                    (None, None),
                    (None, None),
                )

            if self.conv_pool_fused_flag:
                al1_nparts = int_ceil_div(pooling_out[0], al1_facter_pooling)
                align_16_nums = int_ceil_div(16, conv_w)
                _cub_buffer_tile(al1_facter_pooling)
                _al0_buffer_tile(al1_facter_pooling)
                _row_major_buffer_tile(al1_facter_pooling)
                _al1_buffer_tile(al1_facter_pooling)

                sch[res].partition(m_outer_outer_outer_inner, ((0, 0),))
                sch[res].partition(m_outer_outer_inner, ((0, 0),))

        def aipp_conv_fuse_yuv_align():
            """
            aipp(yuv format) fusion needs height_index of fmap in L1 buffer
            is 2x aligned
            yuv_align pragma means:
            1: there is offset +1 of height_index  0: no offset
            pooling fusion uses special L1 tiling
            """
            yuv_align = None
            yuv_pad_align = None # only for 1951 C0=4
            if self._aipp_fuse_flag and al1.op.attrs["input_format"] \
                == "YUV420SP_U8":
                if not self.conv_pool_fused_flag:
                    howo = res.shape[2].value
                    blockdim = tiling.get('block_dim')[2]
                    height_out = dim_map.get("out_img_height_width")[0]
                    width_out = dim_map.get("out_img_height_width")[1]

                    al1_multiple = int_ceil_div(int_ceil_div(ceil(howo, 16), \
                                   tiling.get('AL0_matrix')[0]*16), \
                                   al1_factor[1])

                    data_in_l1 = tiling.get('AL0_matrix')[0]*16*al1_multiple
                    if blockdim > 1:
                        mooo = bido % blockdim * int_ceil_div(al1_factor[1], \
                               blockdim) + m_outer_outer_outer_inner
                    else:
                        mooo = m_outer_outer_outer_inner

                    howo_offset = data_in_l1*mooo

                    c0_4 = \
                        ConvParam.tiling_query_param["fmap_shape_nc1hwc0"][-1]
                    if is_support_v200() and c0_4 <= 4:
                        howo_offset = howo_offset//__lcm(width_out, 16) \
                                      *__lcm(width_out, 16)

                    ho_offset = howo_offset//width_out
                    hi_offset_with_pad = stride_h*ho_offset - \
                    ConvParam.padding[0]

                    hi_offset = tvm.select(hi_offset_with_pad >= 0, \
                                           hi_offset_with_pad, \
                                           tvm.const(0, "int32"))

                    yuv_align = hi_offset % 2

                    if is_support_v200() and c0_4 <= 4:
                        pad_offset = \
                        tvm.select(tvm.any(hi_offset_with_pad >= 0, \
                            hi_offset_with_pad <= -ConvParam.padding[0]), \
                        tvm.const(0, "int32"), hi_offset_with_pad)
                        yuv_pad_align = pad_offset % 2

                else:
                    windows_stride_h = POOLING_STRIDE

                    yuv_first_with_pad = block_tile*(dim_map["img_shape"][2] \
                                         // blocks) - ConvParam.padding[0]

                    if not isinstance(yuv_first_with_pad, tvm.expr.Sub):
                        yuv_first_with_pad = int(yuv_first_with_pad)

                    yuv_bound_start = tvm.select(yuv_first_with_pad > 0, \
                                                 yuv_first_with_pad, \
                                                 tvm.const(0, "int32"))

                    yuv_bound = yuv_first_with_pad \
                        + m_outer_outer_outer_inner * al1_facter_pooling\
                        * windows_stride_h * stride_h + stride_h

                    yuv_align = tvm.select(
                        m_outer_outer_outer_inner.var == 0,
                        yuv_bound_start,
                        yuv_bound) % 2
            return yuv_align, yuv_pad_align

        def conv1d_w_split_tile():
            """
            for conv1d case, use buffer_tile to split width
            """
            if self._conv1d_split_w_flag:
                howo = res.shape[2].value
                blockdim = tiling.get('block_dim')[2]

                al1_multiple = int_ceil_div(int_ceil_div(ceil(howo, 16), \
                               tiling.get('AL0_matrix')[0]*16), \
                               al1_factor[1])

                data_in_l1 = tiling.get('AL0_matrix')[0]*16*al1_multiple
                if blockdim > 1:
                    mooo = bido % blockdim * int_ceil_div(al1_factor[1], \
                           blockdim) + m_outer_outer_outer_inner
                else:
                    mooo = m_outer_outer_outer_inner

                wo_offset = tvm.min(data_in_l1*mooo, (w_out-1))
                wi_offset_with_pad = stride_w*wo_offset - ConvParam.padding[2]
                if w_out == 1 and w_out != howo:
                    wi_extent = tvm.select(int(kw_dilate + (data_in_l1-1)*stride_w) + \
                        wi_offset_with_pad > int(fmap.shape[3].value), \
                        int(fmap.shape[3].value), int(kw_dilate + (data_in_l1-1)*stride_w))
                else:
                    wi_extent = kw_dilate + (data_in_l1-1)*stride_w
                sch[al1].buffer_tile((None, None), (None, None), (None, None),\
                     (wi_offset_with_pad, wi_extent), (None, None))

        def get_al1_bound():
            """
            for al1_bound set for dynamic batch
            """
            ho, wo = dim_map['out_img_height_width']
            stride_h, stride_w = c_ub.op.attrs['stride']
            mc, m0 = tiling['CL0_matrix'][1:3]
            stride_update = 1 if strideh_opti_flag else stride_h
            if tiling['AL1_shape']:
                al1_m = tiling['AL1_shape'][1]*mc*m0
            else:
                al1_m_raw = int_ceil_div_tvm(dim_map["out_img_shape"][-2], al1_factor[1])
                al1_m = int_ceil_div_tvm(al1_m_raw, m0)*m0

            if l0a_load2d_flag:
                return al1_m

            if (al1_m % wo == 0):
                extend_h = 0
            elif (al1_m*2 % wo == 0):
                extend_h = 1
            else:
                extend_h = 2
            ho_len = tvm.floordiv(al1_m, wo) + extend_h
            hi_max = c_ub.op.attrs['kernel_h'] + (ho_len - 1)*stride_update

            if tiling["AL1_shape"]:
                al1_bound = hi_max*wo*tiling["AL1_shape"][0]*m0
            else:
                al1_bound = hi_max*wo*dim_map['out_img_shape'][1]*m0

            return al1_bound

        def self_init():
            self._tiling_case = tiling_case
            self._var_range = var_range
            self._schedule = sch_list[0]
            self._convbn1_flag = convbn1_flag
            self._l1_fusion_type = int(ConvParam.fusion_para.get(\
                "l1_fusion_type", -1))
            self._fmap_l1_addr_flag = \
            ConvParam.fusion_para.get("fmap_l1_addr_flag", 1)
            self._fmap_l1_valid_size = \
            ConvParam.fusion_para.get("fmap_l1_valid_size", 0)
            self._input_memory_type = \
                [ConvParam.fusion_para.get("input_memory_type")]
            self._output_memory_type = \
                ConvParam.fusion_para.get("output_memory_type")
            self._valid_shape = ConvParam.fusion_para.get("valid_shape")
            self._op_graph = AutoScheduleOp(res)
            self.unzip_parameters["weight_zip_flag"] = \
            self._op_graph.weight_zip_flag
            self._v200_data_flow_type = \
            ConvParam.TENSOR_MAP.get("v200_data_flow_type")
            self._conv1d_split_w_flag = ConvParam.conv1d_split_w_flag
            self._res_tensor = res
            self._aipp_fuse_flag = ConvParam.TENSOR_MAP["aipp_fuse_flag"]
            self._fused_flag = ConvParam.TENSOR_MAP["conv_vector_fused_flag"]
            self.conv_pool_fused_flag = \
                self._max_pool_tensor_map["is_conv_pool_fused"]
            self._compute_at_buffer = []
            self._compute_at_axis = []

        def set_overload_flag(overload_flag, param, noi):
            """
            set flag on the first axis

            Parameters
            ----------
            overload_flag: True means overload

            noi: axis to set flag

            Returns
            -------
            """

            if not self._convbn1_flag:
                if overload_flag:
                    param.pragma(noi, "json_info_cache_read_mode", 0)
                else:
                    param.pragma(noi, "json_info_cache_read_mode", 1)

        self_init()
        tensor_map = ConvParam.TENSOR_MAP
        sch = self._schedule
        var_map = self._var_map
        if self._dynamic_mode == "dynamic_hw":
            fmap_h_range = self._var_range['fmap_h']
            fmap_w_range = self._var_range['fmap_w']
            ho_range = self._var_range['ho']
            wo_range = self._var_range['wo']
            sch.set_var_range(var_map['fmap_h'], fmap_h_range[0], \
                fmap_h_range[1])
            sch.set_var_range(var_map['fmap_w'], fmap_w_range[0], \
                fmap_w_range[1])
            sch.set_var_range(var_map['ho'], ho_range[0], ho_range[1])
            sch.set_var_range(var_map['wo'], wo_range[0], wo_range[1])
        elif self._dynamic_mode == "dynamic_batch":
            batch_range = self._var_range['batch_n']
            sch.set_var_range(var_map['batch_n'], batch_range[0], \
                batch_range[1])

        if cce_conf.get_soc_spec("SOC_VERSION") in \
        ("Ascend310", "Hi3796CV300ES") or is_support_v200():
            double_num, tensor_map = \
            _mini_or_hisi_checkout_quant_dequant(tensor_map)
        double_operand_num_fetch()
        if is_support_v200():
            self._lhisi_data_flow_type = None

        # mark if sread + conv + swrite or conv + swrite
        # to set double operand num as 0
        swrite_onlyconv_flag = False

        if res.op.tag == "strided_write":
            ConvParam.swrite_flag = True
            _, _, swrite_hw, swrite_c0 = list(
                i.value for i in res.shape)
            swrite_stride = res.op.attrs["stride"].value
            sch[res].bind_buffer(res.op.axis[0],
                                   swrite_stride*swrite_hw*swrite_c0,
                                   0)
            if "quant" in res.op.input_tensors[0].op.tag:
                for i, j in enumerate(self._op_graph.body_ops):
                    if j["op"] == "quant":
                        del self._op_graph.body_ops[i]
                sch[res.op.input_tensors[0]].compute_inline()
                ConvParam.swrite_dequant_flag = True
            else:
                ConvParam.swrite_dequant_flag = False
                if res.op.input_tensors[0].name == "C":
                    self._fused_flag = False
                    swrite_onlyconv_flag = True
                    sch[res.op.input_tensors[0]].compute_inline()
        else:
            ConvParam.swrite_flag = False
            ConvParam.swrite_dequant_flag = False

        if ConvParam.swrite_dequant_flag:
            del_op = None
            if self._v200_data_flow_type == DataFlowType.S32toFp16:
                del_op = "dequant_remove_pad"
            if self._v200_data_flow_type == DataFlowType.S32toS8:
                del_op = "requant_remove_pad"
            if del_op:
                for i, j in enumerate(self._op_graph.body_ops):
                    if j["op"] == del_op:
                        del self._op_graph.body_ops[i]

        if "c_ub" in tensor_map.keys():
            c_ub = tensor_map["c_ub"]

        if len(spec_node_list) > 1:
            multi_out = spec_node_list[:-1]
        else:
            multi_out = None

        dim_map = self._dim_map
        res_c = self._res_tensor
        _non_convolution_body_set_scope()

        if self._lhisi_data_flow_type:
            v100_cache_buffer, scale_ub = handle_v100_quant_input()
            handle_lhisi_fuse_res()
        v200_cache_buffer = []
        v200_fm2_cache_buffer = []

        strided_read_flag = tensor_map["strided_read_flag"]
        _input_cache_read()

        # v200: handle remove_pad
        if self._v200_data_flow_type or self._lhisi_data_flow_type:
            l0c_remove_pad = tensor_map["l0c_remove_pad"]
            sch[l0c_remove_pad].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
            sch[l0c_remove_pad].compute_inline()

        if self._v200_data_flow_type in (DataFlowType.S16EltWiseS8,
                                         DataFlowType.S16EltWiseS8S16):
            dequant_s16_remove_pad = ConvParam.TENSOR_MAP[
                "dequant_s16_remove_pad"]
            sch[dequant_s16_remove_pad].compute_inline()

        # v200: get c_ub_reform
        c_ub_reform = get_c_ub_reform()

        if self._lhisi_data_flow_type or self._conv_quant_fused_flag:
            ConvParam.tiling_query_param["res_dtype"] = res_c.dtype
            if self._lhisi_data_flow_type == DataFlowTypeLhisi.S32toFp16S8:
                ConvParam.tiling_query_param["res_dtype"] = "int8"

        l0a_load2d_flag = tensor_map["l0a_load2d_flag"]
        bias_optimize_flag = tensor_map["bias_optimize_flag"]
        strideh_opti_flag = tensor_map["strideh_opti_flag"]
        c0_optim_flg = tensor_map["c0_optim_flg"]
        c04_v200_flg = tensor_map["c04_v200_flg"]
        has_bias_ub = False

        if "bias" in tensor_map.keys():
            bias, bias_l0c, c_col_bias, bias_ub_brc, bias_ub, has_bias_ub = \
                _quant_bias_set_scope()

        fmap = tensor_map["fmap"]
        if strideh_opti_flag:
            fmap_l1 = tensor_map["fmap_l1"]
        weight = tensor_map["filter"]
        c_col = tensor_map["c_col"]
        if strideh_opti_flag and strided_read_flag:
            strided_read_flag = False
            sread_strideh_flag = True
            fmap_sread = tensor_map["fmap_l1"].op.input_tensors[0]
            sch[fmap_sread].compute_inline()
        else:
            sread_strideh_flag = False

        if l0a_load2d_flag and strided_read_flag:
            strided_read_flag = False
            sread_load2d_flag = True
            fmap_sread = tensor_map["al1_load2d"].op.input_tensors[0]
            sch[fmap_sread].compute_inline()
        else:
            sread_load2d_flag = False

        if self._valid_shape and self._input_memory_type[0] != 1:
            if not tensor_map["strideh_opti_flag"] and not l0a_load2d_flag:
                fusion_fmap_select = tensor_map['fusion_fmap_select']

        config = CUBE_MKN[weight.dtype]

        if l0a_load2d_flag:
            al1 = tensor_map["al1_load2d"]
            al0 = tensor_map["al0_load2d"]
        else:
            if self._dynamic_mode is None:
                fmap_col_before = tensor_map["fmap_im2col_row_major_res"]
            fmap_col = tensor_map["fmap_im2col_fractal_res"]
            c04_row_major_reshape_compute(tensor_map)

        bias_preload_flag = False
        if "bias" in tensor_map.keys():
            bias_preload_flag = True

        kernel_h = ConvParam.filter_h
        stride_h = ConvParam.stride_h
        stride_w = ConvParam.stride_w
        kw_dilate = (ConvParam.filter_w - 1)*ConvParam.dilate_w + 1
        kh_dilate = (kernel_h - 1)*ConvParam.dilate_h + 1

        if self._valid_shape:
            filter_w = self._valid_shape[3]
        else:
            if self._aipp_fuse_flag:
                filter_w = fmap_col_before.op.input_tensors[0].shape[3]
            elif fmap.op.input_tensors:
                filter_w = fmap.shape[3]
            else:
                filter_w = fmap.shape[3]

        w_out = ConvParam.w_out
        _align_fmap_col_before(l0a_load2d_flag, w_out)

        if self.conv_pool_fused_flag:
            pass
        else:
            sch[c_ub].buffer_align((1, 1),
                                   (1, 1),
                                   (1, CUBE_MKN[c_ub.dtype]["mac"][0]),
                                   (1, CUBE_MKN[c_ub.dtype]["mac"][2]))

        has_vector_flag = False

        if not ((cce_conf.get_soc_spec("SOC_VERSION") in \
            ("Ascend310", "Hi3796CV300ES") or is_support_v200()) \
        and ConvParam.res_dtype == 'int32') and not self.conv_pool_fused_flag:
            has_vector_flag = (c_ub.op.attrs['no_vector'].value == 0)

        if self._convbn1_flag or "write_select" in \
        self._op_graph.output_ops[0]["op"]:
            has_vector_flag = 0

        if (has_vector_flag and self._fused_flag and \
            not ConvParam.swrite_flag) or \
        (self._v200_data_flow_type in (DataFlowType.V200_General_Fusion,) and \
            res.op.name != "conv_virtual_res" and res.op.tag != "quant"):
            res_ub = sch.cache_write(res, cce.scope_ubuf)
            self._op_graph.output_ops[0]["tensorize_axis"] = self._schedule[
                res_ub].op.axis[0]
            self._op_graph.output_ops[0]["dst_buffer"] = res_ub

        tiling = ConvParam.tiling
        multiout_ub2 = get_multiout_ub2(multi_out)
        set_output_memory_type()

        if tiling is None:
            tiling = tiling_fetch()

        # change the tiling of conv+pooling
        if self.conv_pool_fused_flag:
            pooling_out = self._max_pool_tensor_map["pooling_out"]
            pooling_padding = self._max_pool_tensor_map["pooling_padding"]
            conv_w = self._max_pool_tensor_map["conv_width"]
            cube_m, al1_facter_pooling = _tiling_of_pooling()

        tiling["al1_batch"] = 1
        if (len(tiling["AL1_shape"]) > 2) and (tiling["AL1_shape"][2] > 1):
            tiling["al1_batch"] = tiling["AL1_shape"][2]
            if self._convbn1_flag:
                self._l0b_first_flag = True
        if len(tiling["CL0_matrix"]) > 4 and tiling["CL0_matrix"][4] > 1:
            if tiling["BL1_shape"] is None:
                tiling["al1_batch"] = tiling["CL0_matrix"][4]
                self._l0b_first_flag = True
        tiling["CL0_matrix"] = tiling["CL0_matrix"][:4]

        filter_matrix = list(dim_map["filter_matrix_dim"])
        filter_matrix[1] = filter_matrix[1] // tiling["block_dim"][1]
        if tiling["BL0_matrix"] == filter_matrix:
            tiling["BL0_matrix"] = []
        if tiling["BL0_matrix"] == [] and not self.unzip_parameters.get(
                "weight_zip_flag"):
            tiling["BL1_shape"] = None

        if self.unzip_parameters.get("weight_zip_flag"):
            if tiling["BL1_shape"] is None:
                bl1 = weight
                bl0 = weight
                sch[bl0].set_scope(cce.scope_cb)
            elif tiling["BL1_shape"] == []:
                bl1 = weight
                sch[bl1].set_scope(cce.scope_cbuf)
                bl0 = sch.cache_read(bl1, cce.scope_cb, [c_col])
            else:
                bl1 = weight
                sch[bl1].set_scope(cce.scope_cbuf)
                bl0 = sch.cache_read(bl1, cce.scope_cb, [c_col])
        else:
            if tiling["BL1_shape"] is not None:
                bl1 = sch.cache_read(weight, cce.scope_cbuf, [c_col])
            else:
                bl1 = weight
            bl0 = sch.cache_read(bl1, cce.scope_cb, [c_col])

        sch[c_col].set_scope(cce.scope_cc)
        sch[c_ub].set_scope(cce.scope_ubuf)

        if l0a_load2d_flag:
            fmap_col = al0
            sch[al1].set_scope(cce.scope_cbuf)
        else:
            if self._dynamic_mode:
                if strideh_opti_flag:
                    sch[fmap_l1].set_scope(cce.scope_cbuf)
                    al1 = fmap_l1
                else:
                    al1 = sch.cache_read(fmap, cce.scope_cbuf, [fmap_col])
            else:
                # fmap DDR in  and read select
                if self._input_memory_type[0] in (0, 2) and self._valid_shape and \
                    not tensor_map["strideh_opti_flag"]:
                    if self._l1_fusion_type in (0, 1):
                        sch[fusion_fmap_select].set_scope(cce.scope_cbuf_fusion)
                    else:
                        sch[fusion_fmap_select].set_scope(cce.scope_cbuf)
                    fmap_cbuf_nc1hwc0 = fusion_fmap_select
                # fmap L1 in
                elif self._input_memory_type[0] == 1:
                    sch[fmap].set_scope(cce.scope_cbuf_fusion)
                    fmap_cbuf_nc1hwc0 = sch.cache_read(fmap, \
                                                    cce.scope_cbuf_fusion, \
                                                    [fmap_col_before])
                    if self._valid_shape:
                        sch[fmap_cbuf_nc1hwc0].buffer_tile( \
                            (None, None), \
                            (None, None), \
                            (-ConvParam.padding[0], fmap.shape[2] + \
                                ConvParam.padding[0] + ConvParam.padding[1]), \
                            (-ConvParam.padding[2], fmap.shape[3] + \
                                ConvParam.padding[2] + ConvParam.padding[3]), \
                            (None, None))
                # DDR in
                else:
                    if strideh_opti_flag or strided_read_flag \
                        or self._aipp_fuse_flag:
                        fmap_cbuf_nc1hwc0 = fmap_col_before.op.input_tensors[0]
                        if self._l1_fusion_type in (0, 1):
                            sch[fmap_cbuf_nc1hwc0].set_scope(cce.scope_cbuf_fusion)
                        else:
                            sch[fmap_cbuf_nc1hwc0].set_scope(cce.scope_cbuf)
                    elif self._l1_fusion_type in (0, 1):
                        fmap_cbuf_nc1hwc0 = sch.cache_read(
                            fmap, cce.scope_cbuf_fusion, [fmap_col_before])
                    else:
                        fmap_cbuf_nc1hwc0 = sch.cache_read(
                            fmap, cce.scope_cbuf, [fmap_col_before])

                al1 = fmap_cbuf_nc1hwc0
                sch[fmap_col_before].set_scope(cce.scope_cbuf)

        al1_shape = al1.shape
        if self._l1_fusion_type == 1 or self._input_memory_type[0] == 1:
            sch[al1].buffer_align(
                (1, 1),
                (1, 1),
                (al1_shape[2], al1_shape[2]),
                (al1_shape[3], al1_shape[3]),
                (1, 1))
        elif self._aipp_fuse_flag and \
            al1.op.attrs["input_format"] == "YUV420SP_U8":
            sch[al1].buffer_align((1, 1), (1, 1), (2, 2), (1, 1), (1, 1))
        elif c04_v200_flg:
            sch[al1].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (al1_shape[3], al1_shape[3]),
            (1, 1))
        sch[fmap_col].set_scope(cce.scope_ca)

        factor_m = tiling["AL0_matrix"][0]
        factor_k = tiling["AL0_matrix"][1]

        a1_axis, a3_axis = sch[fmap_col].split(
            sch[fmap_col].op.axis[1], factor_m)
        a2_axis, a4_axis = sch[fmap_col].split(
            sch[fmap_col].op.axis[2], factor_k)

        # split N begin
        fmap_col_no, fmap_col_ni = sch[fmap_col].split(
            sch[fmap_col].op.axis[0], 1)

        sch[fmap_col].reorder(
            fmap_col_no, a1_axis, a2_axis, fmap_col_ni, a3_axis, a4_axis,
            sch[fmap_col].op.axis[3], sch[fmap_col].op.axis[4])
        new_fmap_col_axis = [fmap_col_no, a1_axis, a2_axis,
                             fmap_col_ni, a3_axis, a4_axis,
                             sch[fmap_col].op.axis[3],
                             sch[fmap_col].op.axis[4]]
        # split N end

        new_c_col_axis = [sch[c_col].op.axis[0], sch[c_col].op.axis[1],
                          sch[c_col].op.axis[2], sch[c_col].op.axis[3]]
        _, _, _, nn_axis = new_c_col_axis

        if "bias" in tensor_map.keys():
            a2_axis, a3_axis = sch[c_col_bias].split(
                sch[c_col_bias].op.axis[2], config["mac"][0])
            sch[c_col_bias].reorder(
                sch[c_col_bias].op.axis[0], sch[c_col_bias].op.axis[1],
                a2_axis, a3_axis, sch[c_col_bias].op.axis[3])

        c_tiling_factor = [tiling["CL0_matrix"][0],
                           tiling["CL0_matrix"][1]*tiling["CL0_matrix"][2]]
        c_factor = [int_ceil_div(dim_map["out_img_shape"][1],
                                 c_tiling_factor[0]),
                    int_ceil_div(dim_map["out_img_shape"][2],
                                 c_tiling_factor[1])]

        c_ub_tiling_factor = tiling["CUB_matrix"]
        c_ub_factor = [int_ceil_div(c_tiling_factor[0],
                                    c_ub_tiling_factor[0]),
                       int_ceil_div(
                           c_tiling_factor[1],
                           c_ub_tiling_factor[1]*c_ub_tiling_factor[2])]

        if self.conv_pool_fused_flag:
            c_tiling_factor[1] = pooling_out[1]
            c_factor[1] = pooling_out[0]

        if tiling["AL1_shape"]:
            if len(tiling["AL1_shape"]) == 1:
                tiling["AL1_shape"] = tiling["AL1_shape"] + [1]
            al1_factor = [int(dim_map["img_shape"][1] // tiling["AL1_shape"][0]),
                          int_ceil_div(c_factor[1], tiling["AL1_shape"][1])]
        else:
            al1_factor = [1, 1]

        self.overload_flag = check_feature_map(tiling, al1_factor)

        if tiling["BL1_shape"]:
            if len(tiling["BL1_shape"]) > 1:
                if c_factor[0] % tiling["BL1_shape"][1] != 0:
                    err_man.raise_err_specific("conv2d", \
                        "second value of BL1_shape should "\
                        + "be factor of n block num")
                if tiling["BL1_shape"][1] > 1 and \
                        tiling["BL1_shape"][1] % 2 != 0:
                    err_man.raise_err_specific("conv2d", \
                        "second value of BL1_shape better to be even number")
            if len(tiling["BL1_shape"]) == 1:
                tiling["BL1_shape"] = tiling["BL1_shape"] + [1]
            bl1_factor = [int((dim_map["img_shape"][1] +
                               tiling["BL1_shape"][0] - 1) // \
                              tiling["BL1_shape"][0]),
                              (c_factor[0] + tiling["BL1_shape"][1] - 1) // \
                              tiling["BL1_shape"][1]]
        else:
            bl1_factor = [1, tiling["block_dim"][1]]

        # --------------------------double buffer------------------------
        double_buffer_flag = {'AL1_pbuffer': False,
                              'BL1_pbuffer': False,
                              'AL0_pbuffer': False,
                              'BL0_pbuffer': False,
                              'CL0_pbuffer': False,
                              'CUB_pbuffer': False,
                              'UBG_pbuffer': False}

        if "manual_pingpong_buffer" in tiling:
            double_buffer_flag = tiling["manual_pingpong_buffer"]

        # --------------------------tile res_c------------------------
        if self._convbn1_flag:
            k_0, k_1 = res_c.op.reduce_axis
            block_dim = [1, 1, 1]
            if "block_dim" in tiling:
                block_dim = tiling["block_dim"]

            m_outer_outer, m_outer_inner = sch[res_c].split(
                k_1, c_tiling_factor[1])
            m_outer_outer_outer, m_outer_outer_inner = sch[
                res_c].split(m_outer_outer, nparts=al1_factor[1])

            batch_outer, batch_inner = sch[res_c].split(
                k_0, nparts=block_dim[0])
            m_outer_outer_outer_outer, m_outer_outer_outer_inner = sch[
                res_c].split(m_outer_outer_outer, nparts=block_dim[2])
            sch[res_c].reorder(batch_outer, m_outer_outer_outer_outer,
                               batch_inner, m_outer_outer_outer_inner)
            batch_h_fused = sch[res_c].fuse(
                batch_outer, m_outer_outer_outer_outer)

            sum_x_ub_rf, _ = sch.rfactor(res_c, batch_h_fused)
            sch[sum_x_ub_rf].set_scope(cce.scope_ubuf)
            sum_x_global, square_sum_x_global = sch.cache_write(
                [res_c, res_c], "global")
            sch[sum_x_global].reorder(
                sum_x_global.op.reduce_axis[0],
                sum_x_global.op.axis[0],
                sum_x_global.op.axis[1])

            sch_list.append(multi_out[0])
            sch_list.append(sum_x_global)
            sch_list.append(square_sum_x_global)

            c_rf_outer_outer, c_rf_outer_inner = sch[sum_x_global].split(
                sum_x_global.op.axis[0],
                (sum_x_global.shape[0].value // c_factor[0]))
            tmp_value = (sum_x_global.op.axis[0].dom.extent) // \
                (sum_x_global.shape[0].value // c_factor[0])
            factor_ac0 = tmp_value // bl1_factor[1]
            if factor_ac0.value == 0:
                factor_ac0 = 1
            factor_ac = bl1_factor[1] // block_dim[1]

            c_rf_outer_outer_outer, c_rf_outer_outer_inner = sch[
                sum_x_global].split(c_rf_outer_outer, factor_ac0)
            c_rf_outer_outer_outer_outer, c_rf_outer_outer_outer_inner = sch[
                sum_x_global].split(c_rf_outer_outer_outer, factor_ac)

            c_outer_outer, c_outer_inner = sch[sum_x_ub_rf].split(
                sum_x_ub_rf.op.axis[1],
                (sum_x_ub_rf.shape[1].value // c_factor[0]))
            c_outer_outer_outer, c_outer_outer_inner = sch[sum_x_ub_rf].split(
                c_outer_outer, factor_ac0)
            c_outer_outer_outer_outer, c_outer_outer_outer_inner = sch[
                sum_x_ub_rf].split(c_outer_outer_outer, factor_ac)
            bl1_at_c_axis = c_outer_outer_outer_inner

            batch_inner_outer, batch_inner_inner = sch[sum_x_ub_rf].split(
                sum_x_ub_rf.op.reduce_axis[0], tiling["al1_batch"])

            if self._l0b_first_flag:
                sch[sum_x_ub_rf].reorder(
                    c_outer_outer_outer_outer, c_outer_outer_outer_inner,
                    batch_inner_outer, sum_x_ub_rf.op.reduce_axis[1],
                    c_outer_outer_inner, sum_x_ub_rf.op.reduce_axis[2],
                    c_outer_inner, sum_x_ub_rf.op.reduce_axis[3],
                    batch_inner_inner, sum_x_ub_rf.op.axis[2])
            else:
                sch[sum_x_ub_rf].reorder(
                    c_outer_outer_outer_outer, batch_inner_outer,
                    c_outer_outer_inner, c_outer_outer_outer_inner,
                    sum_x_ub_rf.op.reduce_axis[1],
                    sum_x_ub_rf.op.reduce_axis[2],
                    c_outer_inner, sum_x_ub_rf.op.reduce_axis[3],
                    batch_inner_inner, sum_x_ub_rf.op.axis[2])

            sch[sum_x_global].reorder(
                sum_x_global.op.reduce_axis[0],
                c_rf_outer_outer_outer_outer, c_rf_outer_outer_inner,
                c_rf_outer_outer_outer_inner, c_rf_outer_inner,
                sum_x_global.op.axis[1])

            batch_fuse_channel = sch[sum_x_global].fuse(
                sum_x_global.op.reduce_axis[0], c_rf_outer_outer_outer_outer)
            c_slice_axis = sum_x_ub_rf.op.reduce_axis[2]
            al1_at_c_axis = sum_x_ub_rf.op.reduce_axis[1]
            sch[sum_x_ub_rf].compute_at(sch[sum_x_global], batch_fuse_channel)
            mc_flag = False

            blocks = block_dim[0]*block_dim[1]*block_dim[2]
            block = tvm.thread_axis("blockIdx.x")
            sch[sum_x_global].bind(batch_fuse_channel, block)
            if blocks == 1:
                noo_true = batch_inner
            if blocks == block_dim[0]:
                sch[sum_x_global].pragma(c_rf_outer_outer_inner, \
                                    'json_info_batchBindOnly', 1)

            sch[res_c].emit_insn(sch[res_c].op.axis[0], "phony_insn")
            noi_true = batch_inner_outer
            res_c = sum_x_ub_rf
        else:
            if self._dynamic_mode == "dynamic_hw":
                c_outer_outer, c_outer_inner = sch[res_c].split(
                    res_c.op.axis[1], (res_c.shape[1].value // c_factor[0]))

                # split for mul_core
                m_mulcore_factor = int_ceil_div(dim_map["out_img_shape"][-2],
                                                tiling["block_dim"][2])
                m_mulcore_factor = int_ceil_div(m_mulcore_factor, 16) * 16
                dynamic_outer, dynamic_inner = sch[res_c].split(
                    res_c.op.axis[2], m_mulcore_factor)

                # split res_c for al1 attach
                if tiling["AL1_shape"]:
                    al1_factor_for_dynamic = tiling["AL1_shape"][1]*c_tiling_factor[1]
                    al1_bound = al1_factor_for_dynamic
                    dynamic_inner_outer, dynamic_inner_inner = sch[res_c].split(
                        dynamic_inner, al1_factor_for_dynamic)
                else:
                    al1_bound = int_ceil_div_tvm( \
                        dim_map["out_img_shape"][-2], \
                        tiling["block_dim"][2])
                    al1_bound = int_ceil_div_tvm(al1_bound, 16)*16
                    al1_naparts_for_dynamic = 1
                    dynamic_inner_outer, dynamic_inner_inner = sch[res_c].split(
                        dynamic_inner, nparts=al1_naparts_for_dynamic)

                # The al1_m of load2d and load3d are different
                if not l0a_load2d_flag:
                    # load3d can not split wo
                    additional_rows = 2
                    ho_len = tvm.floordiv(al1_bound, var_map['wo']) + additional_rows
                    if strideh_opti_flag:
                        hi_max = c_ub.op.attrs['kernel_h'] + (ho_len - 1)
                    else:
                        hi_max = c_ub.op.attrs['kernel_h'] + \
                                 (ho_len - 1)*c_ub.op.attrs['stride'][0]
                    al1_m = hi_max*var_map['fmap_w']
                else:
                    # load2d unconstrained
                    al1_m = al1_bound

                # calculate al1_bound
                fmap_shape_nc1hwc0 = ConvParam.tiling_query_param.get("fmap_shape_nc1hwc0")
                _, c1, _, _, c0 = fmap_shape_nc1hwc0
                if tiling["AL1_shape"]:
                    al1_bound = al1_m*tiling["AL1_shape"][0]*c0
                else:
                    c1 = fmap_shape_nc1hwc0[1]
                    al1_bound = al1_m*c1*c0

                # split res_c for al0 attach
                dynamic_inner_inner_outer, dynamic_inner_inner_inner = sch[
                    res_c].split(dynamic_inner_inner, factor=c_tiling_factor[1])

                m_outer_outer_outer_outer = dynamic_outer
                m_outer_outer_outer_inner = dynamic_inner_outer
                m_outer_outer_inner = dynamic_inner_inner_outer
                m_outer_inner = dynamic_inner_inner_inner
                sch[res_c].reorder(m_outer_outer_outer_outer, \
                                   c_outer_outer, \
                                   m_outer_outer_outer_inner, \
                                   m_outer_outer_inner,
                                   c_outer_inner, m_outer_inner)
                c_outer_outer_outer, c_outer_outer_inner = \
                    sch[res_c].split(c_outer_outer, nparts=bl1_factor[1])
            else:
                c_outer_outer, c_outer_inner = sch[res_c].split(
                    res_c.op.axis[1], (res_c.shape[1].value // c_factor[0]))
                m_outer_outer, m_outer_inner = sch[res_c].split(
                    res_c.op.axis[2], c_tiling_factor[1])
                sch[res_c].reorder(c_outer_outer, m_outer_outer,
                                   c_outer_inner, m_outer_inner)
                m_outer_outer_outer, m_outer_outer_inner = sch[res_c].split(
                    m_outer_outer, nparts=al1_factor[1])
                c_outer_outer_outer, c_outer_outer_inner = sch[res_c].split(
                    c_outer_outer, nparts=bl1_factor[1])

            block_dim = [1, 1, 1]
            if "block_dim" in tiling:
                block_dim = tiling["block_dim"]

            # split batch of res_c
            if self._dynamic_mode == "dynamic_batch":
                al1_bound = get_al1_bound()
                batch_dim_factor = int_ceil_div(dim_map["out_img_shape"][0],
                                                tiling["block_dim"][0])
                batch_dim_factor = tvm.max(1, batch_dim_factor)
                batch_outer, batch_inner = sch[res_c].split(
                    res_c.op.axis[0], batch_dim_factor)
            else:
                batch_outer, batch_inner = sch[res_c].split(
                    res_c.op.axis[0], nparts=block_dim[0])

            batch_inner_outer, batch_inner_inner = sch[res_c].split(
                batch_inner, tiling["al1_batch"])
            # split cout of res_c
            c_outer_outer_outer_outer, c_outer_outer_outer_inner = sch[
                res_c].split(c_outer_outer_outer, nparts=block_dim[1])
            if self._dynamic_mode is None or self._dynamic_mode == "dynamic_batch":
                m_outer_outer_outer_outer, m_outer_outer_outer_inner = sch[
                    res_c].split(m_outer_outer_outer, nparts=block_dim[2])

            bl1_at_c_axis = c_outer_outer_outer_inner
            al1_at_c_axis = m_outer_outer_outer_inner
            m_outer_inner_outer, m_outer_inner_inner = sch[res_c].split(
                m_outer_inner, nparts=1)

            if self._l0b_first_flag:
                sch[res_c].reorder(batch_outer,
                                   c_outer_outer_outer_outer,
                                   m_outer_outer_outer_outer,
                                   c_outer_outer_outer_inner,
                                   batch_inner_outer,
                                   m_outer_outer_outer_inner,
                                   c_outer_outer_inner,
                                   m_outer_outer_inner,
                                   c_outer_inner,
                                   m_outer_inner_outer,
                                   batch_inner_inner,
                                   m_outer_inner_inner)
            else:
                sch[res_c].reorder(batch_outer,
                                   c_outer_outer_outer_outer,
                                   m_outer_outer_outer_outer,
                                   batch_inner_outer,
                                   c_outer_outer_outer_inner,
                                   m_outer_outer_outer_inner,
                                   batch_inner_inner)

            c_slice_axis = m_outer_outer_inner

            mc_flag = False
            blocks = block_dim[0]*block_dim[1]*block_dim[2]

            if blocks != 1:
                batch_cout_fused = sch[res_c].fuse(
                    batch_outer, c_outer_outer_outer_outer,
                    m_outer_outer_outer_outer)
                if self._dynamic_mode in ("dynamic_hw", "dynamic_batch"):
                    noo_true, _ = sch[res_c].split(batch_cout_fused, factor=1)
                else:
                    noo_true, _ = sch[res_c].split(batch_cout_fused, nparts=blocks)
                bido, bidi = sch[res_c].split(noo_true, 1)
                block = tvm.thread_axis("blockIdx.x")
                sch[res_c].bind(bido, block)
                mc_flag = True
                if blocks == block_dim[0]:
                    sch[res_c].pragma(bidi, 'json_info_batchBindOnly', 1)
            else:
                noo_true = batch_inner_outer

            noi_true = batch_inner_outer

        noo, noi = sch[res_c].split(noi_true, factor=1)

        if not mc_flag:
            bido = noo

        if self._dynamic_mode is None:
            set_overload_flag(self.overload_flag,
                              self._schedule[res], noi)

        reorder_flag = False

        if not tiling["BL1_shape"]:
            reorder_flag = True
        elif double_buffer_flag["AL1_pbuffer"] == double_buffer_flag[
                "BL1_pbuffer"]:
            if self._dynamic_mode in ("dynamic_hw", "dynamic_batch"):
                pass
            elif bl1_factor[1] >= al1_factor[1]:
                reorder_flag = True
        elif double_buffer_flag["BL1_pbuffer"] == 2:
            reorder_flag = True

        if self.unzip_parameters.get("weight_zip_flag"):
            reorder_flag = False

        if self._convbn1_flag:
            if not self._l0b_first_flag:
                if reorder_flag:
                    sch[res_c].reorder(res_c.op.reduce_axis[1],
                                       noi,
                                       c_outer_outer_outer_inner)
                else:
                    sch[res_c].reorder(c_outer_outer_outer_inner,
                                       res_c.op.reduce_axis[1],
                                       noi)
            m_outer_inner_outer, m_outer_inner_inner = sch[res_c].split(
                res_c.op.reduce_axis[3], nparts=1)
        else:
            if reorder_flag:
                sch[res_c].reorder(m_outer_outer_outer_inner,
                                   noi,
                                   c_outer_outer_outer_inner)
                if tiling["BL1_shape"]:
                    self.unzip_parameters["compress_tiling"][1] = \
                    self.unzip_parameters["compress_tiling"][1] // \
                    tiling["BL1_shape"][1]
            else:
                sch[res_c].reorder(c_outer_outer_outer_inner,
                                   m_outer_outer_outer_inner, noi)

        batch_cout_reorder_flag = False
        if "n_bef_batch_flag" in tiling and not self._dynamic_mode:
            if tiling["n_bef_batch_flag"] and (not reorder_flag) and \
                    (not self._l0b_first_flag):
                sch[res_c].reorder(c_outer_outer_outer_inner, noo)
                batch_cout_reorder_flag = True

        # ========= handle conv + Pooling fusion========
        if self.conv_pool_fused_flag:
            if block_dim[2] != 1:
                block_tile = bido % block_dim[2]
            else:
                block_tile = bido - bido
            handle_max_pooling()
            build_config = get_fusion_build_cfg()
            build_config["read_write_bank_conflict"] = 1
            build_config["sync_mode"] = 3

        # ============ tile cub ========================
        c_outer_inner_outer, c_outer_inner_inner = sch[
            res_c].split(c_outer_inner, nparts=c_ub_factor[0])

        if not self._l0b_first_flag:
            sch[res_c].reorder(c_outer_inner_outer, m_outer_inner_outer,
                               c_outer_inner_inner, m_outer_inner_inner)
        # v200 compute_at
        if self._v200_data_flow_type in (DataFlowType.S16EltWiseS8,
                                         DataFlowType.S16EltWiseS8S16):
            sch[c_ub_reform].compute_at(
                self._schedule[res_c], m_outer_inner_outer)
            for lop in self._op_graph.body_ops:
                if lop["op"] in ("requant_s16_vaddrelu", "requant_s16_vadd",
                                 "requant_s16_vector", "requant_s16_scale",
                                 "dequant_s16_vector", "dequant_s16_scale",
                                 "res_remove_pad_s16", "res_remove_pad_u8"):
                    self._schedule[lop["dst_buffer"]].compute_at(
                        self._schedule[res_c], m_outer_inner_outer)
            for buffer_fm2 in v200_fm2_cache_buffer:
                sch[buffer_fm2].compute_at(self._schedule[res_c], c_slice_axis)

        if self._v200_data_flow_type in (DataFlowType.V200_General_Fusion,):
            handle_lhisi_fuse_compute_at()

        if self._v200_data_flow_type:
            for buffer_data in v200_cache_buffer:
                # deq_scale/req_scale/bias_s16 
                if buffer_data.dtype in ("uint64", "int16"):
                    sch[buffer_data].compute_at(self._schedule[res_c], bido)
                # input_y in conv + dequant + add + quant
                elif buffer_data.dtype == "float16":
                    sch[buffer_data].compute_at(
                        self._schedule[res_c], c_slice_axis)
                else:
                    pass

        # v100 compute_at
        elif self._lhisi_data_flow_type:
            handle_lhisi_fuse_compute_at()
            for buffer_eltwise in v100_cache_buffer:
                sch[buffer_eltwise].compute_at(
                    self._schedule[res_c], m_outer_inner_outer)

        if self._l0b_first_flag:
            sch[c_ub].reorder(c_ub.op.axis[1], c_ub.op.axis[0],
                              c_ub.op.axis[2], c_ub.op.axis[3])
        sch[c_ub].compute_at(sch[res_c], m_outer_inner_outer)  # k.inner.outer

        if self._convbn1_flag:
            d_pad = tensor_map["C"]
            sch[d_pad].set_scope(cce.scope_ubuf)
            sch[d_pad].compute_at(sch[res_c], m_outer_inner_outer)
        c_pragma_axis = c_outer_inner_inner

        if self._l0b_first_flag and not self._convbn1_flag:
            c_pragma_axis = batch_inner_inner
        # ============ tile c_col =======================
        self._compute_at_buffer.append(res_c)
        self._compute_at_axis.append(c_slice_axis)
        self._compute_at_buffer.append(res_c)
        self._compute_at_axis.append(m_outer_inner_outer)

        if "bias" in tensor_map.keys():
            sch[c_col_bias].compute_at(sch[res_c], c_slice_axis)
            sch[bias_l0c].compute_at(sch[res_c], c_slice_axis)

        _, reduce_kk = sch[c_col].op.reduce_axis

        if self.conv_pool_fused_flag:
            m_axis_num = self._m_part_nums * config["mac"][0]
        else:
            m_axis_num = tiling['AL0_matrix'][0] * config["mac"][0]
        boo, boi = sch[c_col].split(new_c_col_axis[2],
                                    m_axis_num)

        if tiling['BL0_matrix'] == []:
            coo, coi = sch[c_col].split(new_c_col_axis[1], nparts=1)
        else:
            coo, coi = sch[c_col].split(new_c_col_axis[1], \
                tiling['BL0_matrix'][1])

        # for reduce axis, al0 and bl0 should be the same
        k_outer_outer, k_outer_inner = sch[c_col].split(
            c_col.op.reduce_axis[0], tiling['AL0_matrix'][1])
        k_outer_outer_size = c_col.op.reduce_axis[0].dom.extent // \
        tiling['AL0_matrix'][1]
        if int(al1_factor[0]) > int(bl1_factor[0]):
            k_outer_outer_inner_size = int(k_outer_outer_size // al1_factor[0])
        else:
            k_outer_outer_inner_size = int(k_outer_outer_size // bl1_factor[0])

        # split N begin
        c_col_batch, cn_axis = sch[c_col].split(c_col.op.axis[0], 1)

        if self._l0b_first_flag:
            sch[c_col].reorder(k_outer_outer,
                               coo,
                               c_col_batch,
                               boo,
                               cn_axis,
                               coi,
                               boi,
                               nn_axis,
                               k_outer_inner,
                               reduce_kk)
            sch[c_col].compute_at(sch[res_c], c_slice_axis)
        elif self.conv_pool_fused_flag:
            coio, coii = sch[c_col].split(coi, 1)
            sch[c_col].reorder(k_outer_outer,
                               coo,
                               boo,
                               coio,
                               cn_axis,
                               coii,
                               boi,
                               nn_axis,
                               k_outer_inner,
                               reduce_kk)
            sch[c_col].compute_at(sch[res_c], c_slice_axis)
        else:
            sch[c_col].reorder(k_outer_outer,
                               coo,
                               boo,
                               cn_axis,
                               coi,
                               boi,
                               nn_axis,
                               k_outer_inner,
                               reduce_kk)
            sch[c_col].compute_at(sch[res_c], c_slice_axis)

        sch[fmap_col].compute_at(sch[c_col], boo)

        bl0_attach()

        # v200 tiling_res
        if self._v200_data_flow_type == DataFlowType.S16EltWiseS8S16:
            res_remove_pad_s16 = tensor_map["res_remove_pad_s16"]
            res_remove_pad_u8 = tensor_map["res_remove_pad_u8"]

        # v200 handle_reform
        if self._v200_data_flow_type in (DataFlowType.S16EltWiseS8,
                                         DataFlowType.S16EltWiseS8S16):
            s16_to_s8 = tensor_map["s16_to_s8"]
            _handle_deq_and_bias(s16_to_s8, c_ub_reform)
        elif self._v200_data_flow_type == DataFlowType.S32toS8:
            _handle_deq_and_bias(c_ub, c_ub_reform)
        elif self._v200_data_flow_type in (DataFlowType.S32toS16,
                                           DataFlowType.S32toFp16, \
                                           DataFlowType.V200_General_Fusion):
            _handle_deq_and_bias(c_ub)

        # ============ al1 and bl1 slice can be different with cub & CL0 =====
        outer_factor = max(al1_factor[0], bl1_factor[0])
        inner_factor = min(al1_factor[0], bl1_factor[0])
        if outer_factor % inner_factor != 0:
            err_man.raise_err_specific("conv2d", \
                "illegal value of AL1_shape & BL1_shape")
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

        # Nbuffer split axis
        not_convbn1_and_bef_flag = not self._convbn1_flag \
        and not batch_cout_reorder_flag
        if tiling["A_overhead_opt_flag"] and not_convbn1_and_bef_flag:
            shape_w = ConvParam.tiling_query_param["shape_w_nc1hwc0"]
            if (shape_w[2]*shape_w[3]) % tiling["AL0_matrix"][1] == 0:
                nbuffer_size = shape_w[2]*shape_w[3] // tiling["AL0_matrix"][1]
            else:
                nbuffer_size = shape_w[2]*shape_w[3]
            if int(k_outer_outer_inner_size % nbuffer_size) == 0 and \
                    k_outer_outer_inner_size > nbuffer_size:
                k_outer_outer_inner_outer, _ = sch[c_col].split(
                    k_outer_outer_inner, nbuffer_size)
                nbuffer_flag_al1 = True
            else:
                nbuffer_flag_al1 = False

        if l0a_load2d_flag:
            if tiling["A_overhead_opt_flag"] and not_convbn1_and_bef_flag:
                if tiling["AL1_shape"]:
                    if al1_factor[0] != 1:
                        if nbuffer_flag_al1:
                            sch[al1].compute_at(
                                sch[c_col], k_outer_outer_inner_outer)
                        else:
                            sch[al1].compute_at(sch[c_col], al1_at_ccol_axis)
                        sch[al1].allocate_at(sch[c_col], al1_at_ccol_axis)
                    else:
                        if nbuffer_flag_al1:
                            sch[al1].compute_at(
                                sch[c_col], k_outer_outer_inner_outer)
                        else:
                            sch[al1].compute_at(sch[res_c], al1_at_c_axis)
                        if reorder_flag and nbuffer_flag_al1:
                            sch[al1].allocate_at(
                                sch[res_c], al1_at_c_axis,
                                run_once_axes=[c_outer_outer_inner,
                                               c_outer_outer_outer_inner])
                        else:
                            sch[al1].allocate_at(sch[res_c], al1_at_c_axis)
                else:
                    if nbuffer_flag_al1:
                        sch[al1].compute_at(
                            sch[c_col], k_outer_outer_inner_outer)
                        sch[al1].allocate_at(
                            sch[res_c], noo,
                            run_once_axes=[c_outer_outer_inner,
                                           c_outer_outer_outer_inner])
                    else:
                        sch[al1].compute_at(sch[res_c], noo)
            else:
                if tiling["AL1_shape"]:
                    if al1_factor[0] != 1:
                        sch[al1].compute_at(sch[c_col], al1_at_ccol_axis)
                    else:
                        sch[al1].compute_at(sch[res_c], al1_at_c_axis)
                else:
                    sch[al1].compute_at(sch[res_c], noo)
        else:
            if tiling["A_overhead_opt_flag"] and not_convbn1_and_bef_flag:
                if tiling["AL1_shape"]:
                    if al1_factor[0] != 1:
                        if int(k_outer_outer_inner_size % nbuffer_size) == 0 \
                                and k_outer_outer_inner_size > nbuffer_size:
                            sch[al1].compute_at(
                                sch[c_col], k_outer_outer_inner_outer)
                            if self._dynamic_mode is None:
                                sch[fmap_col_before].compute_at(
                                    sch[c_col], k_outer_outer_inner_outer)
                        else:
                            sch[al1].compute_at(sch[c_col], al1_at_ccol_axis)
                            if self._dynamic_mode is None:
                                sch[fmap_col_before].compute_at(
                                    sch[c_col], al1_at_ccol_axis)
                        sch[al1].allocate_at(sch[c_col], al1_at_ccol_axis)
                    else:
                        sch[al1].compute_at(sch[res_c], noi)
                        if self._dynamic_mode is None:
                            sch[fmap_col_before].compute_at(sch[res_c], noi)
                        sch[al1].allocate_at(sch[res_c], al1_at_c_axis)
                else:
                    sch[al1].compute_at(sch[res_c], noi)
                    if self._dynamic_mode is None:
                        sch[fmap_col_before].compute_at(sch[res_c], noi)
                    sch[al1].allocate_at(sch[res_c], noo)
            else:
                if tiling["AL1_shape"]:
                    if al1_factor[0] != 1:
                        sch[al1].compute_at(sch[c_col], al1_at_ccol_axis)
                        if self._dynamic_mode is None:
                            sch[fmap_col_before].compute_at(
                                sch[c_col], al1_at_ccol_axis)
                    else:
                        sch[al1].compute_at(sch[res_c], al1_at_c_axis)
                        if self._dynamic_mode is None:
                            sch[fmap_col_before].compute_at(
                                sch[res_c], al1_at_c_axis)
                else:
                    if (self._aipp_fuse_flag and \
                        al1.op.attrs["input_format"] == "YUV420SP_U8") \
                        or self._conv1d_split_w_flag:
                        sch[al1].compute_at(sch[res_c], al1_at_c_axis)
                        sch[fmap_col_before].compute_at( \
                            sch[res_c], al1_at_c_axis)
                    else:
                        sch[al1].compute_at(sch[res_c], noo)
                        if self._dynamic_mode is None:
                            sch[fmap_col_before].compute_at(sch[res_c], noo)

        c04_row_major_buffer_tile()

        if self._lhisi_data_flow_type:
            if not tiling["scale_drq_split_flag"]:
                if blocks == 1:
                    sch[scale_ub].compute_at(sch[res_c], batch_outer)
                else:
                    sch[scale_ub].compute_at(sch[res_c], bido)
            else:
                sch[scale_ub].compute_at(sch[res_c], m_outer_inner_outer)

        if has_bias_ub:
            if tiling["bias_split_flag"]:
                sch[bias_ub].compute_at(sch[res_c], m_outer_outer_inner)
                if bias_optimize_flag:
                    sch[bias_ub_brc].compute_at(
                        sch[res_c], m_outer_outer_inner)
            else:
                sch[bias_ub].compute_at(sch[res_c], bido)
                if bias_optimize_flag:
                    sch[bias_ub_brc].compute_at(
                        sch[res_c], m_outer_outer_inner)

        if self.unzip_parameters.get("weight_zip_flag"):
            tiling["B_overhead_opt_flag"] = False

        out_extract_axis = -1
        if tiling["B_overhead_opt_flag"] and not self._convbn1_flag:
            if tiling["BL1_shape"]:
                if bl1_factor[0] != 1:
                    sch[bl1].compute_at(sch[c_col], coo)
                    sch[bl1].allocate_at(sch[c_col], bl1_at_ccol_axis)
                else:
                    if reorder_flag:
                        sch[bl1].compute_at(sch[c_col], coo)
                        sch[bl1].allocate_at(
                            sch[res_c], bl1_at_c_axis,
                            run_once_axes=[m_outer_outer_inner])
                    else:
                        sch[bl1].compute_at(sch[res_c], bl1_at_c_axis)
            else:
                sch[bl1].compute_at(sch[res_c], noo)
                sch[bl1].allocate_at(sch[res_c], bido)
        else:
            if tiling["BL1_shape"]:
                if self._l0b_first_flag:
                    sch[bl1].compute_at(sch[res_c], c_outer_outer_outer_inner)
                else:
                    if bl1_factor[0] != 1:
                        sch[bl1].compute_at(sch[c_col], bl1_at_ccol_axis)
                        if self._dynamic_mode is None and \
                           dim_map['out_img_shape'][2] > \
                           tiling["CL0_matrix"][1] * 16:
                            out_extract_axis = m_outer_outer_inner
                    else:
                        sch[bl1].compute_at(sch[res_c], bl1_at_c_axis)
                        if self._dynamic_mode is None and reorder_flag \
                           and al1_factor[1] > tiling["block_dim"][2]:
                            out_extract_axis = m_outer_outer_outer_inner
                if not self._dynamic_mode and out_extract_axis == -1 and \
                    dim_map["out_img_shape"][0] > 1:
                    out_extract_axis = noo
            else:
                if self._l0b_first_flag:
                    sch[bl1].compute_at(sch[res_c], c_outer_outer_outer_inner)
                elif tiling["BL1_shape"] is not None:
                    if self.unzip_parameters.get(
                         "weight_zip_flag") and blocks == 1:
                         sch[bl1].compute_at(sch[res_c], batch_outer)
                    else:
                        sch[bl1].compute_at(sch[res_c], bido)
                else:
                    if tiling["BL0_matrix"] and self._dynamic_mode is \
                    None and dim_map['out_img_shape'][2] > \
                       tiling["CL0_matrix"][1] * 16:
                        out_extract_axis = m_outer_outer_inner
                    elif not self._dynamic_mode and out_extract_axis == -1 \
                    and dim_map["out_img_shape"][0] > 1:
                        out_extract_axis = noo


        yuv_align, yuv_pad_align = aipp_conv_fuse_yuv_align()
        conv1d_w_split_tile()
        double_buffer()
        ############################ intrin mapping ###########################
        intrin_mapping(weight, tiling)
        ########################### cube schedule end #########################
        _body_ops_compute_at()
        _body_ops_convbn1_flag(multiout_ub2)
        _input_ops_compute_at()

        # mean_out
        if self._convbn1_flag:
            sch[sum_x_global].emit_insn(c_rf_outer_outer_inner, "dma_copy")
            sch[sum_x_ub_rf].emit_insn(m_outer_inner_inner, \
                    "vector_dichotomy_add_for_bn_reduce")

        for lop in self._op_graph.body_ops:
            pragma_flag = (("convolution" not in lop["op"]) or \
                ("convolution_A" in lop["op"])) or \
            (self._fused_flag and (lop["op"] == "convolution_C"))
            if pragma_flag:
                lop["tensorize_axis"] = self._schedule[
                    lop["dst_buffer"]].op.axis[0]
                if self._lhisi_data_flow_type or self._conv_quant_fused_flag\
                    or self._v200_data_flow_type in \
                        (DataFlowType.V200_General_Fusion,):
                    self.__pragma_for_op_vector(lop, res)
                if "_convolution_A" in lop["op"]:
                    lop["op"] = lop["op"].replace("_convolution_A", "")
                if lop["op"] == "broadcast":
                    self._schedule[lop["dst_buffer"]].emit_insn(\
                        lop["dst_buffer"].op.axis[0], 'vector_dup')
                if lop["op"] == "broadcast_for_tensor":
                    sch[lop["dst_buffer"]].compute_inline()
                if "fusion_fmap_select" in lop["op"]:
                    continue
                elif self._convbn1_flag:
                    self.__pragma_for_op(lop, c_outer_inner_inner, batch_inner_inner, \
                        fmap, c_ub, None, d_pad, c_pragma_axis=c_pragma_axis, \
                        tiling=tiling)
                else:
                    self.__pragma_for_op(lop, c_outer_inner_inner, batch_inner_inner, \
                        fmap, c_ub, c_pragma_axis=c_pragma_axis, tiling=tiling)

        self.__pragma_for_input(tensor_map)
        if self._lhisi_data_flow_type:
            sch[scale_ub].emit_insn(
                scale_ub.op.axis[0], 'dma_copy')
            for buffer_eltwise in v100_cache_buffer:
                sch[buffer_eltwise].emit_insn(
                    buffer_eltwise.op.axis[0], 'dma_copy')

        # for L1 append tensor in cce kernel function
        l1_tensor_map = {}
        if self._fmap_l1_addr_flag == "nothing":
            l1_tensor_map = None
        else:
            if self._input_memory_type[0] in (0, 2) \
                and self._l1_fusion_type in (0, 1):
                l1_tensor_map[fmap] = al1
                if self._fmap_l1_valid_size>0:
                    sch[al1].set_storage_bound(self._fmap_l1_valid_size)
            else:
                l1_tensor_map = None
        util.L1CommonParam.l1_fusion_tensors_map = l1_tensor_map

        if self._dynamic_mode:
            sch[al1].set_storage_bound(al1_bound)
            # disable_allocate
            sch.disable_allocate(cce.scope_cbuf)
            sch.disable_allocate(cce.scope_ca)
            sch.disable_allocate(cce.scope_cb)
            sch.disable_allocate(cce.scope_cc)
            sch.disable_allocate(cce.scope_ubuf)

            # mem_unique
            sch[al1].mem_unique()
            sch[fmap_col].mem_unique()
            if tiling["BL1_shape"] is not None:
                sch[bl1].mem_unique()
            sch[bl0].mem_unique()
            sch[c_col].mem_unique()

            if not self._fused_flag:
                sch[c_ub].mem_unique()
        _conv_pooling_optm()

        if self._dynamic_mode:
            return True

        _conv_pooling_optm()
        tensor_map.clear()
        dim_map.clear()
        tiling.clear()
        return True

    def __pragma_for_input(self, tensor_map):

        def _pragma_continue(lop):
            if self._lhisi_data_flow_type:
                return True
            if "convolution" in lop["op"]:
                return True
            if "max_pooling_pad_" in lop["op"]:
                return True
            if lop["dst_buffer"] == tensor_map.get("bias"):
                return True
            if ("bias_tensor" in lop["op"]) and ("bias" in tensor_map.keys()):
                return True
            if ("bias_tensor" in lop["op"]) \
                and ("fp16_bias" in tensor_map.keys()):
                self._schedule[lop["cache_buffer"]] \
                    .emit_insn(lop["cache_buffer"].op.axis[0], 'dma_copy')
                return True
            if "dma_copy" in lop["next_op"][0]["dst_buffer"].op.attrs:
                return True
            if "fusion_fmap_select" in lop["next_op"][0]["op"]:
                return True

            return False

        for lop in self._op_graph.input_ops:
            if _pragma_continue(lop):
                continue
            if lop["dst_buffer"].name == 'compress_index' or \
                    lop["dst_buffer"].name == "Filter":
                continue
            if "cache_buffer" in lop:
                self._schedule[lop["cache_buffer"]].emit_insn(
                    lop["cache_buffer"].op.axis[0], 'dma_copy')

    def __pragma_for_op_vector(self, lop, res):
        # for vector auto
        cache_buffer = lop["dst_buffer"]
        tensorize_axis = lop["tensorize_axis"]
        vector_auto_list = ["dequant2_vector", "dequant2_scale",
                            "dequant_relu", "scale_sqrt_ub",
                            "offset_ub"]
        if lop["dst_buffer"] != res:
            if "elewise" in lop["op"]:
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, 'vector_auto')
            if lop["op"] in vector_auto_list:
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, 'vector_auto')
            if self._conv_quant_fused_flag and ("convolution_C" in lop["op"] \
                or "conv_vector_bias_add" in lop["op"]):
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, 'vector_auto')

    def _get_elmwise_instr(self, elm_instr):
        emit_insn_pragma = self._emit_insn_map.get(elm_instr)
        if emit_insn_pragma:
            out_instr = emit_insn_pragma
        else:
            out_instr = elm_instr

        return out_instr

    def __pragma_for_op(self, lop, c_outer_inner_inner=None, batch_inner_inner = None,
                        feature_map=None, c_ub=None, scale_ub_reform=None,
                        res_c=None, c_pragma_axis=None, tiling=None):
        # for not in conv op pragma

        def _lhisi_data_flow_type_emit_insn(tiling=None):

            def _handle_lx_fusion_lhisi_data_flow():
                if 'write_select' in lop["op"]:
                    if ConvParam.swrite_flag:
                        self._schedule[lop["dst_buffer"]].emit_insn(
                            lop["dst_buffer"].op.axis[0], 'phony_insn')

                        align_length = int(lop["dst_buffer"].op.attrs["HWC0"])
                        self._schedule[self._res_tensor].bind_buffer(
                            self._res_tensor.op.axis[1], align_length, 0)
                    else:
                        align_length = int(cache_buffer.op.attrs["HWC0"])
                        self._schedule[cache_buffer].bind_buffer(
                            lop["dst_buffer"].op.axis[1], align_length, 0)
                        if self._lhisi_data_flow_type == \
                                DataFlowTypeLhisi.S32toFp16S8:
                            self._schedule[cache_buffer].allocate_root()
                            self._schedule[cache_buffer].emit_insn(
                                tensorize_axis, 'dma_copy')
                        else:
                            self._schedule[cache_buffer].emit_insn(
                                c_outer_inner_inner, 'dma_copy')
                elif "output_ub" in lop["op"]:
                    self._schedule[cache_buffer].emit_insn(
                        tensorize_axis, 'dma_copy')

            def _reform_emit_insn_optimize():
                if hw_in_ub > c_reform_vector.shape[2].value \
                    and hw_floor_align >= 16:
                    abatch, ac1, ahw = axis_list
                    ahwo, ahwi = self._schedule[c_reform_vector] \
                                    .split(ahw, hw_floor_align)
                    self._schedule[c_reform_vector]. \
                        reorder(abatch, ahwo, ac1, coo, ahwi)
                    self._schedule[c_reform_vector].emit_insn(
                        ac1, "vector_auto")
                else:
                    self._schedule[c_reform_vector].reorder(coo, *axis_list)
                    self._schedule[c_reform_vector].emit_insn(
                        self._schedule[c_reform_vector].op.axis[2],
                        "vector_auto")

            def _double_out_emit_insn():
                if not self._l0b_first_flag:
                    self._schedule[cache_buffer].emit_insn(c_outer_inner_inner, 'phony_insn')
                else:
                    self._schedule[cache_buffer].emit_insn(batch_inner_inner, 'phony_insn')

            if lop["op"] in dma_move_list:
                if not self._conv_quant_fused_flag:
                    if self._lhisi_data_flow_type == \
                            DataFlowTypeLhisi.S32toFp16S8 or \
                            self._write_select or (self._v200_data_flow_type \
                            in (DataFlowType.V200_General_Fusion,) and self._res_tensor.op.tag != "quant"):
                        self._schedule[cache_buffer].emit_insn(
                            tensorize_axis, 'dma_copy')
                    else:
                        self._schedule[cache_buffer].emit_insn(
                            c_outer_inner_inner, 'dma_copy')
            elif "reform" in lop["op"]:
                c_reform_vector = lop['dst_buffer']
                ndim = len(self._schedule[c_reform_vector].op.axis)
                factor = CUBE_MKN["float16"]["mac"][1]
                coo, _ = self._schedule[c_reform_vector].split(
                    self._schedule[c_reform_vector].op.axis[ndim - 1], factor)
                axis_list = self._schedule[c_reform_vector].op.axis[0:ndim - 1]
                hw_in_ub = tiling["CUB_matrix"][1]*tiling["CUB_matrix"][3]
                hw_floor_align = c_reform_vector.shape[2].value//16*16
                _reform_emit_insn_optimize()
            elif lop["op"] == "cast_i8_ub":
                round_mode_emit_insn = \
                'vector_conv_%s' % self._lhisi_dequant_quant_para[
                    'quant_round'].value.lower()
                if cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310"):
                    round_mode_emit_insn = 'vector_conv'
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, round_mode_emit_insn)
            elif lop["op"] == "conv_virtual_res":
                _double_out_emit_insn()
            elif lop["op"] == "input_ub":
                if self._lhisi_dequant_quant_para["quant_padding"] or \
                ConvParam.TENSOR_MAP["quant_input"].op.attrs["c_out"].value \
                % 2 == 1:
                    self._schedule[cache_buffer].emit_insn(
                        tensorize_axis, 'dma_padding')
            _handle_lx_fusion_lhisi_data_flow()

        def handle_lx_fusion():
            if 'write_select' in lop["op"]:
                if ConvParam.swrite_flag:
                    self._schedule[lop["dst_buffer"]].emit_insn(
                        lop["dst_buffer"].op.axis[0], 'phony_insn')

                    align_length = int(lop["dst_buffer"].op.attrs["HWC0"])
                    self._schedule[self._res_tensor].bind_buffer(
                        self._res_tensor.op.axis[1], align_length, 0)

                else:
                    align_length = int(lop["dst_buffer"].op.attrs["HWC0"])
                    self._schedule[lop["dst_buffer"]].bind_buffer(
                        lop["dst_buffer"].op.axis[1], align_length, 0)
                    self._schedule[lop["dst_buffer"]].emit_insn(
                        c_outer_inner_inner, 'dma_copy')
            elif "output_ub" in lop["op"]:
                self._schedule[cache_buffer].emit_insn(
                    self._schedule[cache_buffer].op.axis[0], 'dma_copy')
            else:
                pass

        def _vector_ci32():
            if feature_map.op.input_tensors:
                fm_w = feature_map.shape[3].value
            else:
                fm_w = feature_map.op.shape[3].value
            stride_w = c_ub.op.attrs['stride'][1].value
            if 1 < stride_w < fm_w:
                axis_dma_pad = self._schedule[cache_buffer].op.axis[3]
            elif stride_w > 1 and fm_w < stride_w:
                axis_dma_pad = self._schedule[cache_buffer].op.axis[4]
            elif stride_w == 1 and fm_w == 1:
                axis_dma_pad = self._schedule[cache_buffer].op.axis[4]
            else:
                axis_dma_pad = self._schedule[cache_buffer].op.axis[2]
            self._schedule[cache_buffer].emit_insn(axis_dma_pad, "dma_padding")

        def __convolution_c_emit_insn():
            if self._convbn1_flag:
                self._schedule[c_ub].reused_by(cache_buffer)
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], 'dma_copy')
            if self.conv_pool_fused_flag and \
                    "fp16_bias" not in ConvParam.TENSOR_MAP:
                self._schedule[cache_buffer].compute_inline()

        def check_int8_data_flow_type():
            if self._lhisi_data_flow_type:
                return True
            if self._conv_quant_fused_flag:
                return True
            if lop["op"] in dma_move_list:
                return True
            if lop["op"] in ("cast_i8_ub", "conv_virtual_res", "input_ub"):
                return True
            if "reform" in lop["op"]:
                return True
            return False

        def handle_remove_pad():
            if self._v200_data_flow_type:
                pass
            else:
                self._schedule[cache_buffer].emit_insn(
                    c_pragma_axis, 'dma_copy')

        op_cmd = lop["op"].split("_")
        cache_buffer = lop["dst_buffer"]
        tensorize_axis = lop["tensorize_axis"]
        dma_move_list = ["quant", "res_out_fp16"]

        if ConvParam.swrite_dequant_flag:
            dma_move_list = ["strided_write", "res_out_fp16"]

        reform_list = ['conv_vector_reform5_reg',
                       'conv_vector_reform4',
                       'conv_vector_reform4_vmuls',
                       'conv_vector_reform5_vadds',
                       'conv_vector_reform4_vadds',
                       'conv_vector_reform5']

        if  check_int8_data_flow_type():
            _lhisi_data_flow_type_emit_insn(tiling)
        elif op_cmd[0].lower() == "elewise":
            ele_instr = self._get_elmwise_instr(lop["op"])
            self._schedule[cache_buffer].emit_insn(tensorize_axis, ele_instr)
        elif lop["op"] == "cast_0" and self._convbn1_flag:
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], "dma_copy")
        elif lop["op"] == "cast_1" and self._convbn1_flag:
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], "phony_insn")
            self._schedule[res_c].reused_by(cache_buffer)
        elif op_cmd[0].lower() == "cast" and self._convbn1_flag:
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], "vector_conv")
        elif lop["op"] == 'conv_vector_bias_add':
            self._schedule[cache_buffer].emit_insn(
                tensorize_axis, "vector_add")
        elif lop["op"] in reform_list:
            c_reform_vector = lop['dst_buffer']
            ndim = len(self._schedule[c_reform_vector].op.axis)
            factor = CUBE_MKN["float16"]["mac"][1]  # Only Support fp16->uint8
            coo, _ = self._schedule[c_reform_vector].split(
                self._schedule[c_reform_vector].op.axis[ndim - 1], factor)
            axis_list = self._schedule[c_reform_vector].op.axis[0:ndim - 1]
            self._schedule[c_reform_vector].reorder(coo, *axis_list)
            self._schedule[c_reform_vector].emit_insn(
                self._schedule[c_reform_vector].op.axis[2], "vector_auto")
        elif lop["op"] == 'conv_vector_reform4_vmul':
            c_reform_vector = lop['dst_buffer']
            ndim = len(self._schedule[c_reform_vector].op.axis)
            factor = CUBE_MKN["float16"]["mac"][1]  # Only Support fp16->uint8
            coo, _ = self._schedule[c_reform_vector].split(
                self._schedule[c_reform_vector].op.axis[ndim - 1], factor)
            axis_list = self._schedule[c_reform_vector].op.axis[0:ndim - 1]
            self._schedule[c_reform_vector].reorder(coo, *axis_list)
            self._schedule[c_reform_vector].emit_insn(
                self._schedule[c_reform_vector].op.axis[2], "vector_mul")
            if scale_ub_reform is not None:
                scale_ub_reform.pragma(scale_ub_reform.op.axis[0], 'empty')
            else:
                pass
        elif "conv_vector_vmul_vector" in lop["op"]:
            self._schedule[cache_buffer].emit_insn(
                tensorize_axis, "vector_mul")
        elif "conv_vector_vumls_reg" in lop["op"]:
            self._schedule[cache_buffer].emit_insn(
                tensorize_axis, "vector_muls")
        elif "conv_vector_vadds_reg" in lop["op"]:
            self._schedule[cache_buffer].emit_insn(
                tensorize_axis, "elewise_single_VS_adds_with_reg")
        elif "conv_vector_Ci32" in lop["op"]:
            _vector_ci32()
        elif lop["op"] == 'convolution_C':
            __convolution_c_emit_insn()
        elif lop["op"] == 'conv_vector_remove_pad':
            handle_remove_pad()
        elif lop["op"] == "requant_s16_data_transfer" and \
                self._v200_data_flow_type == DataFlowType.S16EltWiseS8:
            self._schedule[cache_buffer].emit_insn(
                c_outer_inner_inner, 'dma_copy')
        elif lop["op"] in ("dequant_remove_pad", "requant_remove_pad", \
                            "dequant_s16_remove_pad") and \
            self._v200_data_flow_type in (
                    DataFlowType.S32toS8,
                    DataFlowType.S32toS16,
                    DataFlowType.S32toFp16):
            self._schedule[cache_buffer].emit_insn(
                c_outer_inner_inner, 'dma_copy')
        elif lop["op"] == "dequant_remove_pad" and self._v200_data_flow_type \
            in (DataFlowType.V200_General_Fusion,):
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], "dma_copy")
        elif lop["op"] == 'pooling2d_max_max_pool_res':
            self._schedule[cache_buffer].emit_insn(
                c_outer_inner_inner, 'dma_copy')
        else:
            handle_lx_fusion()


class AutoScheduleOp:
    """
    class of AutoScheduleOp
    """
    def __init__(self, *init_args):
        if len(init_args) <= 2 and isinstance(init_args[0], tvm.tensor.Tensor):
            res_tensor = init_args[0]
            self.weight_zip_flag = False
            self._op = []
            self.body_ops = []
            self.input_ops = []
            self.output_ops = []
            self._res_tensor = res_tensor
            self._visited_node_list = []
            self.fused_double_operand_num = 0
            self.fusion_type = 0
            self.__scrapy_tensor_graph(self._res_tensor)
            self.__connect_op()
            self._end_op = self.get_op_by_tensor(self._res_tensor)
            self.__analyse_input_output()
            self.__analyse_v200_data_flow_type()
            self.__analyse_fused_double_operand_num()
            self.__analyse_fusion_type()

    def __split_tensor(self, tensor):
        def check_v200_version(tmp_op, tensor):
            if tmp_op["op"] == "dequant_s16_remove_pad":
                ConvParam.TENSOR_MAP["dequant_s16_remove_pad"] = tensor
            if tmp_op["op"] in ("requant_vector", "requant_scale"):
                ConvParam.TENSOR_MAP["c_ub"] = tensor
            if "dequant_s16" in tmp_op["op"]:
                ConvParam.TENSOR_MAP["c_ub"] = tensor
            if tmp_op["op"] in ("dequant_vector", "dequant_scale"):
                ConvParam.TENSOR_MAP["c_ub"] = tensor
            if "data_transfer" in tmp_op["op"]:
                ConvParam.TENSOR_MAP["data_transfer"] = tensor
            if "requant_s16_vector" in tmp_op["op"] or "requant_s16_scale" in tmp_op["op"]:
                ConvParam.TENSOR_MAP["s16_to_s8"] = tensor
            if "requant_s16_vaddrelu" in tmp_op["op"]:
                ConvParam.TENSOR_MAP["requant_s16_vaddrelu"] = tensor
            if "requant_s16_vadd" in tmp_op["op"]:
                ConvParam.TENSOR_MAP["requant_s16_vadd"] = tensor

        def handle_int8_tensor_map():
            tensor_map_lhisi_in = [('dequant1', 'c_ub'), 'dequant_remove_pad',
                                   'dequant_relu',
                                   ('dequant2', 'c_ub_dequant_sqrt'),
                                   ('input_ub', 'quant_input'),
                                   ('reform_by_vmuls', 'reform_vmuls'),
                                   ('scale_sqrt_ub', 'vmuls'),
                                   ('offset_ub', 'offset_add'),
                                   ('cast_i8_ub', 'cast2s8'),
                                   ('reform_by_vadds', 'reform_vadds')]
            tensor_map_lhisi_equal = ['quant']
            for item in tensor_map_lhisi_in:
                if isinstance(item, tuple):
                    name, key = item
                else:
                    name, key = item, item
                if name in tmp_op['op']:
                    ConvParam.TENSOR_MAP[key] = tensor
            for item in tensor_map_lhisi_equal:
                if isinstance(item, tuple):
                    name, key = item
                else:
                    name, key = item, item
                if name == tmp_op['op']:
                    ConvParam.TENSOR_MAP[key] = tensor

        tmp_op = {}
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
        if is_support_v200():
            check_v200_version(tmp_op, tensor)
        if "weight_unzip" in tmp_op["op"]:
            self.weight_zip_flag = True
        # this c_ub is stand for tensor first in ub which in conv2d quant fuse
        if cce_conf.get_soc_spec("SOC_VERSION") in \
        ("Ascend310", "Hi3796CV300ES") or is_support_v200():
            handle_int8_tensor_map()

        return tmp_op

    def __scrapy_tensor_graph(self, res_tensor):
        tmp_op = self.__split_tensor(res_tensor)
        if res_tensor.op.name in self._visited_node_list:
            return
        if len(tmp_op["src_buffer"]) > 1 or "cast" in tmp_op["op"]:
            self.fused_double_operand_num += len(tmp_op["src_buffer"])
        self._visited_node_list.append(res_tensor.op.name)
        self._op.append(tmp_op)
        for i in tmp_op["src_buffer"]:
            if tmp_op["op"] == "convolution_c_col":
                i.tag = "convolution_Input"
            if tmp_op["op"] == "convolution_im2col_fractal_v2_convolution_Input":
                i.tag = "convolution_A"
            if tmp_op["op"] == "convolution_im2col_row_major" and \
            "fmap_l1" not in tmp_op["src_buffer"][0].name and \
            "aipp_res" not in tmp_op["src_buffer"][0].name:
                i.tag = "convolution_A"
            if tmp_op["op"] == "convolution_al1_load2d":
                i.tag = "convolution_A"
            if "fmap_l1" in tmp_op["op"] or "aipp_res" in tmp_op["op"]:
                i.tag = "convolution_A"
            if tmp_op["op"] == "convolution_bias_l0c":
                i.tag = "convolution_bias_tensor"
            if "aipp_res" in tmp_op["op"]:
                ConvParam.TENSOR_MAP["aipp_input_format"] = \
                tmp_op["dst_buffer"].op.attrs["input_format"]
            self.__scrapy_tensor_graph(i)

    def __connect_op(self):
        for lop in self._op:
            lop["prev_op"] = []
            lop["next_op"] = []

        for lop in self._op:
            for src_tensor in lop["src_buffer"]:
                tmp_op = self.get_op_by_tensor(src_tensor)
                lop["prev_op"].append(tmp_op)
                tmp_op["next_op"].append(lop)

    def get_op_by_tensor(self, tensor):
        """
        get op by tensor

        Parameters
        ----------
        tensor: the source tensor

        Returns
        -------
        tensor : op
        """
        for i in self._op:
            if i["dst_buffer"].same_as(tensor):
                return i
        return None

    def __analyse_input_output(self):
        input_ops = []
        output_ops = []
        body_ops = []
        input_tensor_name = []
        body_tensor_name = []

        for lop in self._op:
            if (not lop["prev_op"]) and (lop["op"] != "broadcast"):
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

        self.input_ops = input_ops
        self.output_ops = output_ops
        self.body_ops = body_ops

    def __analyse_v200_data_flow_type(self):
        tag_to_type_map = {
            "data_transfer": DataFlowType.S32toS8,
            "dequant_s16_vector": DataFlowType.S32toS16,
            "dequant_s16_scale": DataFlowType.S32toS16,
            "requant_s16_vector": DataFlowType.S16EltWiseS8,
            "requant_s16_scale": DataFlowType.S16EltWiseS8,
            "res_remove_pad_u8": DataFlowType.S16EltWiseS8S16,
            "dequant_vector": DataFlowType.S32toFp16,
            "dequant_scale": DataFlowType.S32toFp16}
        out_src = self.output_ops[0]['src_buffer'][0].op.tag
        if self._res_tensor.op.tag == "strided_write":
            out_src = self.output_ops[0][
                'src_buffer'][0].op.input_tensors[0].op.tag

        weight = ConvParam.TENSOR_MAP["filter"]
        if out_src in tag_to_type_map.keys():
            ConvParam.TENSOR_MAP["v200_data_flow_type"] = \
            tag_to_type_map[out_src]
        elif is_support_v200() and weight.dtype == "int8" and \
            out_src != 'convolution_C_UB':
            ConvParam.TENSOR_MAP["v200_data_flow_type"] = \
                DataFlowType.V200_General_Fusion
        if "pooling2d_max" in out_src:
            MaxPoolParam.tensor_map["is_conv_pool_fused"] = True
        else:
            MaxPoolParam.tensor_map["is_conv_pool_fused"] = False


    def __analyse_fused_double_operand_num(self):
        tmp = self.fused_double_operand_num
        total_ub_op_num = len(self.body_ops) - CONV_OP_NUM
        self.fused_double_operand_num = tmp if tmp <= total_ub_op_num \
            else total_ub_op_num


    def __analyse_fusion_type(self):

        def __get_fusion_type_dict_enhance():
            fusion_type_dict_enhance = {}
            fusion_type_list_all = []
            for fusion_key_temp in fusion_type_dict.keys():
                fusion_type_list_all.append(fusion_key_temp)
            for fusion_type_list_temp in fusion_type_list_all:
                fusion_type_temp = fusion_type_dict.get(
                    fusion_type_list_temp,
                    fusion_type_dict["fusion_type_unknow"])
                fusion_type_list_temp = fusion_type_list_temp[
                    len('fusion_type'):]
                fusion_type_list_temp = fusion_type_list_temp.split('_')
                fusion_type_list_temp.sort()
                fusion_type_list_temp = "_".join(fusion_type_list_temp)
                fusion_type_dict_enhance.update(
                    {fusion_type_list_temp: fusion_type_temp})
            return fusion_type_dict_enhance

        op_to_fusion_type_map = {
            "conv_vector_remove_pad": OpFusionype.conv,
            "convolution_C_UB": OpFusionype.conv,
            "convolution_c_col": OpFusionype.conv,
            "convolution_C": OpFusionype.conv,
            "convolution_im2col_fractal_convolution_Input": OpFusionype.conv,
            "convolution__im2col_fractal_convolution_Input": OpFusionype.conv,
            #for read select
            "strided_read_convolution_A": OpFusionype.op_empty,
            "aipp_res_convolution": OpFusionype.op_empty,
            "fusion_fmap_select_convolution_A": OpFusionype.conv,
            "convolution_al0_load2d_convolution_Input": OpFusionype.conv,
            "convolution_im2col_row_major": OpFusionype.conv,
            "convolution_row_major_reshape": OpFusionype.conv,
            "convolution_fmap_l1_c0_optim": OpFusionype.conv,
            "fmap_l1": OpFusionype.conv,
            "convolution_al1_load2d": OpFusionype.conv,
            "conv_vector_bias_add": OpFusionype.bias,
            "elewise_single_relu": OpFusionype.elewise_one_op,
            "elewise_binary_add": OpFusionype.elewise_two_op,
            "elewise_binary_max": OpFusionype.elewise_two_op,
            "elewise_single_VS_mul": OpFusionype.elewise_one_op,
            "output_ub_4d": OpFusionype.dma_common,
            "output_ub_5d": OpFusionype.dma_common,
            "write_select": OpFusionype.dma_common,
            "strided_write": OpFusionype.op_empty,
            # quant bias
            "convolution_c_col_bias": OpFusionype.bias_quant,
            "convolution_bias_l0c": OpFusionype.bias_quant,
            "convolution_bias_ub_brc_convolution_bias_tensor": \
                OpFusionype.bias_quant,
            # quant case data_flow 0
            "dequant1_vector": OpFusionype.conv_dequant,
            "dequant1_scale": OpFusionype.conv_dequant,
            "dequant2_scale": OpFusionype.conv_dequant,
            "dequant2_vector": OpFusionype.conv_dequant,
            "dequant_relu": OpFusionype.conv_dequant,
            "dequant_remove_pad": OpFusionype.conv_dequant,
            "dequant_scale": OpFusionype.conv_dequant,
            # requant
            "data_transfer": OpFusionype.conv_requant,
            "requant_scale": OpFusionype.conv_requant,
            "requant_remove_pad": OpFusionype.conv_requant,
            # quant case data_flow 1 2
            "quant": OpFusionype.conv_quant,
            "cast_i8_ub": OpFusionype.conv_quant,
            "offset_ub": OpFusionype.conv_quant,
            "input_ub": OpFusionype.conv_quant,
            "reform_by_vmuls": OpFusionype.conv_quant,
            "reform_by_vadds": OpFusionype.conv_quant,
            "scale_sqrt_ub": OpFusionype.conv_quant,
            # quant case data_flow 2
            "res_out_fp16": OpFusionype.double_out,
            "conv_virtual_res": OpFusionype.double_out,
            "mean_out": OpFusionype.bn_op,
            "cast_1": OpFusionype.elewise_one_op,
            "cast_0": OpFusionype.elewise_one_op,
            "elewise_binary_mul": OpFusionype.elewise_two_op,
            "cast_0_ub": OpFusionype.elewise_one_op,
            "convolution_c_ub": OpFusionype.dma_common}

        fusion_type_dict = {
            # unknown
            "fusion_type_unknow": 0,
            # fp16
            "fusion_type_1": 1,
            "fusion_type_2_1": 2,
            # Conv2d+Relu
            "fusion_type_3_1": 3,
            "fusion_type_3_2_1": 4,
            "fusion_type_4_3_1": 7,
            "fusion_type_3_4_1": 7,
            "fusion_type_4_3_2_1": 8,
            "fusion_type_3_4_2_1": 8,
            "fusion_type_3_4_1_2": 8,
            # quant
            "fusion_type_6_1": 9,
            "fusion_type_6_1_9": 10,
            "fusion_type_7_6_1": 11,
            "fusion_type_7_6_1_9": 12,
            "fusion_type_7_3_4_6_1": 13,
            "fusion_type_7_4_3_6_1": 13,
            "fusion_type_7_3_4_6_1_9": 14,
            "fusion_type_7_4_3_6_1_9": 14,
            "fusion_type_8_7_3_4_6_1": 15,
            "fusion_type_8_7_4_3_6_1": 15,
            "fusion_type_8_7_3_4_6_1_9": 16,
            "fusion_type_8_7_4_3_6_1_9": 16,
            "fusion_type_8_7_4_6_1_9_3": 16,

            # l1 fusion fp16
            "fusion_type_1_2": 2,
            "fusion_type_5_3_4_1": 5,
            "fusion_type_5_4_3_1": 5,
            "fusion_type_3_4_5_1": 5,
            "fusion_type_4_3_5_1": 5,
            "fusion_type_5_3_4_2_1": 6,
            "fusion_type_5_4_3_2_1": 6,
            "fusion_type_3_4_5_2_1": 6,
            "fusion_type_4_3_5_2_1": 6,
            # add : conv_res, data_other better
            "fusion_type_3_4_1_5": 5,
            "fusion_type_4_3_1_5": 5,
            "fusion_type_3_4_2_1_5": 6,
            "fusion_type_4_3_2_1_5": 6,

            # l1 fusion quant
            "fusion_type_5_7_3_4_6_1": 13,
            "fusion_type_5_7_4_3_6_1": 13,
            "fusion_type_7_3_4_6_1_5": 13,
            "fusion_type_7_4_3_6_1_5": 13,
            "fusion_type_5_7_3_4_6_1_9": 14,
            "fusion_type_5_7_4_3_6_1_9": 14,
            "fusion_type_7_3_4_6_1_9_5": 14,
            "fusion_type_7_4_3_6_1_9_5": 14,
            "fusion_type_3_4_6_1_9": 14,
            "fusion_type_4_3_6_1_9": 14,
            "fusion_type_5_3_4_6_1_9": 14,
            "fusion_type_5_4_3_6_1_9": 14,
            "fusion_type_8_5_7_3_4_6_1": 15,
            "fusion_type_8_5_7_4_3_6_1": 15,
            "fusion_type_8_7_3_4_6_1_5": 15,
            "fusion_type_8_7_4_3_6_1_5": 15,
            "fusion_type_8_5_7_3_4_6_1_9": 16,
            "fusion_type_8_5_7_4_3_6_1_9": 16,
            "fusion_type_8_7_3_4_6_1_9_5": 16,
            "fusion_type_8_7_4_3_6_1_9_5": 16,
            # add data_other out
            "fusion_type_7_3_4_5_6_1": 13,
            "fusion_type_7_4_3_5_6_1": 13,
            "fusion_type_7_3_4_5_6_1_9": 14,
            "fusion_type_7_4_3_5_6_1_9": 14,
            "fusion_type_8_7_3_4_5_6_1": 15,
            "fusion_type_8_7_4_3_5_6_1": 15,
            "fusion_type_8_7_3_4_5_6_1_9": 16,
            "fusion_type_8_7_4_3_5_6_1_9": 16,
            "fusion_type_3_4_6_1": 17,
            "fusion_type_4_3_6_1": 17,
            "fusion_type_5_3_4_6_1": 17,
            "fusion_type_5_4_3_6_1": 17,
            "fusion_type_4_2_1_3": 18,
            "fusion_type_5_4_2_1_3": 19,
            "fusion_type_5_4_1_2_3": 19,
            "fusion_type_11_3_3_3_1_5_4": 20,
            # Conv2d+Eltwise
            "fusion_type_4_1": 22,
            "fusion_type_4_2_1": 23,
            # Conv2d+LeakyRelu
            "fusion_type_4_1_3": 5,
            # Conv2d+LeakyReLU+Eltwise
            "fusion_type_4_4_1_3": 24,
            "fusion_type_4_4_2_1_3": 25,
            # Conv2d+Requant
            "fusion_type_10_1": 26,
            "fusion_type_10_1_9": 27,
            # Conv2d+AscendDequant+Eltwise+AscendQuant
            "fusion_type_7_4_6_1": 28,
            "fusion_type_7_4_6_1_9": 29,
            # Conv2d+AscendDequant+Eltwise
            "fusion_type_4_6_1": 30,
            "fusion_type_4_6_1_9": 31,
            # Conv2d+AscendDequant+Eltwise+AscendQuant double out
            "fusion_type_8_7_4_6_1": 32,
            "fusion_type_8_7_4_6_1_9": 33,
            # Conv2d+AscendDequant+LeakyReLU+Eltwise
            "fusion_type_4_4_6_1_3": 34,
            "fusion_type_4_4_6_1_9_3": 35,
            # Conv2d+AscendDequant+LeakyRelu+AscendQuant double out
            "fusion_type_8_7_4_6_1_3": 15,
            "fusion_type_8_7_4_6_1_9_3": 16,
            # Conv2d+AscendDequant+LeakyReLU+Eltwise+AscendQuant double out
            "fusion_type_8_7_4_4_6_1_3": 36,
            "fusion_type_8_7_4_4_6_1_9_3": 37,
            # Conv2d+AscendDequant+LeakyRelu
            "fusion_type_4_6_1_3": 17,
            "fusion_type_4_6_1_9_3": 21,
            # Conv2d+AscendDequant+LeakyReLU+Eltwise+AscendQuant
            "fusion_type_7_4_4_6_1_3": 38,
            "fusion_type_7_4_4_6_1_9_3": 39,
            # Conv2d+AscendDequant+LeakyRelu+AscendQuant
            "fusion_type_7_4_6_1_3": 13,
            "fusion_type_7_4_6_1_9_3": 14

        }

        conv_flag = 0
        conv_dequant_flag = 0
        conv_quant_flag = 0
        conv_requant_flag = 0
        double_out_flag = 0
        bias_flag = 0
        dma_common_flag = 0
        temp_flag = None
        fusion_type_list = "fusion_type"
        for lop in self.body_ops:
            op_to_fusion_type = op_to_fusion_type_map.setdefault(
                lop["op"], OpFusionype.default)
            if op_to_fusion_type == OpFusionype.conv:
                temp_flag = conv_flag
                conv_flag += 1
            elif op_to_fusion_type == OpFusionype.conv_dequant:
                temp_flag = conv_dequant_flag
                conv_dequant_flag += 1
            elif op_to_fusion_type == OpFusionype.conv_requant:
                temp_flag = conv_requant_flag
                conv_requant_flag += 1
            elif op_to_fusion_type == OpFusionype.conv_quant:
                temp_flag = conv_quant_flag
                conv_quant_flag += 1
            elif op_to_fusion_type == OpFusionype.double_out:
                temp_flag = double_out_flag
                double_out_flag += 1
            elif op_to_fusion_type == OpFusionype.bias_quant:
                temp_flag = bias_flag
                bias_flag += 1
            elif op_to_fusion_type == OpFusionype.dma_common:
                temp_flag = dma_common_flag
                dma_common_flag += 1
            elif op_to_fusion_type == OpFusionype.op_empty:
                temp_flag = 1
            else:
                temp_flag = 0

            if temp_flag == 0:
                fusion_type_list = fusion_type_list + "_" + \
                                   str(op_to_fusion_type.value)
        self.fusion_type = fusion_type_dict.get(
            fusion_type_list, fusion_type_dict["fusion_type_unknow"])
        if self.fusion_type == 0:
            fusion_type_dict_enhance = __get_fusion_type_dict_enhance()
            fusion_type_list = fusion_type_list[len('fusion_type'):]
            fusion_type_list = fusion_type_list.split('_')
            fusion_type_list.sort()
            fusion_type_list = "_".join(fusion_type_list)
            self.fusion_type = fusion_type_dict_enhance.get(
                fusion_type_list, fusion_type_dict_enhance["_unknow"])
         # aipp input format: RGB(<512), YUV[768, 1024)
        if ConvParam.TENSOR_MAP.get("aipp_input_format") == "YUV420SP_U8":
            self.fusion_type += 768


class DataFlowType(Enum):
    """
    describe four data flow in v200 version
    S32toS8: conv + requant
    S32toS16: conv + dequantS16
    S16EltWiseS8: conv + dequantS16 + addrelu + requantS16
    S16EltWiseS8S16: conv + dequantS16 + addrelu + requantS16, double output
    S32toFp16 : conv + dequant
    """
    S32toS8 = 0
    S32toS16 = 1
    S16EltWiseS8 = 2
    S16EltWiseS8S16 = 3
    S32toFp16 = 4
    V200_General_Fusion = 7


class DataFlowTypeLhisi(Enum):
    """
    describe four data flow in v100 version
    S32toFp16: conv + dequant fuse
    S32toS8: conv + dequant + quant fuse
    """
    S32toFp16 = 0
    S32toS8 = 1
    S32toFp16S8 = 2


class OpFusionype(Enum):
    """
    OpFusionype
    """
    default = 0
    conv = 1
    bias = 2
    elewise_one_op = 3
    elewise_two_op = 4
    dma_common = 5
    # quant
    conv_dequant = 6
    conv_quant = 7
    double_out = 8
    bias_quant = 9
    conv_requant = 10
    bn_op = 11
    op_empty = 100
