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
gemm schedule
"""
from enum import Enum
from functools import reduce  # pylint: disable=C0302

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.buildcfg import build_config
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import in_dynamic
from tbe.dsl.boost_schedule_kit import Compare
from tbe.dsl.boost_schedule_kit import ScheduleAgent


class GEMM_Params:
    """
    all params and print function
    """
    def __init__(self):
        self.ops_mode = "fp16fp16"
        self.DEBUG_PARAM = False
        self.DEBUG_IR = False
        self.CUB_FUSED_NUM = {"fp16fp16": 4, "fp16fp32": 1, "int8int32": 2, "int8fp32": 1}
        self.AUB_FUSED_NUM = {"fp16fp16": 0, "fp16fp32": 0, "int8int32": 0, "int8fp32": 40}
        self.BUB_FUSED_NUM = {"fp16fp16": 0, "fp16fp32": 0, "int8int32": 0, "int8fp32": 40}
        self.INPUT_SIZE = {"fp16fp16": 2, "fp16fp32": 2, "int8int32": 1, "int8fp32": 1}
        self.L1_L0_SIZE = {"fp16fp16": 2, "fp16fp32": 2, "int8int32": 1, "int8fp32": 2}
        self.OUTPUT_SIZE = {"fp16fp16": 2, "fp16fp32": 4, "int8int32": 4, "int8fp32": 4}
        self.MAD_TYPE = {
            "fp16fp16": "float32",
            "fp16fp32": "float32",
            "int8int32": "int32",
            "int8fp32": "float32"
        }
        self.ops_mode = "fp16fp16"
        self.ops_format_mode = "ND"
        self.block_in = tbe_platform.BLOCK_IN
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        self.block_out = tbe_platform.BLOCK_OUT
        self.CONST_AL1_SHAPE_DIM = 4
        self.CONST_BL1_SHAPE_DIM = 4
        self.UB_SPACE_SIZE = tbe_platform_info.get_soc_spec("UB_SIZE")
        self.L1_SPACE_SIZE = tbe_platform_info.get_soc_spec("L1_SIZE")
        self.L0_SPACE_SIZE = tbe_platform_info.get_soc_spec("L0A_SIZE")
        self.L0C_SPACE_SIZE = tbe_platform_info.get_soc_spec("L0C_SIZE")
        self.SOC_VERSION = "Ascend310"
        self.TENSOR_MAP = {}
        self.TILING = {}
        self.DIM_MAP = {}
        self.DATA_SIZE = {"int8": 1, "int32": 4, "float16": 2, "float32": 4}
        self.cube_vector_split = False
        self.trans_a = False
        self.trans_b = False
        self.cv_split_nd_in_flag = False
        self.is_dynamic = False
        self.MAT_MUL = False
        self.fusion_type = FusionType.DEFAULT_MODE


    def print_debug(self, *info):
        """
        print log if debug
        :param info:
        :return:
        """
        if self.DEBUG_PARAM:
            print(info)


    def print_ir_matmul(self, process, sch):
        """
        print ir for input sch
        :param process: tag
        :param sch: schedule
        :return: IR process
        """
        if self.DEBUG_IR:
            with build_config():
                start = process + " IR start"
                end = process + " IR end\n"
                sch = sch.normalize()
                print(start)
                bounds = tvm.schedule.InferBound(sch)
                stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
                print(stmt)
                print(end)


class FusionType(Enum):
    """
    fusion type
    """
    DEFAULT_MODE = 0
    ELEWISE_FUSION = 1
    REDUCE_FUSION = 2


class GEMM_Schedule:
    """
    class of gemm schedule
    """
    def __init__(self):
        self.gemm_params = GEMM_Params()


    def _is_int82fp32_nd(self):
        return self.gemm_params.TENSOR_MAP.get("tensor_a_float16_normalize_ub") is not None


    def _get_value(self, shape_object):
        """
        get the value of shape object when having attr value
        """
        return shape_object.value if hasattr(shape_object, "value") else shape_object


    def _get_ops_mode(self):
        """
        Get ops mode from input and output
        :return:
        """
        a_type = self.gemm_params.TENSOR_MAP["a_placehold"].dtype
        a_format = self.gemm_params.TENSOR_MAP["a_placehold"].shape
        c_type = self.gemm_params.TENSOR_MAP["c_gm"].dtype
        if len(a_format) == 2:
            self.gemm_params.ops_format_mode = "ND"
        else:
            self.gemm_params.ops_format_mode = "Nz"
        if a_type == "float16" and c_type == "float16":
            self.gemm_params.ops_mode = "fp16fp16"
            self.gemm_params.block_reduce = tbe_platform.BLOCK_REDUCE
        elif a_type == "float16" and c_type == "float32":
            self.gemm_params.ops_mode = "fp16fp32"
            self.gemm_params.block_reduce = tbe_platform.BLOCK_REDUCE
        elif a_type == "int8" and c_type == "int32":
            self.gemm_params.ops_mode = "int8int32"
            self.gemm_params.block_reduce = tbe_platform.BLOCK_REDUCE_INT8
        elif a_type == "int8" and c_type == "float32":
            if self._is_int82fp32_nd():
                self.gemm_params.ops_mode = "fp16fp32"
            else:
                self.gemm_params.ops_mode = "int8fp32"
            self.gemm_params.block_reduce = tbe_platform.BLOCK_REDUCE
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "Unsupported data type",
                "value": "a_type = {}, c_type = {}".format(a_type, c_type)
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))


    def _int_ceil_div(self, divisor_a, divisor_b):
        """
        round up function
        :param divisor_a: int.
        :param divisor_b: int.
        :return: int
        """
        if divisor_b == 0:
            args_dict = {
                "errCode": "E60114",
                "reason": "division by zero",
                "value": "divisor_b = {}".format(divisor_b)
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        return (divisor_a + divisor_b - 1) // divisor_b


    def _get_all_tensors(self, res):
        """
        get all tensor
        :param res: tensor
        :return: list
        """

        all_tensor = dict()
        leaf_tensor = dict()
        all_tensor["res"] = res

        def get(tensor):
            """
            find all tensor
            :param tensor: c_gm
            :return: all tensor
            """
            tensor_list = tensor.op.input_tensors
            for one_tensor in tensor_list:
                # check which tensor has not been checked
                if not one_tensor.op.input_tensors:
                    leaf_tensor[one_tensor.op.name] = tensor
                if one_tensor.op.name not in all_tensor:
                    all_tensor[one_tensor.op.name] = one_tensor
                    get(one_tensor)

        get(res)
        return all_tensor, leaf_tensor


    def _check_align(self, b_n, stridew, divide_factor):
        if 1 <= b_n <= divide_factor or b_n % divide_factor == 0:
            stridew = 1
        elif b_n % divide_factor < 16:
            stridew = 0
        return stridew


    def _check_align_int8(self, b_n, stridew, divide_factor):
        if 1 <= b_n <= divide_factor or b_n % divide_factor == 0:
            stridew = 1
        elif b_n % divide_factor < 8:
            stridew = 0
        return stridew


    def _get_transpose(self):
        transpose_a = (
            "transpose_a" in self.gemm_params.TENSOR_MAP["a_l0a"].op.attrs
            and self.gemm_params.TENSOR_MAP["a_l0a"].op.attrs["transpose_a"] == "true"
        )

        transpose_b = (
            "transpose_b" in self.gemm_params.TENSOR_MAP["b_l0b"].op.attrs
            and self.gemm_params.TENSOR_MAP["b_l0b"].op.attrs["transpose_b"] == "true"
        )
        return transpose_a, transpose_b


    def _get_index(self, transpose_a, transpose_b):
        if transpose_a:
            a_m_index = 1
            a_k_index = 0
        else:
            a_m_index = 0
            a_k_index = 1

        if transpose_b:
            b_k_index = 1
            b_n_index = 0
        else:
            b_k_index = 0
            b_n_index = 1
        return a_m_index, a_k_index, b_k_index, b_n_index


    def _get_tiling_params(self, transpose_a, transpose_b):
        if self.gemm_params.ops_mode == "int8int32":
            if transpose_a:
                pad_l = 20
            else:
                pad_l = 10
            if transpose_b:
                pad_r = 10
            else:
                pad_r = 20
        elif self._is_int82fp32_nd():
            pad_l = 40
            pad_r = 40
        else:
            pad_l = 10
            pad_r = 10

        if self.gemm_params.ops_mode == "fp16fp16":
            fused_num = 5
        elif self.gemm_params.ops_mode == "int8int32":
            fused_num = 1
        else:
            fused_num = 2

        if self.gemm_params.TENSOR_MAP.get("b_transpose_only") is not None:
            pad_r = 60
            if self.gemm_params.ops_mode == "int8int32":
                pad_r = 30
        return pad_l, pad_r, fused_num


    def _get_trans_flag(self, transpose_a, transpose_b):
        trans_flag = 1
        if transpose_a:
            if transpose_b:
                trans_flag = 4
            else:
                trans_flag = 2
        elif transpose_b:
            trans_flag = 3

        return trans_flag


    def _get_tiling_result_nd(self, kernel_name):  # pylint: disable=R0914
        """
        :param None:
        :return: TILING result and data_byte
        """
        a_type = self.gemm_params.TENSOR_MAP["a_placehold"].dtype
        b_type = self.gemm_params.TENSOR_MAP["b_placehold"].dtype
        c_type = self.gemm_params.TENSOR_MAP["c_gm"].dtype

        transpose_a, transpose_b = self._get_transpose()
        a_m_index, a_k_index, b_k_index, b_n_index = self._get_index(transpose_a, transpose_b)

        b_n = self.gemm_params.DIM_MAP["b_shape"][b_n_index]

        if a_type == "float16":
            a_shape = [
                1,
                (self.gemm_params.DIM_MAP["a_shape"][a_k_index] + 16 - 1) // 16,
                (self.gemm_params.DIM_MAP["a_shape"][a_m_index] + 16 - 1) // 16,
                16,
                16
            ]
            b_shape = [
                self.gemm_params.DIM_MAP["b_shape"][b_k_index],
                (self.gemm_params.DIM_MAP["b_shape"][b_n_index] + 16 - 1) // 16,
                1,
                1,
                16
            ]
        elif self._is_int82fp32_nd():
            a_shape = [
                1,
                (self.gemm_params.DIM_MAP["a_shape"][a_k_index] + 32 - 1) // 32,
                (((self.gemm_params.DIM_MAP["a_shape"][a_m_index] + 32 - 1) // 32) * 32 // 16),
                16,
                32
            ]
            b_shape = [
                ((self.gemm_params.DIM_MAP["b_shape"][b_k_index] + 32 - 1) // 32) * 32,
                (self.gemm_params.DIM_MAP["b_shape"][b_n_index] + 32 - 1) // 32 * 32 // 16,
                1,
                1,
                16
            ]
        else:
            a_shape = [
                1,
                (self.gemm_params.DIM_MAP["a_shape"][a_k_index] + 32 - 1) // 32,
                (((self.gemm_params.DIM_MAP["a_shape"][a_m_index] + 32 - 1) // 32) * 32 // 16),
                16,
                32
            ]
            b_shape = [
                ((self.gemm_params.DIM_MAP["b_shape"][b_k_index] + 32 - 1) // 32) * 32,
                (self.gemm_params.DIM_MAP["b_shape"][b_n_index] + 32 - 1) // 32,
                1,
                1,
                32
            ]

        pad_l, pad_r, fused_num = self._get_tiling_params(transpose_a, transpose_b)
        mad_type = self.gemm_params.MAD_TYPE.get(self.gemm_params.ops_mode)
        trans_flag = self._get_trans_flag(transpose_a, transpose_b)

        stridew = 1
        if self.gemm_params.ops_mode == "int8int32" or self._is_int82fp32_nd():
            divide_factor = 32
            stridew = self._check_align_int8(b_n, stridew, divide_factor)
        else:
            divide_factor = 16
            stridew = self._check_align(b_n, stridew, divide_factor)

        info_dict = {
            "op_type": "matmul",
            "A_shape": a_shape,
            "B_shape": b_shape,
            "C_shape": None,
            "A_dtype": a_type,
            "B_dtype": b_type,
            "C_dtype": c_type,
            "mad_dtype": mad_type,
            "padl": pad_l,
            "padr": pad_r,
            "padu": 0,
            "padd": 0,
            "strideH": 0,
            "strideW": stridew,
            "strideH_expand": 1,
            "strideW_expand": 1,
            "dilationH": trans_flag,
            "dilationW": 1,
            "group": 1,
            "bias_flag": 0,
            "fused_double_operand_num": fused_num,
            "kernel_name": kernel_name.value
        }
        tiling = get_tiling(info_dict)

        if not tiling:
            args_dict = {"errCode": "E60114", "reason": "tiling is None", "value": "None"}
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

        tiling = self._no_solution_tiling(tiling)
        if self._is_int82fp32_nd():
            tiling["AL0_matrix"][2] = 16
            tiling["AL0_matrix"][3] = 16
            tiling["BL0_matrix"][2] = 16
            tiling["BL0_matrix"][3] = 16

        self.gemm_params.print_debug("-----------auto tiling result-----------------")
        self.gemm_params.print_debug(tiling)
        self.gemm_params.print_debug("----------------------------------------------")
        return tiling


    def _set_data_layout(self, res, sch):  # pylint: disable=too-many-statements
        """
        get DIM_MAP which contains all ops

        Parameter:
        ----------------------------------------------------------
        :param res: op
        :param sch: schedule
        :return: None
        ----------------------------------------------------------
        """

        all_tensor, leaf_tensor = self._get_all_tensors(res)

        def _init_common_tensor():
            if "reduce_sum" in res.op.tag:
                self.gemm_params.TENSOR_MAP["c_gm"] = all_tensor.get("tensor_c_ub")
                sch[all_tensor.get("tensor_c_gm")].compute_inline()
            else:
                self.gemm_params.TENSOR_MAP["c_gm"] = all_tensor.get("res")

            if all_tensor.get("tensor_alpha") is not None and all_tensor.get("tensor_beta") is not None:
                self.gemm_params.TENSOR_MAP["alpha"] = all_tensor.get("tensor_alpha")
                self.gemm_params.TENSOR_MAP["beta"] = all_tensor.get("tensor_beta")
            else:
                self.gemm_params.MAT_MUL = True
            if not self.gemm_params.MAT_MUL:
                self.gemm_params.TENSOR_MAP["tensor_a_float16_normalize_ub"] = all_tensor.get(
                    "tensor_a_float16_normalize_ub"
                )
                self.gemm_params.TENSOR_MAP["tensor_b_float16_normalize_ub"] = all_tensor.get(
                    "tensor_b_float16_normalize_ub"
                )
                if  self._is_int82fp32_nd():
                    sch[self.gemm_params.TENSOR_MAP["tensor_a_float16_normalize_ub"]].set_scope(
                        tbe_platform_info.scope_ubuf
                    )
                    sch[self.gemm_params.TENSOR_MAP["tensor_b_float16_normalize_ub"]].set_scope(
                        tbe_platform_info.scope_ubuf
                    )

            # tensor in aicore
            self.gemm_params.TENSOR_MAP["c_ub"] = all_tensor.get("tensor_c_ub")
            if self.gemm_params.TENSOR_MAP["c_ub"] is not None:
                sch[self.gemm_params.TENSOR_MAP["c_ub"]].set_scope(tbe_platform_info.scope_ubuf)
            self.gemm_params.TENSOR_MAP["a_l0a"] = all_tensor.get("tensor_a_l0a")
            sch[self.gemm_params.TENSOR_MAP["a_l0a"]].set_scope(tbe_platform_info.scope_ca)
            self.gemm_params.TENSOR_MAP["a_l1"] = all_tensor.get("tensor_a_l1")
            sch[self.gemm_params.TENSOR_MAP["a_l1"]].set_scope(tbe_platform_info.scope_cbuf)
            self.gemm_params.TENSOR_MAP["b_l0b"] = all_tensor.get("tensor_b_l0b")
            sch[self.gemm_params.TENSOR_MAP["b_l0b"]].set_scope(tbe_platform_info.scope_cb)
            self.gemm_params.TENSOR_MAP["b_l1"] = all_tensor.get("tensor_b_l1")
            sch[self.gemm_params.TENSOR_MAP["b_l1"]].set_scope(tbe_platform_info.scope_cbuf)
            self.gemm_params.TENSOR_MAP["c_l0c"] = all_tensor.get("tensor_c")
            sch[self.gemm_params.TENSOR_MAP["c_l0c"]].set_scope(tbe_platform_info.scope_cc)
            self.gemm_params.TENSOR_MAP["bias_ub"] = all_tensor.get("tensor_bias_ub")

            if self.gemm_params.fusion_type == FusionType.DEFAULT_MODE:
                self.gemm_params.TENSOR_MAP["a_placehold"] = all_tensor.get("tensor_a")
                self.gemm_params.TENSOR_MAP["b_placehold"] = all_tensor.get("tensor_b")
                self.gemm_params.TENSOR_MAP["bias"] = all_tensor.get("tensor_bias")
            else:
                self.gemm_params.TENSOR_MAP["a_placehold"] = self.gemm_params.TENSOR_MAP["a_l1"].op.input_tensors[0]
                self.gemm_params.TENSOR_MAP["b_placehold"] = self.gemm_params.TENSOR_MAP["b_l1"].op.input_tensors[0]
                if self.gemm_params.TENSOR_MAP["bias_ub"] is not None:
                    self.gemm_params.TENSOR_MAP["bias"] = self.gemm_params.TENSOR_MAP["bias_ub"].op.input_tensors[0]
            if self.gemm_params.MAT_MUL:
                if self.gemm_params.TENSOR_MAP["bias_ub"] is not None:
                    sch[self.gemm_params.TENSOR_MAP["bias_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                    self.gemm_params.TENSOR_MAP["bias_l0c"] = all_tensor.get("tensor_bias_l0c")
                    sch[self.gemm_params.TENSOR_MAP["bias_l0c"]].set_scope(tbe_platform_info.scope_cc)
                    self.gemm_params.TENSOR_MAP["c_add_bias"] = all_tensor.get("tensor_c_add_bias")
                    sch[self.gemm_params.TENSOR_MAP["c_add_bias"]].set_scope(tbe_platform_info.scope_cc)
            else:
                sch[self.gemm_params.TENSOR_MAP["bias_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["beta_bias_ub"] = all_tensor.get("tensor_beta_bias_ub")
                sch[self.gemm_params.TENSOR_MAP["beta_bias_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["beta_ub"] = all_tensor.get("tensor_beta_ub")
                sch[self.gemm_params.TENSOR_MAP["beta_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["alpha_ub"] = all_tensor.get("tensor_alpha_ub")
                sch[self.gemm_params.TENSOR_MAP["alpha_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["alpha_c_ub"] = all_tensor.get("tensor_alpha_c_ub")
                sch[self.gemm_params.TENSOR_MAP["alpha_c_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["c_before_mul_ub"] = all_tensor.get("tensor_c_before_mul_ub")
                sch[self.gemm_params.TENSOR_MAP["c_before_mul_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["c_ub_temp"] = all_tensor.get("tensor_c_ub_temp")
                sch[self.gemm_params.TENSOR_MAP["c_ub_temp"]].set_scope(tbe_platform_info.scope_ubuf)

        def _init_fp16_fp16_tensor():
            if self.gemm_params.ops_mode == "fp16fp16" and not self.gemm_params.MAT_MUL:
                self.gemm_params.TENSOR_MAP["float32_bias_ub"] = all_tensor.get(
                    "tensor_float32_bias_ub"
                )
                sch[self.gemm_params.TENSOR_MAP["float32_bias_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["beta_temp_ub"] = all_tensor.get("tensor_beta_temp_ub")
                sch[self.gemm_params.TENSOR_MAP["beta_temp_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["alpha_temp_ub"] = all_tensor.get("tensor_alpha_temp_ub")
                sch[self.gemm_params.TENSOR_MAP["alpha_temp_ub"]].set_scope(tbe_platform_info.scope_ubuf)

            if self.gemm_params.ops_format_mode == "ND":
                self.gemm_params.TENSOR_MAP["a_normalize_ub"] = all_tensor.get(
                    "tensor_a_normalize_ub"
                )
                sch[self.gemm_params.TENSOR_MAP["a_normalize_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["a_fract_k_ub"] = all_tensor.get("a_fract_k")
                sch[self.gemm_params.TENSOR_MAP["a_fract_k_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["b_normalize_ub"] = all_tensor.get(
                    "tensor_b_normalize_ub"
                )
                sch[self.gemm_params.TENSOR_MAP["b_normalize_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["b_fract_ub"] = all_tensor.get("b_fract")
                sch[self.gemm_params.TENSOR_MAP["b_fract_ub"]].set_scope(tbe_platform_info.scope_ubuf)

                self.gemm_params.TENSOR_MAP["b_transpose_only"] = all_tensor.get("b_transpose_only")
                self.gemm_params.TENSOR_MAP["b_transpose_zero"] = all_tensor.get("b_transpose_zero")
                self.gemm_params.TENSOR_MAP["b_after_process"] = all_tensor.get("b_after_process")
                if self.gemm_params.TENSOR_MAP["b_transpose_only"] is not None:
                    sch[self.gemm_params.TENSOR_MAP["b_transpose_only"]].set_scope(
                        tbe_platform_info.scope_ubuf
                    )
                    sch[self.gemm_params.TENSOR_MAP["b_transpose_zero"]].set_scope(
                        tbe_platform_info.scope_ubuf
                    )
                    sch[self.gemm_params.TENSOR_MAP["b_after_process"]].set_scope(
                        tbe_platform_info.scope_ubuf
                    )

                if self.gemm_params.ops_mode == "int8int32":
                    self.gemm_params.TENSOR_MAP["b_transpose"] = all_tensor.get("b_transpose")
                    if self.gemm_params.TENSOR_MAP["b_transpose"] is not None:
                        sch[self.gemm_params.TENSOR_MAP["b_transpose"]].set_scope(
                            tbe_platform_info.scope_ubuf
                        )
                    self.gemm_params.TENSOR_MAP["a_transpose"] = all_tensor.get("a_transpose")
                    if self.gemm_params.TENSOR_MAP["a_transpose"] is not None:
                        sch[self.gemm_params.TENSOR_MAP["a_transpose"]].set_scope(
                            tbe_platform_info.scope_ubuf
                        )

        def _init_int8_fp32_tensor():
            if self.gemm_params.ops_mode == "int8fp32" and not self.gemm_params.MAT_MUL:
                self.gemm_params.TENSOR_MAP["a_ub"] = all_tensor.get("tensor_a_ub")
                sch[self.gemm_params.TENSOR_MAP["a_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["float16_a_ub"] = all_tensor.get("tensor_float16_a_ub")
                sch[self.gemm_params.TENSOR_MAP["float16_a_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["zz_a_ub"] = all_tensor.get("tensor_zz_a_ub")
                sch[self.gemm_params.TENSOR_MAP["zz_a_ub"]].set_scope(tbe_platform_info.scope_ubuf)

                self.gemm_params.TENSOR_MAP["b_ub"] = all_tensor.get("tensor_b_ub")
                sch[self.gemm_params.TENSOR_MAP["b_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["float16_b_ub"] = all_tensor.get("tensor_float16_b_ub")
                sch[self.gemm_params.TENSOR_MAP["float16_b_ub"]].set_scope(tbe_platform_info.scope_ubuf)
                self.gemm_params.TENSOR_MAP["zn_b_ub"] = all_tensor.get("tensor_zn_b_ub")
                sch[self.gemm_params.TENSOR_MAP["zn_b_ub"]].set_scope(tbe_platform_info.scope_ubuf)

        def _init_fract_tensor():
            if "tensor_bias_ub_fract" in all_tensor and not self.gemm_params.MAT_MUL:
                self.gemm_params.TENSOR_MAP["bias_ub_fract"] = all_tensor.get("tensor_bias_ub_fract")
                sch[self.gemm_params.TENSOR_MAP["bias_ub_fract"]].set_scope(tbe_platform_info.scope_ubuf)

        def _init_map():
            # fill in dimmap
            self.gemm_params.DIM_MAP["out_shape"] = [self._get_value(i) for i in res.shape]
            self.gemm_params.DIM_MAP["a_shape"] = [
                self._get_value(i) for i in self.gemm_params.TENSOR_MAP["a_placehold"].shape
            ]
            self.gemm_params.DIM_MAP["A_matrix_dim"] = [
                self._get_value(i) for i in self.gemm_params.TENSOR_MAP["a_l0a"].shape
            ]
            self.gemm_params.DIM_MAP["B_matrix_dim"] = [
                self._get_value(i) for i in self.gemm_params.TENSOR_MAP["b_l0b"].shape
            ]
            self.gemm_params.DIM_MAP["b_shape"] = [
                self._get_value(i) for i in self.gemm_params.TENSOR_MAP["b_placehold"].shape
            ]

        def _init_fusion_case():
            self.gemm_params.fusion_type = FusionType.DEFAULT_MODE
            if "elewise" in res.op.tag:
                self.gemm_params.fusion_type = FusionType.ELEWISE_FUSION
                ub_list = list()
                input_list = list()
                output_ub = sch.cache_write(res, tbe_platform_info.scope_ubuf)
                ub_list.append(output_ub)
                for key, value in all_tensor.items():
                    if (not value.op.input_tensors and leaf_tensor[key].op.name not in
                            ("tensor_a_l1", "tensor_b_l1", "tensor_bias_ub")):
                        if res != leaf_tensor[key]:
                            input_ub = sch.cache_read(value, tbe_platform_info.scope_ubuf, leaf_tensor[key])
                        else:
                            input_ub = sch.cache_read(value, tbe_platform_info.scope_ubuf, output_ub)
                        input_list.append(input_ub)
                    elif "elewise" in value.op.tag and res != value:
                        sch[value].set_scope(tbe_platform_info.scope_ubuf)
                        ub_list.append(value)
                    elif "broadcast" in value.op.tag or key == "tensor_c_gm":
                        sch[value].compute_inline()
                self.gemm_params.TENSOR_MAP["fusion_ub"] = ub_list
                self.gemm_params.TENSOR_MAP["fusion_input"] = input_list
            elif "reduce_sum" in res.op.tag:
                self.gemm_params.fusion_type = FusionType.REDUCE_FUSION
    

        _init_fusion_case()
        _init_common_tensor()
        self._get_ops_mode()
        _init_fp16_fp16_tensor()
        _init_int8_fp32_tensor()
        _init_fract_tensor()
        _init_map()


    def _get_tiling(self, kernel_name):  # pylint: disable=too-many-statements
        """
        get tilling parameter from tilling guery and check all parameter

        Parameter:
        ---------------------------------------------------------
        :return: None
        ---------------------------------------------------------
        """

        # check tilling parts
        # get data amount in AL1 BL1
        def _get_data_amount_l1(l1_shape, isdouble, data_size):
            """
            using tilling parameter calculate data amount in l1

            Parameters:
            ---------------------------------------------------
            :param l1_shape:  'AL1_shape' or 'BL1_shape'
            :param isdouble:  True or False
            :return:  data amount in l1_shape
            ---------------------------------------------------
            """
            data_amount_l1 = 0
            if self.gemm_params.TILING.get(l1_shape) is None:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "l1_shape can not be None",
                    "value": "None"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if self.gemm_params.TILING.get(l1_shape) == []:
                if l1_shape == "AL1_shape":
                    data_amount_l1 = (
                        reduce(lambda x, y: x * y, self.gemm_params.DIM_MAP["A_matrix_dim"][-4:])
                        // self.gemm_params.TILING["block_dim"][2]
                    )
                if l1_shape == "BL1_shape":
                    data_amount_l1 = (
                        reduce(lambda x, y: x * y, self.gemm_params.DIM_MAP["B_matrix_dim"][-4:])
                        // self.gemm_params.TILING["block_dim"][1]
                    )
            else:
                l1_k = self.gemm_params.TILING.get(l1_shape)[0]
                l1_mn = self.gemm_params.TILING.get(l1_shape)[1]
                if l1_k == 0 or l1_mn == 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "l1_k or l1_mn can not be zero",
                        "value": "l1_k = {}, l1_mn = {}".format(l1_k, l1_mn)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if l1_k % self.gemm_params.block_reduce != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "l1_k can not be divided by tbe_platform.BLOCK_REDUCE",
                        "value": "l1_k = {}, tbe_platform.BLOCK_REDUCE "
                        "= {}".format(l1_k, self.gemm_params.block_reduce)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if l1_shape == "AL1_shape":
                    data_amount_l1 = (
                        l1_k
                        * l1_mn
                        * self.gemm_params.TILING.get("CL0_matrix")[1]
                        * tbe_platform.BLOCK_IN
                        * data_size
                    )

                else:
                    data_amount_l1 = (
                        l1_k
                        * l1_mn
                        * self.gemm_params.TILING.get("CL0_matrix")[0]
                        * tbe_platform.BLOCK_OUT
                        * data_size
                    )
                if isdouble == 2:
                    data_amount_l1 = data_amount_l1 * 2
            return data_amount_l1

        # check tilling l0
        def _check_tilling_l0(l0_shape, l0_space, isdouble, data_size):
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
            row = self.gemm_params.TILING.get(l0_shape)[0]
            col = self.gemm_params.TILING.get(l0_shape)[1]
            if row == 0 or col == 0:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "k, m, n in L0A/B can not be zero",
                    "value": "row = {}, col = {}".format(row, col)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            data_amount_l0 = (
                row
                * col
                * self.gemm_params.TILING.get(l0_shape)[2]
                * self.gemm_params.TILING.get(l0_shape)[3]
                * data_size
                * isdouble
            )
            if self.gemm_params.ops_mode == "int8fp32":
                data_amount_l0 = data_amount_l0 // 2
            self.gemm_params.print_debug(
                "{} data mount(KB)".format(l0_shape), data_amount_l0 / 1024
            )
            if data_amount_l0 > l0_space:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "tilling size exceed L0A/B Buffer",
                    "value": "tiling size = {}, "
                    "l0_space = {}".format(data_amount_l0, l0_space)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

        # check tilling l0c
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
            cl0_n = self.gemm_params.TILING.get(l0c_shape)[0]
            cl0_m = self.gemm_params.TILING.get(l0c_shape)[1]
            if self.gemm_params.TILING.get("BL0_matrix") != []:
                if cl0_m == 0 or cl0_n == 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "cl0_m, cl0_n can not be zero",
                        "value": "cl0_m = {}, lc0_n = {}".format(cl0_m, cl0_n)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                data_amount_cl0 = (
                    cl0_m
                    * cl0_n
                    * self.gemm_params.TILING.get(l0c_shape)[2]
                    * self.gemm_params.TILING.get(l0c_shape)[3]
                    * 4
                    * isdouble
                )
                self.gemm_params.print_debug("data_amount_l0c(KB)", data_amount_cl0 / 1024)
                if data_amount_cl0 > l0c_space:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "tilling size exceed L0C Buffer",
                        "value": "tiling size = {}, "
                        "l0c_space = {}".format(data_amount_cl0, l0c_space)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

        def _check_tiling_ub():
            """
            check tilling parameter in ub

            Parameter:
            ------------------------------------------------------
            :return: None
            -------------------------------------------------------
            """

            # check tilling cub
            def _check_tilling_cub():
                """
                check tilling parameter in cub

                Parameter:
                ------------------------------------------------------
                :return: None
                -------------------------------------------------------
                """
                nc_factor = self.gemm_params.TILING.get("CUB_matrix")[0]
                mc_factor = self.gemm_params.TILING.get("CUB_matrix")[1]
                if self.gemm_params.TILING.get("CL0_matrix")[0] % nc_factor != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "nc_factor is not factor of nc",
                        "value": "nc_factor = {}".format(nc_factor)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                manual_pingpong_buffer = self.gemm_params.TILING.get("manual_pingpong_buffer")
                is_double = manual_pingpong_buffer.get("CUB_pbuffer")
                data_amount_cub = (
                    nc_factor
                    * mc_factor
                    * self.gemm_params.TILING.get("CUB_matrix")[2]
                    * self.gemm_params.TILING.get("CUB_matrix")[3]
                    * self.gemm_params.OUTPUT_SIZE.get(self.gemm_params.ops_mode)
                    * is_double
                    * (fusion_sums + 1)
                )
                return data_amount_cub

            def _check_aub_bub_tiling(shape_head):
                """
                check tilling parameter in ub

                Parameter: shape_head: 'A' or 'B'
                ------------------------------------------------------
                :return: None
                -------------------------------------------------------
                """
                # check ub between l1
                ub_name = shape_head + "UB_shape"
                if self.gemm_params.TILING.get(ub_name) is None:
                    return 0
                data_size = 2
                if self.gemm_params.ops_mode in ("int8fp32", "int8int32"):
                    data_size = 1
                manual_pingpong_buffer = self.gemm_params.TILING.get("manual_pingpong_buffer")
                is_double = manual_pingpong_buffer.get(shape_head + "UB_pbuffer")
                data_amount_ub = (
                    self.gemm_params.TILING.get(ub_name)[0]
                    * self.gemm_params.TILING.get(ub_name)[1]
                    * 16
                    * is_double
                    * data_size
                )
                return data_amount_ub

            data_amount_aub = _check_aub_bub_tiling("A") * (
                self.gemm_params.AUB_FUSED_NUM.get(self.gemm_params.ops_mode) // 10 + 1
            )
            data_amount_bub = _check_aub_bub_tiling("B") * (
                self.gemm_params.BUB_FUSED_NUM.get(self.gemm_params.ops_mode) // 10 + 1
            )
            data_amount_cub = _check_tilling_cub()
            self.gemm_params.print_debug("data_amount_aub(KB)", data_amount_aub / 1024)
            self.gemm_params.print_debug("data_amount_bub(KB)", data_amount_bub / 1024)
            self.gemm_params.print_debug("data_amount_cub(KB)", data_amount_cub / 1024)
            if self.gemm_params.ops_mode == "fp16fp16":
                alpha_beta_size = 2 * 2
            else:
                alpha_beta_size = 4 * 2

            total_size_ub = (
                data_amount_aub + data_amount_bub + data_amount_cub + alpha_beta_size
            )
            self.gemm_params.print_debug("total_data_amount_ub(KB)", total_size_ub / 1024)
            if total_size_ub > self.gemm_params.UB_SPACE_SIZE:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "tilling size exceed UB Buffer",
                    "value": "tiling size = {}, UB_space = "
                    "{}".format(total_size_ub, self.gemm_params.UB_SPACE_SIZE)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

        def _get_tiling_l0a_l0b(cl0_matrix, l0_matrix, instr):
            """
            :param cl0_matrix:
            :param l0_matrix:
            :param instr:
            :return:
            """
            if self.gemm_params.ops_mode in ("fp16fp16", "fp16fp32"):
                dtype = "float16"
            else:
                dtype = "int8"

            k_dim = self.gemm_params.DIM_MAP.get("A_matrix_dim")[-3]
            if instr == "A":
                if l0_matrix != []:
                    full_ab = [
                        cl0_matrix[1],
                        l0_matrix[0],
                        tbe_platform.CUBE_MKN[dtype]["mac"][0],
                        tbe_platform.CUBE_MKN[dtype]["mac"][1],
                        1
                    ]
                else:
                    full_ab = [
                        cl0_matrix[1],
                        k_dim,
                        tbe_platform.CUBE_MKN[dtype]["mac"][0],
                        tbe_platform.CUBE_MKN[dtype]["mac"][1],
                        1
                    ]
            elif instr == "B":
                if l0_matrix != []:
                    full_ab = [
                        l0_matrix[1],
                        cl0_matrix[0],
                        tbe_platform.CUBE_MKN[dtype]["mac"][2],
                        tbe_platform.CUBE_MKN[dtype]["mac"][1],
                        1
                    ]
                else:
                    full_ab = [
                        k_dim,
                        cl0_matrix[0],
                        tbe_platform.CUBE_MKN[dtype]["mac"][2],
                        tbe_platform.CUBE_MKN[dtype]["mac"][1],
                        1
                    ]
            else:
                args_dict = {
                    "errCode": "E60000",
                    "param_name": "instr",
                    "expected_value": "[A,B]",
                    "input_value": instr
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            return full_ab

        def get_tiling_result(kernel_name):
            """
            :param None:
            :return: TILING result and data_byte
            """
            a_type = self.gemm_params.TENSOR_MAP["a_placehold"].dtype
            b_type = self.gemm_params.TENSOR_MAP["b_placehold"].dtype
            c_type = self.gemm_params.TENSOR_MAP["c_gm"].dtype
            a_shape, b_shape, pad_l, pad_r, fused_num = self.get_tiling_param_nz()
            mad_type = self.gemm_params.MAD_TYPE.get(self.gemm_params.ops_mode)
            bias_flag = self.gemm_params.MAT_MUL and self.gemm_params.TENSOR_MAP.get("bias_ub") is not None
            transpose_op_a, transpose_op_b = self._get_transpose()
            trans_flag = self._get_trans_flag(not transpose_op_a, not transpose_op_b) if self.gemm_params.MAT_MUL else 1
            info_dict = {
                "op_type": "matmul",
                "A_shape": a_shape,
                "B_shape": b_shape,
                "C_shape": None,
                "A_dtype": a_type,
                "B_dtype": b_type,
                "C_dtype": c_type,
                "mad_dtype": mad_type,
                "padl": pad_l,
                "padr": pad_r,
                "padu": 0,
                "padd": 0,
                "strideH": 1,
                "strideW": 1,
                "strideH_expand": 1,
                "strideW_expand": 1,
                "dilationH": trans_flag,
                "dilationW": 1,
                "group": 1,
                "bias_flag": bias_flag,
                "fused_double_operand_num": fused_num,
                "kernel_name": self._get_value(kernel_name)
            }
            tiling = get_tiling(info_dict)

            if not tiling:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "tiling is None",
                    "value": "None"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

            self.gemm_params.print_debug("-----------auto tiling result-----------------")
            self.gemm_params.print_debug(tiling)
            self.gemm_params.print_debug("----------------------------------------------")
            return tiling, fused_num

        def _check_tiling_al1():
            manual_pingpong_buffer = self.gemm_params.TILING.get("manual_pingpong_buffer")
            data_amount_al1 = 0
            if self.gemm_params.TILING.get("AL1_shape") is None:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "AL1_shape can not be None",
                    "value": "AL1_shape is None"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if (
                self.gemm_params.TILING.get("AL1_shape") != []
                and len(self.gemm_params.TILING.get("AL1_shape")) != self.gemm_params.CONST_AL1_SHAPE_DIM
            ):
                args_dict = {
                    "errCode": "E60114",
                    "reason": "AL1_shape should be Four",
                    "value": "AL1_shape = " "{}".format(self.gemm_params.TILING.get("AL1_shape"))
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            data_amount_al1 = _get_data_amount_l1(
                "AL1_shape",
                manual_pingpong_buffer.get("AL1_pbuffer"),
                self.gemm_params.INPUT_SIZE.get(self.gemm_params.ops_mode)
            )
            return data_amount_al1

        def _check_tiling_bl1():
            manual_pingpong_buffer = self.gemm_params.TILING.get("manual_pingpong_buffer")
            data_amount_bl1 = 0
            if self.gemm_params.TILING.get("BL1_shape") is None:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "BL1 can not be None",
                    "value": "BL1 is None"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if (
                self.gemm_params.TILING.get("BL1_shape") != []
                and len(self.gemm_params.TILING.get("BL1_shape")) != self.gemm_params.CONST_BL1_SHAPE_DIM
            ):
                args_dict = {
                    "errCode": "E60114",
                    "reason": "BL1_shape should be Four",
                    "value": "BL1_shape =" " {}".format(self.gemm_params.TILING.get("BL1_shape"))
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            data_amount_bl1 = _get_data_amount_l1(
                "BL1_shape",
                manual_pingpong_buffer.get("BL1_pbuffer"),
                self.gemm_params.INPUT_SIZE.get(self.gemm_params.ops_mode)
            )
            return data_amount_bl1

        def _check_mul_al1_bl1():
            if self.gemm_params.TILING.get("BL1_shape") and self.gemm_params.TILING.get("AL1_shape"):
                k_al1 = self.gemm_params.TILING.get("AL1_shape")[0]
                k_bl1 = self.gemm_params.TILING.get("BL1_shape")[0]
                if k_al1 % k_bl1 != 0 and k_bl1 % k_al1 != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "kal1 should be divisible by kbl1"
                        " or kbl1 should be divisible by kal1",
                        "value": "kal1 = {kal1}, "
                        "kbl1={kbl1}".format(kal1=k_al1, kbl1=k_bl1)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if k_al1 % (self.gemm_params.TILING.get("AL0_matrix")[1] * self.gemm_params.block_reduce) != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "ka should be divisible by kal1",
                        "value": "ka = {ka}, kal1= {kal1}".format(
                            ka=self.gemm_params.TILING.get("AL0_matrix")[1] * self.gemm_params.block_reduce,
                            kal1=k_al1
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if (
                    self.gemm_params.TILING.get("BL0_matrix")
                    and k_bl1 % (self.gemm_params.TILING.get("BL0_matrix")[0] * self.gemm_params.block_reduce)
                    != 0
                ):
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "kb should be divisible by kbl1",
                        "value": "kb = {kb}, kbl1= {kbl1}".format(
                            kb=(self.gemm_params.TILING.get("BL0_matrix")[0] * self.gemm_params.block_reduce),
                            kbl1=k_bl1
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

        def _check_tiling_l0a_l0b(data_amount_l1b):
            data_size = self.gemm_params.L1_L0_SIZE.get(self.gemm_params.ops_mode)
            manual_pingpong_buffer = self.gemm_params.TILING.get("manual_pingpong_buffer")
            if self.gemm_params.TILING.get("AL0_matrix") == []:
                self.gemm_params.TILING["AL0_matrix"] = _get_tiling_l0a_l0b(
                    self.gemm_params.TILING.get("CL0_matrix"), self.gemm_params.TILING.get("BL0_matrix"), "A"
                )

            if self.gemm_params.TILING.get("BL0_matrix") == []:
                self.gemm_params.TILING["BL0_matrix"] = _get_tiling_l0a_l0b(
                    self.gemm_params.TILING.get("CL0_matrix"), self.gemm_params.TILING.get("AL0_matrix"), "B"
                )

            # check tilling in AL0 BL0
            if (
                self.gemm_params.TILING.get("AL0_matrix") is None
                or self.gemm_params.TILING.get("AL0_matrix") == []
            ):
                args_dict = {
                    "errCode": "E60114",
                    "reason": "tiling[AL0_matrix] can not be None or []",
                    "value": "tiling[AL0_matrix] = {}".format(
                        self.gemm_params.TILING.get("AL0_matrix")
                    )
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            _check_tilling_l0(
                "AL0_matrix",
                self.gemm_params.L0_SPACE_SIZE,
                manual_pingpong_buffer.get("AL0_pbuffer"),
                data_size
            )
            if self.gemm_params.TILING.get("BL0_matrix") is None:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "tiling[BL0_matrix] can not be None",
                    "value": "tiling[BL0_matrix] is None"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if self.gemm_params.TILING.get("BL0_matrix") == []:
                data_amount_l0b = data_amount_l1b
                if data_amount_l0b > self.gemm_params.L0_SPACE_SIZE:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "tiling size exceed L0B Buffer",
                        "value": "tiling size = {tiling_size}".format(
                            tiling_size=data_amount_l0b
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
            else:
                _check_tilling_l0(
                    "BL0_matrix",
                    self.gemm_params.L0_SPACE_SIZE,
                    manual_pingpong_buffer.get("BL0_pbuffer"),
                    data_size
                )
                if self.gemm_params.TILING.get("AL0_matrix")[1] != self.gemm_params.TILING.get("BL0_matrix")[0]:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "axis k in tilling AL0 is "
                        "not equal to axis k in tilling BL0",
                        "value": "axis k in tilling AL0 = {},"
                        " axis k in tilling BL0 = {}".format(
                            self.gemm_params.TILING.get("AL0_matrix")[1],
                            self.gemm_params.TILING.get("BL0_matrix")[0]
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

        self.gemm_params.TILING, fusion_sums = get_tiling_result(kernel_name)
        data_amount_al1 = _check_tiling_al1()
        data_amount_bl1 = _check_tiling_bl1()
        self.gemm_params.print_debug("data_amount_al1:", data_amount_al1 / 1024)
        self.gemm_params.print_debug("data_amount_bl1:", data_amount_bl1 / 1024)
        if data_amount_al1 + data_amount_bl1 > self.gemm_params.L1_SPACE_SIZE:
            args_dict = {
                "errCode": "E60114",
                "reason": "tiling size exceed L1 Buffer",
                "value": "tiling size = {tiling_size}".format(
                    tiling_size=data_amount_al1 + data_amount_bl1
                )
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        _check_mul_al1_bl1()
        _check_tiling_l0a_l0b(data_amount_bl1)
        manual_pingpong_buffer = self.gemm_params.TILING.get("manual_pingpong_buffer")
        # check tilling in CL0
        _check_tilling_l0c(
            "CL0_matrix", self.gemm_params.L0C_SPACE_SIZE, manual_pingpong_buffer.get("CL0_pbuffer")
        )

        # check tilling in UB
        _check_tiling_ub()


    def _get_ub_pos(self):
        """
        judge whether UB <= L1
        when input dataType is int8, output dataType is float32

        Parameters
        ----------
        Returns
        ----------
        small_ub: flag (bool)
        """
        small_ub = False

        if self.gemm_params.ops_mode == "int8fp32":
            k_aub, m_aub = self.gemm_params.TILING.get("AUB_shape")[:2]
            k_bub, n_bub = self.gemm_params.TILING.get("BUB_shape")[:2]
            k_al1, multi_m_al1 = self.gemm_params.TILING.get("AL1_shape")[:2]
            k_bl1, multi_n_bl1 = self.gemm_params.TILING.get("BL1_shape")[:2]
            n_cl0, m_cl0 = self.gemm_params.TILING.get("CL0_matrix")[:2]

            small_ub = (k_aub <= k_al1) and (m_aub <= multi_m_al1 * m_cl0)
            small_ub &= (k_bub <= k_bl1) and (n_bub <= multi_n_bl1 * n_cl0)

        return small_ub


    def _get_aicore_tiling_factor(self):
        """
        using tilling parameter calculate factor

        :return: tilling factor from ub to ddr
                tilling factor from l0c to ub
                tilling factor from ddr to AL1
                tilling factor from ddr to Bl1
        """
        l0c_tiling_factor = [self.gemm_params.TILING["CL0_matrix"][0], self.gemm_params.TILING["CL0_matrix"][1]]

        # From LOC to GM [NumOfparts for N axis, NumOfparts for M axis ]
        l0c_parts = [
            self._int_ceil_div(
                self.gemm_params.DIM_MAP.get("out_shape")[-4] // self.gemm_params.TILING["block_dim"][1],
                l0c_tiling_factor[0]
            ),
            self._int_ceil_div(
                self.gemm_params.DIM_MAP.get("out_shape")[-3] // self.gemm_params.TILING["block_dim"][2],
                l0c_tiling_factor[1]
            )
        ]

        l0c_ub_tiling_factor = self.gemm_params.TILING["CUB_matrix"]
        l0c_ub_parts = [
            self._int_ceil_div(l0c_tiling_factor[0], l0c_ub_tiling_factor[0]),
            self._int_ceil_div(l0c_tiling_factor[1], l0c_ub_tiling_factor[1])
        ]

        if self.gemm_params.TILING["AL1_shape"]:  # AL1_shape = [(batch), n/16, m/16, 16, 16]
            al1_parts = [
                self._int_ceil_div(
                    self.gemm_params.DIM_MAP["A_matrix_dim"][-3],
                    self._int_ceil_div(self.gemm_params.TILING["AL1_shape"][0], self.gemm_params.block_reduce)
                ),
                self._int_ceil_div(l0c_parts[1], self.gemm_params.TILING["AL1_shape"][1])
            ]

        else:
            al1_parts = [1, 1]

        if self.gemm_params.TILING["BL1_shape"]:
            bl1_parts = [
                self._int_ceil_div(
                    self.gemm_params.DIM_MAP["B_matrix_dim"][-4],
                    self._int_ceil_div(self.gemm_params.TILING["BL1_shape"][0], self.gemm_params.block_reduce)
                ),
                self._int_ceil_div(l0c_parts[0], self.gemm_params.TILING["BL1_shape"][1])
            ]
        else:
            bl1_parts = [1, 1]

        if self.gemm_params.TILING["AUB_shape"]:
            aub_parts = [
                self._int_ceil_div(
                    self.gemm_params.DIM_MAP["A_matrix_dim"][1],
                    self._int_ceil_div(self.gemm_params.TILING["AUB_shape"][0], self.gemm_params.block_reduce)
                ),
                self._int_ceil_div(
                    l0c_parts[1],
                    self._int_ceil_div(
                        self.gemm_params.TILING["AUB_shape"][1], self.gemm_params.TILING["CL0_matrix"][1]
                    )
                )
            ]

        else:
            aub_parts = [1, 1]

        if self.gemm_params.TILING["BUB_shape"]:
            bub_parts = [
                self._int_ceil_div(
                    self.gemm_params.DIM_MAP["B_matrix_dim"][0],
                    self._int_ceil_div(self.gemm_params.TILING["BUB_shape"][0], self.gemm_params.block_reduce)
                ),
                self._int_ceil_div(
                    l0c_parts[0],
                    self._int_ceil_div(
                        self.gemm_params.TILING["BUB_shape"][1], self.gemm_params.TILING["CL0_matrix"][0]
                    )
                )
            ]
        else:
            bub_parts = [1, 1]

        return l0c_tiling_factor, l0c_ub_parts, al1_parts, bl1_parts, aub_parts, bub_parts


    def _get_aicore_tiling_factor_dynamic(self):
        """
        using tiling parameter calculate factor

        return: tiling factor from ub to ddr
                tiling factor from l0c to ub
                tiling factor from ddr to AL1
                tiling factor from ddr to BL1
        """
        l0c_tiling_factor = [self.gemm_params.TILING["CL0_matrix"][0], self.gemm_params.TILING["CL0_matrix"][1]]

        l0c_ub_tiling_factor = self.gemm_params.TILING["CUB_matrix"]
        # which part from l0c -> ub
        l0c_ub_parts = [
            self._int_ceil_div(l0c_tiling_factor[0], l0c_ub_tiling_factor[0]),
            self._int_ceil_div(l0c_tiling_factor[1], l0c_ub_tiling_factor[1])
        ]

        if self.gemm_params.TILING["AL1_shape"]: # AL1_shape = [K/block_reduce, m/16, 16, 16]
            al1_factors = [
                self.gemm_params.TILING["AL1_shape"][0] // self.gemm_params.block_reduce,
                self.gemm_params.TILING["AL1_shape"][1],
            ]
        else:
            m_parts = self._int_ceil_div(self.gemm_params.DIM_MAP["A_matrix_dim"][-4], l0c_tiling_factor[1])
            al1_factors_m = self._int_ceil_div(m_parts, self.gemm_params.TILING["block_dim"][2])
            al1_factors = [self.gemm_params.DIM_MAP["A_matrix_dim"][-3], al1_factors_m]

        if self.gemm_params.TILING["BL1_shape"]:
            bl1_factors = [
                self.gemm_params.TILING["BL1_shape"][0] // self.gemm_params.block_reduce,
                self.gemm_params.TILING["BL1_shape"][1],
            ]
        else:
            n_parts = self._int_ceil_div(self.gemm_params.DIM_MAP["B_matrix_dim"][-3], l0c_tiling_factor[0])
            bl1_factors_n = self._int_ceil_div(n_parts, self.gemm_params.TILING["block_dim"][1])
            bl1_factors = [self.gemm_params.DIM_MAP["B_matrix_dim"][-4], bl1_factors_n]

        return l0c_tiling_factor, l0c_ub_parts, al1_factors, bl1_factors


    def _get_mmad_factor(self):
        """
        get tilling factor in mmad

        :return:tilling factor for al0
                tilling factor for bl0
                tilling factor for reduce axis
        """
        if self.gemm_params.TILING.get("AL0_matrix"):
            al0_factor = [
                self.gemm_params.TILING.get("AL0_matrix")[0],
                self.gemm_params.TILING.get("AL0_matrix")[1]
            ]
        else:
            if self.gemm_params.TILING.get("BL0_matrix"):
                kl0_factor = self.gemm_params.TILING.get("BL0_matrix")[0]
            else:
                kl0_factor = self.gemm_params.DIM_MAP.get("A_matrix_dim")[-3]
            al0_factor = [self.gemm_params.TILING.get("CL0_matrix")[1], kl0_factor]

        if self.gemm_params.TILING.get("BL0_matrix"):
            bl0_factor = [
                self.gemm_params.TILING.get("BL0_matrix")[0],
                self.gemm_params.TILING.get("BL0_matrix")[1]
            ]
        else:
            if self.gemm_params.TILING.get("AL0_matrix"):
                kl0_factor = self.gemm_params.TILING.get("AL0_matrix")[1]
            else:
                kl0_factor = self.gemm_params.DIM_MAP.get("B_matrix_dim")[-4]
            bl0_factor = [kl0_factor, self.gemm_params.TILING.get("CL0_matrix")[0]]
        reduce_factor = bl0_factor[0]

        return al0_factor, bl0_factor, reduce_factor


    def _bind_multi_core(  # pylint: disable=too-many-arguments
        self,
        sch,
        c_gm,
        bl1_at_c_axis,
        cl1_out_inner,
        al1_at_c_axis,
        c_slice_axis,
        batch
    ):
        """
        bind multi core

        Parameter:
        sch: schedule
        c_gm: op
        bl1_at_c_axis: axis
        cl1_out_inner: axis
        al1_at_c_axis: axis
        c_slice_axis: axis
        batch: axis
        """
        if "block_dim" in self.gemm_params.TILING:
            block_dim = self.gemm_params.TILING["block_dim"]
        else:
            block_dim = [1, 1, 1, 1]
        blockidx = []
        # split batch axis
        if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
            batch_nparts = self.gemm_params.TENSOR_MAP['c_l0c'].shape[0].value
        else:
            batch_nparts = block_dim[0]

        batch_out_in = sch[c_gm].split(batch, nparts=batch_nparts)
        c_out_in = sch[c_gm].split(cl1_out_inner, nparts=block_dim[1])
        h_out_in = sch[c_gm].split(c_slice_axis, nparts=block_dim[2])

        # reorder
        sch[c_gm].reorder(
            batch_out_in[0],
            c_out_in[0],
            h_out_in[0],
            batch_out_in[1],
            bl1_at_c_axis,
            c_out_in[1],
            al1_at_c_axis,
            h_out_in[1]
        )

        def _do_bind():
            blocks = batch_nparts * block_dim[1] * block_dim[2]
            if blocks != 1:
                out_fused = sch[c_gm].fuse(batch_out_in[0], c_out_in[0], h_out_in[0])
                out_fused_out, _ = sch[c_gm].split(out_fused, nparts=blocks)
                bind_out, _ = sch[c_gm].split(out_fused_out, 1)
                blockidx = tvm.thread_axis("blockIdx.x")
                sch[c_gm].bind(bind_out, blockidx)
            else:
                blockidx = [batch_out_in[0], c_out_in[0], h_out_in[0]]

        _do_bind()
        return batch_out_in[1], h_out_in[1], c_out_in[1], blockidx


    def _get_l1_at_c_gm_axis(  # pylint: disable=too-many-arguments
        self,
        sch,
        c_gm,
        aub_parts,
        bub_parts,
        al1_parts,
        bl1_parts,
        c_in,
        c_slice_axis
    ):
        """
        get l1 at c gm axis

        Parameter:
        ------------------------------------------------------------------
        :param sch: schedule
        :param c_gm: op
        :param aub_parts: tilling factor for aub
        :param bub_parts: tilling factor for bub
        :param al1_parts: tilling factor for al1
        :param bl1_parts: tilling factor for bl1
        :param c_in: axis
        :param c_slice_axis: axis
        -------------------------------------------------------------------
        """
        al1_part = [al1_parts[0] // aub_parts[0], al1_parts[1] // aub_parts[1]]
        bl1_part = [bl1_parts[0] // bub_parts[0], bl1_parts[1] // bub_parts[1]]

        bl1_at_c_axis, _ = sch[c_gm].split(c_in, nparts=bl1_part[1])
        al1_at_c_axis, c_slice_axis = sch[c_gm].split(c_slice_axis, nparts=al1_part[1])

        return bl1_at_c_axis, al1_at_c_axis, c_slice_axis


    def _get_l1_mn_axis_at_l0c(  # pylint: disable=too-many-arguments
        self,
        sch,
        c_l0c,
        al0_m_outer,
        bl0_n_outer,
        al1_parts,
        bl1_parts,
        aub_parts,
        bub_parts
    ):
        """
        get l1 mn axis at l0c

        Parameter:
        ------------------------------------------------------------------
        :param sch: schedule
        :param c_gm: op
        :param al0_m_outer: axis
        :param bl0_n_outer: axis
        :param al1_parts: tilling factor for al1
        :param bl1_parts: tilling factor for bl1
        :param aub_parts: tilling factor for aub
        :param bub_parts: tilling factor for bub
        -------------------------------------------------------------------
        """
        aub_part = [al1_parts[0] // aub_parts[0], al1_parts[1] // aub_parts[1]]
        bub_part = [bl1_parts[0] // bub_parts[0], bl1_parts[1] // bub_parts[1]]
        al0_m_outer_outer, al0_m_outer_inner = sch[c_l0c].split(
            al0_m_outer, nparts=aub_part[1]
        )
        bl0_n_outer_outer, bl0_n_outer_inner = sch[c_l0c].split(
            bl0_n_outer, nparts=bub_part[1]
        )

        return bl0_n_outer_outer, al0_m_outer_outer, bl0_n_outer_inner, al0_m_outer_inner


    def _get_ub_k_axis_at_l0c(  # pylint: disable=too-many-arguments
        self,
        sch,
        c_l0c,
        al1_parts,
        bl1_parts,
        aub_parts,
        bub_parts,
        k_outer_outer
    ):
        """
        get ub k axis at l0c

        Parameter:
        ------------------------------------------------------------------
        :param sch: schedule
        :param c_l0c: op
        :param al1_parts: tilling factor for al1
        :param bl1_parts: tilling factor for bl1
        :param aub_parts: tilling factor for aub
        :param bub_parts: tilling factor for bub
        :param k_outer_outer: axis
        -------------------------------------------------------------------
        """

        def _get_order(k_dict):
            tmp_order = sorted(k_dict.items(), key=lambda d: d[1], reverse=True)
            axis_order = [i[0] for i in tmp_order]
            k_val_order = [i[1] for i in tmp_order]
            return k_val_order, axis_order

        def _change_order(axis_order, name):
            ub_tag = name + "ub"
            l1_tag = name + "l1"
            if axis_order.index(ub_tag) < axis_order.index(l1_tag) and k_dict.get(
                ub_tag
            ) == k_dict.get(l1_tag):
                index_ub = axis_order.index(ub_tag)
                index_l1 = axis_order.index(l1_tag)
                axis_order[index_ub] = l1_tag
                axis_order[index_l1] = ub_tag

        # get order
        k_dict = {
            "aub": aub_parts[0],
            "bub": bub_parts[0],
            "al1": al1_parts[0],
            "bl1": bl1_parts[0]
        }
        k_val_order, axis_order = _get_order(k_dict)
        _change_order(axis_order, "a")
        _change_order(axis_order, "b")

        reduce_axis_serial = list()

        def _do_split():
            outer = k_outer_outer
            for axis_parts in k_val_order:
                outer, inner = sch[c_l0c].split(outer, nparts=axis_parts)
                reduce_axis_serial.append(inner)
            reduce_axis_serial.append(outer)
            self.gemm_params.print_debug("axis_order", axis_order)

        _do_split()
        return reduce_axis_serial, axis_order


    def _get_l0c_and_l1_axis(  # pylint: disable=too-many-locals, too-many-arguments
        self,
        sch,
        c_gm,
        l0c_factor,
        al1_parts,
        bl1_parts
    ):
        """
        get l0c and l1 axis

        Parameter:
        ------------------------------------------------------------------
        :param sch: schedule
        :param c_gm: op
        :param l0c_factor: tilling factor for l0c
        :param al1_parts: tilling parts for al1
        :param bl1_parts: tilling parts for bl1
        -------------------------------------------------------------------
        """

        def _get_reorder_flag():
            reorder_flag = False
            if (
                self.gemm_params.TILING["AL1_shape"]
                and al1_parts[0] != 1
                and self.gemm_params.TILING["BL1_shape"]
                and bl1_parts[0] != 1
            ):
                if bl1_parts[1] >= al1_parts[1]:
                    reorder_flag = True
            if (
                self.gemm_params.TILING["AL1_shape"]
                and al1_parts[0] == 1
                and self.gemm_params.TILING["BL1_shape"]
                and bl1_parts[0] == 1
            ):
                if bl1_parts[1] >= al1_parts[1]:
                    reorder_flag = True
            if (
                self.gemm_params.TILING["BL1_shape"]
                and bl1_parts[0] != 1
                and self.gemm_params.TILING["AL1_shape"]
                and al1_parts[0] == 1
            ):
                reorder_flag = True
            return reorder_flag

        # split c_gm according to factor of loc and out_shape
        if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
            l0c_n_outer, l0c_n_inner = sch[c_gm].split(sch[c_gm].op.axis[-4], l0c_factor[0])
            l0c_m_outer, l0c_m_inner = sch[c_gm].split(sch[c_gm].op.axis[-3], l0c_factor[1])
        else:
            l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[-4], l0c_factor[0])
            l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[-3], l0c_factor[1])
        sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)
        # split c_gm according to factor of a_l1 and b_l1
        l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(
            l0c_m_outer, nparts=al1_parts[1]
        )
        l1_n_outer_outer, cl1_out_inner = sch[c_gm].split(l0c_n_outer, nparts=bl1_parts[1])

        # get batch axis, if batch is None, make one
        if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
            batch = sch[c_gm].op.reduce_axis[0]
        elif len(c_gm.shape) > 4:
            batch = c_gm.op.axis[0]
        else:
            batch, l1_n_outer_outer = sch[c_gm].split(l1_n_outer_outer, nparts=1)

        bl1_at_c_axis = l1_n_outer_outer
        al1_at_c_axis = l1_m_outer_outer
        c_slice_axis = l1_m_outer_inner

        batch_in, c_slice_axis, noii_axis, blockidx = self._bind_multi_core(
            sch, c_gm, bl1_at_c_axis, cl1_out_inner, al1_at_c_axis, c_slice_axis, batch
        )
        # reorder al1 and bl1 axis according to double buffer
        batch_in_out_axis, batch_in_inner_axis = sch[c_gm].split(batch_in, factor=1)
        reorder_flag = _get_reorder_flag()
        if self.gemm_params.ops_mode == "int8fp32":
            reorder_flag = False
        if reorder_flag:
            sch[c_gm].reorder(l1_m_outer_outer, batch_in_inner_axis, l1_n_outer_outer)
        else:
            sch[c_gm].reorder(l1_n_outer_outer, l1_m_outer_outer, batch_in_inner_axis)
        al1_at_c_axis = l1_m_outer_outer

        return (
            batch_in_out_axis,
            bl1_at_c_axis,
            al1_at_c_axis,
            c_slice_axis,
            l0c_n_inner,
            l0c_m_inner,
            l1_m_outer_outer,
            noii_axis,
            blockidx
        )


    def _get_l0c_and_l1_axis_dynamic(   # pylint: disable=too-many-locals, too-many-arguments
        self,
        sch,
        c_gm,
        l0c_factor,
        al1_factor,
        bl1_factor
    ):
        """
        get l0c and l1 axis

        Parameter:
        ------------------------------------------------------------------
        :param sch: schedule
        :param c_gm: op
        :param l0c_factor: tilling factor for l0c
        :param al1_factor: tilling factor for al1
        :param bl1_factor: tilling factor for bl1
        -------------------------------------------------------------------
        """

        def _get_reorder_flag():
            reorder_flag = False
            if self.gemm_params.TILING["AL1_shape"] and self.gemm_params.TILING["BL1_shape"]:
                if al1_factor[1] > bl1_factor[1]:
                    reorder_flag = True
            elif self.gemm_params.TILING["BL1_shape"] and self.gemm_params.TILING["AL1_shape"] == []:
                reorder_flag = True

            return reorder_flag

        def _bind_multi_core_dynamic():
            if self.gemm_params.TILING.get("block_dim"):
                block_dim = self.gemm_params.TILING["block_dim"]
            else:
                block_dim = [1, 1, 1, 1]

            # split batch axis
            batch_out = sch[c_gm].split(batch, nparts=block_dim[0])
            n_out = sch[c_gm].split(l1_n_outer_outer, nparts=block_dim[1])
            m_out = sch[c_gm].split(l1_m_outer_outer, nparts=block_dim[2])

            # reorder
            sch[c_gm].reorder(
                batch_out[0],
                n_out[0],
                m_out[0],
                batch_out[1],
                n_out[1],
                l1_n_outer_inner,
                m_out[1],
                l1_m_outer_inner
            )
            blocks = block_dim[0] * block_dim[1] * block_dim[2]
            if blocks != 1:
                out_fused = sch[c_gm].fuse(batch_out[0], n_out[0], m_out[0])
                out_fused_out, _ = sch[c_gm].split(out_fused, nparts=blocks)
                bind_out, _ = sch[c_gm].split(out_fused_out, 1)
                blockidx = tvm.thread_axis("blockIdx.x")
                sch[c_gm].bind(bind_out, blockidx)
            else:
                blockidx = [batch_out[0], n_out[0], m_out[0]]

            return batch_out[1], n_out[1], m_out[1], blockidx

        # split c_gm according to factor of l0c and out_shape
        l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[-4], l0c_factor[0])
        l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[-3], l0c_factor[1])
        sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)

        # split c_gm according to factor of a_l1 and b_l1
        l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(l0c_m_outer, al1_factor[1])
        l1_n_outer_outer, l1_n_outer_inner = sch[c_gm].split(l0c_n_outer, bl1_factor[1])

        # get batch axis, if batch is None, make one
        if len(c_gm.shape) > 4:
            batch = c_gm.op.axis[0]
        else:
            batch, l1_n_outer_outer = sch[c_gm].split(l1_n_outer_outer, nparts=1)

        batch_in, m_block_in, n_block_in, _ = _bind_multi_core_dynamic()
        bl1_at_c_axis = n_block_in
        al1_at_c_axis = m_block_in
        c_slice_axis = l1_m_outer_inner

        # reorder al1 and bl1 axis according to double buffer
        batch_in_out_axis, batch_in_inner_axis = sch[c_gm].split(batch_in, factor=1)
        reorder_flag = _get_reorder_flag()

        if reorder_flag:
            sch[c_gm].reorder(m_block_in, batch_in_inner_axis, n_block_in)
        else:
            sch[c_gm].reorder(n_block_in, m_block_in, batch_in_inner_axis)


        return (
            batch_in_out_axis,
            bl1_at_c_axis,
            al1_at_c_axis,
            c_slice_axis,
            l0c_n_inner,
            l0c_m_inner
        )


    def _get_l0a_and_l0b_axis(  # pylint: disable=too-many-arguments
        self,
        sch,
        c_l0c,
        new_c_col_axis,
        al0_axis_factor,
        bl0_axis_factor,
        reduce_axis_factor
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
        al0_m_outer, al0_m_inner = sch[c_l0c].split(new_c_col_axis[1], al0_axis_factor[0])
        bl0_n_outer, bl0_n_inner = sch[c_l0c].split(new_c_col_axis[0], bl0_axis_factor[1])
        # for reduce axis, al0 and b_l0b should be the same
        k_outer_outer, k_outer_inner = sch[c_l0c].split(reduce_out, reduce_axis_factor)

        sch[c_l0c].reorder(
            k_outer_outer,
            bl0_n_outer,
            al0_m_outer,
            bl0_n_inner,
            al0_m_inner,
            new_c_col_axis[2],
            new_c_col_axis[3],
            k_outer_inner,
            reduce_inner
        )

        return al0_m_outer, bl0_n_outer, k_outer_outer, bl0_n_inner


    def _get_al1_and_bl1_axis_dynamic(self, sch, c_l0c, al1_factor, bl1_factor, k_outer_outer, reduce_axis_factor):
        """
        get al1 and bli axis for dynamic mode
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
        # a_l1 and b_l1 slice can be different with CUB & CL0
        factor_al1 = al1_factor[0] // reduce_axis_factor
        factor_bl1 = bl1_factor[0] // reduce_axis_factor

        if self.gemm_params.TILING.get("AL1_shape") and self.gemm_params.TILING.get("BL1_shape"):
            if factor_bl1 > factor_al1:
                k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer, factor=factor_al1
                )
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer_outer, factor_bl1 // factor_al1
                )
                al1_at_l0c_axis = k_outer_outer_outer_inner
                bl1_at_l0c_axis = k_outer_outer_outer_outer
            else:
                k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer, factor=factor_bl1
                )
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer_outer, factor_al1 // factor_bl1
                )
                al1_at_l0c_axis = k_outer_outer_outer_outer
                bl1_at_l0c_axis = k_outer_outer_outer_inner
        else:
            if self.gemm_params.TILING.get("AL1_shape"):
                k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer, factor=factor_al1
                )
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer_outer, nparts=1
                )
                al1_at_l0c_axis = k_outer_outer_outer_inner
                bl1_at_l0c_axis = k_outer_outer_outer_outer
            elif self.gemm_params.TILING.get("BL1_shape"):
                k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer, factor=factor_bl1
                )
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer_outer, nparts=1
                )
                al1_at_l0c_axis = k_outer_outer_outer_outer
                bl1_at_l0c_axis = k_outer_outer_outer_inner
            else:
                k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer, nparts=1
                )
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer_outer, nparts=1
                )
                al1_at_l0c_axis = k_outer_outer_outer_inner
                bl1_at_l0c_axis = k_outer_outer_outer_outer

        reduce_axis_serial = [
            k_outer_outer_outer_outer,
            k_outer_outer_outer_inner,
            k_outer_outer_inner
        ]

        return al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial


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
        outer_factor = max(al1_parts[0], bl1_parts[0])
        inner_factor = min(al1_parts[0], bl1_parts[0])
        if outer_factor % inner_factor != 0:
            args_dict = {
                "errCode": "E60114",
                "reason": "illegal value of AL1_shape & BL1_shape",
                "value": "outer_factor = {}, inner_factor = {}".format(
                    outer_factor, inner_factor
                )
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        if al1_parts[0] > bl1_parts[0]:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer, nparts=al1_parts[0]
            )
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer_outer, nparts=(bl1_parts[0])
            )
            al1_at_l0c_axis = k_outer_outer_outer_inner
            bl1_at_l0c_axis = k_outer_outer_outer_outer
        else:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer, nparts=bl1_parts[0]
            )
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer_outer, nparts=(al1_parts[0])
            )
            al1_at_l0c_axis = k_outer_outer_outer_outer
            bl1_at_l0c_axis = k_outer_outer_outer_inner
        reduce_axis_serial = [
            k_outer_outer_outer_outer,
            k_outer_outer_outer_inner,
            k_outer_outer_inner
        ]
        return al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial


    def _set_data_layout_cv_split(self, res, sch):
        """set data layout for cube vector split
        input:
            res: tensor, the res of compute
            sch: raw schedule
        --------------------------
        return:
            TENSOR_MAP: tenosr info
            DIM_MAP: shape info
        """
        all_tensor, _ = self._get_all_tensors(res)

        def _init_map():
            """set DIM_MAP
            """
            # fill in dimmap
            self.gemm_params.DIM_MAP["out_shape"] = [int(i) for i in res.shape]
            self.gemm_params.DIM_MAP["a_shape"] = [
                int(i) for i in self.gemm_params.TENSOR_MAP["a_placehold"].shape
            ]
            self.gemm_params.DIM_MAP["A_matrix_dim"] = [
                int(i) for i in self.gemm_params.TENSOR_MAP["a_l0a"].shape
            ]
            self.gemm_params.DIM_MAP["B_matrix_dim"] = [
                int(i) for i in self.gemm_params.TENSOR_MAP["b_l0b"].shape
            ]
            self.gemm_params.DIM_MAP["b_shape"] = [
                int(i) for i in self.gemm_params.TENSOR_MAP["b_placehold"].shape
            ]


        self.gemm_params.TENSOR_MAP["c_gm"] = res
        self.gemm_params.TENSOR_MAP["a_l0a"] = all_tensor.get("tensor_a_matrix")
        sch[self.gemm_params.TENSOR_MAP["a_l0a"]].set_scope(tbe_platform_info.scope_ca)
        self.gemm_params.TENSOR_MAP["b_l0b"] = all_tensor.get("tensor_b_matrix")
        sch[self.gemm_params.TENSOR_MAP["b_l0b"]].set_scope(tbe_platform_info.scope_cb)

        self.gemm_params.TENSOR_MAP["a_placehold"] = all_tensor.get("tensor_a")
        self.gemm_params.TENSOR_MAP["b_placehold"] = all_tensor.get("tensor_b")

        if "tensor_a_fract" in all_tensor and "tensor_b_fract" in all_tensor:
            self.gemm_params.TENSOR_MAP["a_l1"] = all_tensor.get("tensor_a_fract")
            sch[self.gemm_params.TENSOR_MAP["a_l1"]].set_scope(tbe_platform_info.scope_cbuf)
            self.gemm_params.TENSOR_MAP["b_l1"] = all_tensor.get("tensor_b_fract")
            sch[self.gemm_params.TENSOR_MAP["b_l1"]].set_scope(tbe_platform_info.scope_cbuf)
            self.gemm_params.cv_split_nd_in_flag = True
        else:
            self.gemm_params.TENSOR_MAP["a_l1"] = sch.cache_read(self.gemm_params.TENSOR_MAP["a_placehold"],
                                                    tbe_platform_info.scope_cbuf,
                                                    [self.gemm_params.TENSOR_MAP["a_l0a"]])
            self.gemm_params.TENSOR_MAP["b_l1"] = sch.cache_read(self.gemm_params.TENSOR_MAP["b_placehold"],
                                                    tbe_platform_info.scope_cbuf,
                                                    [self.gemm_params.TENSOR_MAP["b_l0b"]])
            self.gemm_params.cv_split_nd_in_flag = False

        self.gemm_params.TENSOR_MAP["fix_pipe_bias"] = all_tensor.get("tensor_bias")
        self.gemm_params.TENSOR_MAP["c_l0c"] = all_tensor.get("tensor_c_matrix")
        sch[self.gemm_params.TENSOR_MAP["c_l0c"]].set_scope(tbe_platform_info.scope_cc)
        self._get_ops_mode()
        _init_map()


    def get_tiling_param_nz(self):
        """
        get tiling param for nz
        -------------------------
        return
        a_shape: list, a shape, format is nc1hwc0
        b_shape: list, b shape, format is nc1hwc0
        pad_l: fused_num in aub
        pad_r: fused_num in bub
        fused_num: fused_num in cub
        """
        a_type = self.gemm_params.TENSOR_MAP["a_placehold"].dtype
        b_type = self.gemm_params.TENSOR_MAP["b_placehold"].dtype
        c_type = self.gemm_params.TENSOR_MAP["c_gm"].dtype

        l0a_shape = self.gemm_params.DIM_MAP["A_matrix_dim"]
        l0b_shape = self.gemm_params.DIM_MAP["B_matrix_dim"]

        a_shape = [
            1,
            l0a_shape[-3],
            l0a_shape[-4],
            self.gemm_params.block_in,
            self.gemm_params.block_reduce
        ]
        b_shape = [
            l0b_shape[-4] * self.gemm_params.block_reduce,
            l0b_shape[-3],
            1,
            1,
            self.gemm_params.block_out
        ]

        if len(l0a_shape) == 5:
            a_shape[0] = l0a_shape[0]
        if (a_type == "int8" and c_type == "float32"):
            a_shape[1] //= 2
            a_shape[4] *= 2

        pad_l = 0
        pad_r = 0
        fused_num = 0
        if not self.gemm_params.cube_vector_split and not self.gemm_params.MAT_MUL:
            pad_l = self.gemm_params.AUB_FUSED_NUM.get(self.gemm_params.ops_mode)
            pad_r = self.gemm_params.BUB_FUSED_NUM.get(self.gemm_params.ops_mode)
            fused_num = self.gemm_params.CUB_FUSED_NUM.get(self.gemm_params.ops_mode)
        if self.gemm_params.fusion_type == FusionType.ELEWISE_FUSION:
            if self.gemm_params.TENSOR_MAP["fusion_ub"]:
                fused_num += 1
            if self.gemm_params.TENSOR_MAP["fusion_input"]:
                fused_num += 1
            rec_num = 0
            fp32_ub_flag = False
            for ub_tensor in self.gemm_params.TENSOR_MAP.get("fusion_ub", []):
                if ub_tensor.dtype == "float32":
                    fp32_ub_flag = True
                if "rec" in ub_tensor.op.name:
                    rec_num += 1
            if rec_num >= 5:
                fused_num += 2
            if fp32_ub_flag:
                fused_num *= 2
        if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
            fused_num = 1
        return a_shape, b_shape, pad_l, pad_r, fused_num


    def _set_data_layout_enter(self, res, sch):
        """
        if cube and vector split ,
        get into the func of self._set_data_layout_cv_split
        """
        if self.gemm_params.cube_vector_split:
            self._set_data_layout_cv_split(res, sch)
        else:
            self._set_data_layout(res, sch)


    def _get_trans_flag_cv_split(self):
        """
        trans_a and trans_b is the flag
        for cube vector split in current
        """
        if self.gemm_params.cube_vector_split:
            self.gemm_params.trans_a = self.gemm_params.TENSOR_MAP["a_l0a"].op.attrs["transpose_a"] == "true"
            self.gemm_params.trans_b = self.gemm_params.TENSOR_MAP["b_l0b"].op.attrs["transpose_b"] == "true"
        else:
            self.gemm_params.trans_a = False
            self.gemm_params.trans_b = False


    def _no_solution_tiling(self, tiling):
        """Determining that there is no solution to tilling
        and change tiling to default
        Input:
        tiling: dict, the tiling from tiling_query
        -----------------------------
        Return:
            default tiling
        """

        if tiling.get("AL0_matrix") == [1, 1, 32, 16, 1, 1]:
            block_reduce = self.gemm_params.block_reduce
            block_in = self.gemm_params.block_in
            block_out = self.gemm_params.block_out
            tiling = {'AUB_shape': [block_reduce, 1, 1, 1],
                    'BUB_shape': [block_reduce, 1, 1, 1],
                    'AL1_shape': [block_reduce, 1, 1, 1],
                    'BL1_shape': [block_reduce, 1, 1, 1],
                    'AL0_matrix': [1, 1, block_in, block_reduce, 1, 1],
                    'BL0_matrix': [1, 1, block_out, block_reduce, 1, 1],
                    'CL0_matrix': [1, 1, block_in, block_reduce, 1, 1],
                    'CUB_matrix': [1, 1, block_in, block_reduce, 1, 1],
                    'block_dim': [1, 1, 1, 1],
                    'n_bef_batch_flag': 0,
                    'n_bef_group_flag': 0,
                    'batch_bef_group_flag': 0,
                    'A_overhead_opt_flag': 0,
                    'B_overhead_opt_flag': 0,
                    'AUB_channel_wise_flag': None,
                    'BUB_channel_wise_flag': None,
                    'CUB_channel_wise_flag': 0,
                    'manual_pingpong_buffer':
                    {'AUB_pbuffer': 1,
                    'BUB_pbuffer': 1,
                    'AL1_pbuffer': 1,
                    'BL1_pbuffer': 1,
                    'AL0_pbuffer': 1,
                    'BL0_pbuffer': 1,
                    'CL0_pbuffer': 1,
                    'CUB_pbuffer': 1,
                    'UBG_pbuffer': 2
                    }
                    }
        return tiling


    def _atomic_add(self, sch, res):
        """
        atomic add according to refactor res
        """
        batch = sch[res].op.reduce_axis
        # set all batch to ddr add
        block_dim_batch = self.gemm_params.TENSOR_MAP['c_l0c'].shape[0].value
        batch_outer, batch_inner = sch[res].split(res.op.reduce_axis[0], nparts = block_dim_batch)
        res_after = res
        res_ub = sch.rfactor(res, batch_outer)
        sch[res_ub].set_scope(tbe_platform_info.scope_ubuf)
        # put reduce axis first
        sch[res_after].reorder(sch[res_after].op.reduce_axis[0], *sch[res_after].op.axis)
        sch[res_ub].reorder(sch[res_ub].op.reduce_axis[0], *sch[res_ub].op.axis[1:])
        return res_after, res_ub


    def gemm_schedule(self, res, sch_list, dynamic_para=None):
        """
        schedule enter
        param:
        res: tensor
        sch_list: list of schedule
        """
        self.gemm_params.UB_SPACE_SIZE = tbe_platform_info.get_soc_spec("UB_SIZE")
        self.gemm_params.L1_SPACE_SIZE = tbe_platform_info.get_soc_spec("L1_SIZE")
        self.gemm_params.L0_SPACE_SIZE = tbe_platform_info.get_soc_spec("L0A_SIZE")
        self.gemm_params.L0C_SPACE_SIZE = tbe_platform_info.get_soc_spec("L0C_SIZE")
        self.gemm_params.SOC_VERSION = tbe_platform_info.get_soc_spec("SOC_VERSION")
        self.gemm_params.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")

        sch = sch_list[0]
        if in_dynamic():
            self.gemm_params.is_dynamic = True
        self._set_data_layout_enter(res, sch)

        self.gemm_params.print_ir_matmul("orgin", sch)
        if self.gemm_params.fusion_type == FusionType.DEFAULT_MODE or self.gemm_params.cube_vector_split:
            kernel_name = self.gemm_params.TENSOR_MAP["c_gm"].op.attrs["kernel_name"]
        else:
            kernel_name = self.gemm_params.TENSOR_MAP["c_ub"].op.attrs["kernel_name"]
        self._get_trans_flag_cv_split()
        if self.gemm_params.is_dynamic:
            self.gemm_params.TILING = dynamic_para["tiling_strategy"]
        elif self.gemm_params.ops_format_mode != "ND":
            self._get_tiling(kernel_name)

        # get tensor
        a_l1, b_l1, a_l0a, b_l0b, c_l0c, c_gm = (
            self.gemm_params.TENSOR_MAP["a_l1"],
            self.gemm_params.TENSOR_MAP["b_l1"],
            self.gemm_params.TENSOR_MAP["a_l0a"],
            self.gemm_params.TENSOR_MAP["b_l0b"],
            self.gemm_params.TENSOR_MAP["c_l0c"],
            self.gemm_params.TENSOR_MAP["c_gm"]
        )

        if not self.gemm_params.is_dynamic and self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
            c_gm, ub_after_reduce = self._atomic_add(sch, res)

        c_ub = self.gemm_params.TENSOR_MAP.get("c_ub")

        if not self.gemm_params.MAT_MUL:
            alpha_ub, alpha_c_ub, beta_ub, bias_ub, c_ub_temp, beta_bias_ub, c_before_mul_ub = (
                self.gemm_params.TENSOR_MAP.get("alpha_ub"),
                self.gemm_params.TENSOR_MAP.get("alpha_c_ub"),
                self.gemm_params.TENSOR_MAP.get("beta_ub"),
                self.gemm_params.TENSOR_MAP.get("bias_ub"),
                self.gemm_params.TENSOR_MAP.get("c_ub_temp"),
                self.gemm_params.TENSOR_MAP.get("beta_bias_ub"),
                self.gemm_params.TENSOR_MAP.get("c_before_mul_ub")
            )
        else:
            bias_ub = self.gemm_params.TENSOR_MAP["bias_ub"]
            if bias_ub is not None:
                c_add_bias, bias_l0c = (
                    self.gemm_params.TENSOR_MAP["c_add_bias"],
                    self.gemm_params.TENSOR_MAP["bias_l0c"]
                )

        if not self.gemm_params.MAT_MUL:
            alpha_temp_ub = self.gemm_params.TENSOR_MAP.get("alpha_temp_ub")
            beta_temp_ub = self.gemm_params.TENSOR_MAP.get("beta_temp_ub")
            float32_bias_ub = self.gemm_params.TENSOR_MAP.get("float32_bias_ub")

        if self.gemm_params.ops_format_mode == "ND":
            a_normalize_ub = self.gemm_params.TENSOR_MAP.get("a_normalize_ub")
            a_fract_k_ub = self.gemm_params.TENSOR_MAP.get("a_fract_k_ub")
            b_normalize_ub = self.gemm_params.TENSOR_MAP.get("b_normalize_ub")
            b_fract_ub = self.gemm_params.TENSOR_MAP.get("b_fract_ub")
            b_transpose_only = self.gemm_params.TENSOR_MAP.get("b_transpose_only")
            b_transpose_zero = self.gemm_params.TENSOR_MAP.get("b_transpose_zero")
            b_after_process = self.gemm_params.TENSOR_MAP.get("b_after_process")
            a_zero = self.gemm_params.TENSOR_MAP.get("a_zero")
            b_zero =  self.gemm_params.TENSOR_MAP.get("b_zero")
            if self.gemm_params.ops_mode == "int8int32":
                b_matrix_transpose = self.gemm_params.TENSOR_MAP.get("b_transpose")
                a_matrix_transpose = self.gemm_params.TENSOR_MAP.get("a_transpose")
            if self._is_int82fp32_nd():
                a_float16 = self.gemm_params.TENSOR_MAP.get("tensor_a_float16_normalize_ub")
                b_float16 = self.gemm_params.TENSOR_MAP.get("tensor_b_float16_normalize_ub")

        # only Nz int8fp32 will in
        if self.gemm_params.ops_mode == "int8fp32" and not self.gemm_params.MAT_MUL:
            a_ub, float16_a_ub, zz_a_ub, b_ub, float16_b_ub, zn_b_ub = (
                self.gemm_params.TENSOR_MAP.get("a_ub"),
                self.gemm_params.TENSOR_MAP.get("float16_a_ub"),
                self.gemm_params.TENSOR_MAP.get("zz_a_ub"),
                self.gemm_params.TENSOR_MAP.get("b_ub"),
                self.gemm_params.TENSOR_MAP.get("float16_b_ub"),
                self.gemm_params.TENSOR_MAP.get("zn_b_ub")
            )

        if self.gemm_params.ops_mode == "int8int32" and self.gemm_params.ops_format_mode != "ND" and not self.gemm_params.MAT_MUL:
            bias_ub_fract = self.gemm_params.TENSOR_MAP.get("bias_ub_fract")

        def _nd_process():  # pylint: disable=R0914,R0915
            """
            gemm nd schdule enter
            """
            # -------------------------------------boost_schedule_kit----------#
            def _tiling_check_none():
                if (
                    (tiling.get("AL1_shape") is None)
                    or (tiling.get("BL1_shape") is None)
                    or (tiling.get("CUB_matrix") is None)
                ):
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "AL1_shape/BL1_shape/CUB_matrix can't be None",
                        "value": "AL1_shape = {al1_shape}, BL1_shape = {bl1_shape},"
                        " CUB_matrix = {cub_matrix}".format(
                            al1_shape=tiling.get("AL1_shape"),
                            bl1_shape=tiling.get("BL1_shape"),
                            cub_matrix=tiling.get("CUB_matrix")
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if (
                    (tiling.get("AL0_matrix") is None)
                    or (tiling.get("BL0_matrix") is None)
                    or (tiling.get("CL0_matrix") is None)
                ):
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "AL0_matrix/BL0_matrix/CL0_matrix can't be None",
                        "value": "AL0_matrix = {al0_matrix}, BL0_matrix = "
                        "{bl0_matrix}, CL0_matrix = {cl0_matrix}".format(
                            al0_matrix=tiling.get("AL0_matrix"),
                            bl0_matrix=tiling.get("BL0_matrix"),
                            cl0_matrix=tiling.get("CL0_matrix")
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

            def _tiling_l0_process():
                if tiling.get("BL0_matrix") != []:
                    (
                        bl0_tiling_kb,
                        bl0_tiling_nb,
                        bl0_tiling_n0,
                        bl0_tiling_k0,
                        _,
                        _
                    ) = tiling.get("BL0_matrix")
                else:
                    (
                        bl0_tiling_kb,
                        bl0_tiling_nb,
                        bl0_tiling_n0,
                        bl0_tiling_k0
                    ) = list(i.value for i in b_l0b.shape)
                return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0

            def _tiling_l1_process():
                if tiling.get("AL1_shape") != []:
                    al1_tiling_k, al1_tiling_m, _, _ = tiling.get("AL1_shape")
                else:
                    if self.gemm_params.cube_vector_split:
                        if self.gemm_params.trans_a:
                            al1_ma, al1_k, al1_k0, _ = list(i.value for i in a_l1.shape)
                        else:
                            al1_k, al1_ma, _, al1_k0 = list(i.value for i in a_l1.shape)
                        al1_tiling_k = al1_k * al1_k0
                    else:
                        if self.gemm_params.ops_mode == "int8int32":
                            al1_ma, al1_k, _, al1_k0 = list(i.value for i in a_l1.shape)
                            al1_tiling_k = al1_k * al1_k0
                        else:
                            al1_ma, al1_k, _ = list(i.value for i in a_l1.shape)
                            al1_tiling_k = al1_k
                    al1_tiling_m = al1_ma
                if tiling.get("BL1_shape") != []:
                    bl1_tiling_k, bl1_tiling_n, _, _ = tiling.get("BL1_shape")
                else:
                    if self.gemm_params.cube_vector_split:
                        if self.gemm_params.trans_b:
                            bl1_kb, bl1_n, _, bl1_k0 = list(i.value for i in b_l1.shape)
                        else:
                            bl1_n, bl1_kb, bl1_k0, _ = list(i.value for i in b_l1.shape)
                    else:
                        if self.gemm_params.ops_mode == "int8int32":
                            bl1_kb, bl1_n, bl1_k0, _ = list(i.value for i in b_l1.shape)
                        else:
                            bl1_kb, bl1_n, bl1_k0 = list(i.value for i in b_l1.shape)
                    bl1_tiling_k = bl1_kb * bl1_k0
                    bl1_tiling_n = bl1_n
                return al1_tiling_k, al1_tiling_m, bl1_tiling_k, bl1_tiling_n

            def _tiling_check_equal():
                if tiling.get("BL0_matrix") != []:
                    if al0_tiling_ka != bl0_tiling_kb:
                        args_dict = {
                            "errCode": "E60114",
                            "reason": "ka != kb",
                            "value": "ka = {}, kb = {}".format(
                                al0_tiling_ka, bl0_tiling_kb
                            )
                        }
                        raise RuntimeError(
                            args_dict, error_manager_util.get_error_message(args_dict)
                        )
                    if bl0_tiling_nb != cl0_tiling_nc:
                        args_dict = {
                            "errCode": "E60114",
                            "reason": "nb != nc.",
                            "value": "nb = {}, nc = {}".format(
                                bl0_tiling_nb, cl0_tiling_nc
                            )
                        }
                        raise RuntimeError(
                            args_dict, error_manager_util.get_error_message(args_dict)
                        )

                if al0_tiling_ma != cl0_tiling_mc:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "ma != mc.",
                        "value": "ma = {}, mc = {}".format(al0_tiling_ma, cl0_tiling_mc)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

            def _tiling_check_factor():
                if al1_tiling_k % al0_tiling_ka != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "k_AL1 % ka != 0.",
                        "value": "k_AL1 = {}, ka = {}".format(al1_tiling_k, al0_tiling_ka)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
                    if bl1_tiling_k % bl0_tiling_kb != 0:
                        args_dict = {
                            "errCode": "E60114",
                            "reason": "k_BL1 % kb != 0.",
                            "value": "k_BL1 = {}, kb = {}".format(
                                bl1_tiling_k, bl0_tiling_kb
                            )
                        }
                        raise RuntimeError(
                            args_dict, error_manager_util.get_error_message(args_dict)
                        )

                if cl0_tiling_nc % cub_tiling_nc_factor != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "nc % nc_factor != 0.",
                        "value": "nc = {}, nc_factor = {}".format(
                            cl0_tiling_nc, cub_tiling_nc_factor
                        )
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                if tiling.get("BL1_shape") != []:
                    if al1_tiling_k > bl1_tiling_k and al1_tiling_k % bl1_tiling_k != 0:
                        args_dict = {
                            "errCode": "E60114",
                            "reason": "k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.",
                            "value": "k_AL1 = {}, k_BL1 = {}".format(
                                al1_tiling_k, bl1_tiling_k
                            )
                        }
                        raise RuntimeError(
                            args_dict, error_manager_util.get_error_message(args_dict)
                        )
                    if bl1_tiling_k > al1_tiling_k and bl1_tiling_k % al1_tiling_k != 0:
                        args_dict = {
                            "errCode": "E60114",
                            "reason": "k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.",
                            "value": "k_BL1 = {}, k_AL1 = {}".format(
                                bl1_tiling_k, al1_tiling_k
                            )
                        }
                        raise RuntimeError(
                            args_dict, error_manager_util.get_error_message(args_dict)
                        )

            def _cub_process():
                if self.gemm_params.cube_vector_split:
                    return
                affine_cub = (
                    cub_tiling_mc_factor * cub_tiling_m0,
                    cub_tiling_nc_factor * cub_tiling_n0
                )

                c_before_mul_ub_shape = list(i.value for i in c_before_mul_ub.shape)
                status = Compare.compare(
                    [
                        cub_tiling_nc_factor,
                        cub_tiling_mc_factor,
                        cub_tiling_m0,
                        cub_tiling_n0
                    ],
                    c_before_mul_ub_shape
                )
                self.gemm_params.print_debug("cub_status: ", status)
                if status == Compare.EQUAL:
                    pass
                elif status == Compare.LESS_EQ:
                    sch_agent.attach_at(c_before_mul_ub, c_gm, affine_shape=affine_cub)
                    sch_agent.same_attach(alpha_c_ub, c_before_mul_ub)
                    sch_agent.same_attach(bias_ub, c_before_mul_ub)
                    sch_agent.same_attach(beta_bias_ub, c_before_mul_ub)
                    sch_agent.same_attach(c_ub_temp, c_before_mul_ub)
                    if self.gemm_params.ops_mode == "fp16fp16":
                        sch_agent.same_attach(float32_bias_ub, c_before_mul_ub)
                        sch_agent.same_attach(c_ub, c_before_mul_ub)
                else:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "c_ub attach error.",
                        "value": "compare status = {}".format(status)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                return affine_cub

            def _cl0_process(affine_cub):
                affine_l0c = cl0_tiling_mc * cl0_tiling_m0, cl0_tiling_nc * cl0_tiling_n0
                if self.gemm_params.cube_vector_split:
                    sch_agent.attach_at(c_l0c, c_gm, affine_shape=affine_l0c)
                    return

                c_l0c_shape = list(i.value for i in c_l0c.shape)
                status_ori = Compare.compare(
                    [cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0],
                    c_l0c_shape
                )
                status = Compare.compare(affine_l0c, affine_cub)
                self.gemm_params.print_debug("cl0_status: ", status_ori, status)
                if status_ori == Compare.EQUAL:
                    pass
                elif status == Compare.EQUAL:
                    sch_agent.same_attach(c_l0c, c_before_mul_ub)
                elif status == Compare.GREATE_EQ:

                    sch_agent.attach_at(c_l0c, c_gm, affine_shape=affine_l0c)
                else:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "tensor_c_l0c attach error.",
                        "value": "compare status = {}".format(status)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                sch[c_l0c].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
                    (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
                    (1, 1),
                    (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][1])
                )

            def _l0a_process():
                l0a2l0c_affine_shape = (
                    None,
                    al0_tiling_ma,
                    al0_tiling_m0,
                    cl0_tiling_n0,
                    al0_tiling_ka,
                    al0_tiling_k0
                )
                tiling_ori_l0a = al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, al0_tiling_k0
                a_l0a_shape = list(i.value for i in a_l0a.shape)
                status_ori = Compare.compare(tiling_ori_l0a, a_l0a_shape)
                status = Compare.compare(
                    [al0_tiling_ma, al0_tiling_m0, al0_tiling_ka, al0_tiling_k0],
                    [cl0_tiling_mc, cl0_tiling_m0, c_col_k1, c_col_k0]
                )
                self.gemm_params.print_debug("al0_status: ", status_ori, status)
                if status_ori == Compare.EQUAL:
                    pass
                elif status == Compare.EQUAL:
                    sch_agent.same_attach(a_l0a, c_l0c)
                elif status == Compare.LESS_EQ:
                    sch_agent.attach_at(a_l0a, c_l0c, affine_shape=l0a2l0c_affine_shape)
                elif status == Compare.GREATE_EQ:
                    l0a2out_affine_shape = [al0_tiling_ma * al0_tiling_m0, None]
                    sch_agent.attach_at(a_l0a, c_gm, affine_shape=l0a2out_affine_shape)
                else:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "l0a attach error.",
                        "value": "compare status = {}".format(status)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

                sch[a_l0a].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][0]),
                    (1, tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][0])
                )

            def _l0b_process():
                l0b2l0c_affine_shape = (
                    bl0_tiling_nb,
                    None,
                    cl0_tiling_m0,
                    bl0_tiling_n0,
                    bl0_tiling_kb,
                    bl0_tiling_k0
                )
                tiling_ori_l0b = bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0
                b_l0b_shape = list(i.value for i in b_l0b.shape)
                status_ori = Compare.compare(tiling_ori_l0b, b_l0b_shape)
                status = Compare.compare(
                    [bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_kb, bl0_tiling_k0],
                    [cl0_tiling_nc, cl0_tiling_n0, c_col_k1, c_col_k0]
                )
                self.gemm_params.print_debug("bl0_status: ", status_ori, status)
                if status_ori == Compare.EQUAL:
                    pass
                elif status == Compare.EQUAL:
                    sch_agent.same_attach(b_l0b, c_l0c)
                elif status == Compare.LESS_EQ:
                    sch_agent.attach_at(b_l0b, c_l0c, affine_shape=l0b2l0c_affine_shape)
                elif status == Compare.GREATE_EQ:
                    l0b2out_affine_shape = [None, bl0_tiling_nb * bl0_tiling_n0]
                    sch_agent.attach_at(b_l0b, c_gm, affine_shape=l0b2out_affine_shape)
                else:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "l0b attach error.",
                        "value": "compare status = {}".format(status)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

            def _al1_process():
                transpose_a, _ = self._get_transpose()
                if al1_tiling_m == 0 or al1_tiling_k == 0:
                    return
                l1_ma = al1_tiling_m * al0_tiling_ma
                l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0
                if self.gemm_params.ops_mode == "int8int32":
                    tiling_ori_al1 = l1_ma, l1_ka, al0_tiling_m0, al0_tiling_k0
                elif transpose_a:
                    tiling_ori_al1 = l1_ka, l1_ma * al0_tiling_m0, al0_tiling_k0
                else:
                    tiling_ori_al1 = l1_ma, l1_ka * al0_tiling_k0, al0_tiling_m0
                al1_shape = list(i.value for i in a_l1.shape)
                if self.gemm_params.ops_mode != "int8int32" and transpose_a:
                    al1_shape[1] = al1_shape[1] // tiling.get("block_dim")[2]
                else:
                    al1_shape[0] = al1_shape[0] // tiling.get("block_dim")[2]
                l1a2l0c_affine_shape = (
                    None,
                    l1_ma,
                    al0_tiling_m0,
                    cl0_tiling_n0,
                    l1_ka,
                    al0_tiling_k0
                )
                status_ori = Compare.compare(tiling_ori_al1, al1_shape)
                status = Compare.compare(
                    [l1_ma, al0_tiling_m0, l1_ka, al0_tiling_k0],
                    [cl0_tiling_mc, cl0_tiling_m0, c_col_k1, c_col_k0]
                )
                self.gemm_params.print_debug("al1_status: ", status_ori, status)
                self.gemm_params.print_debug(
                    "tiling_ori_al1", tiling_ori_al1, "al1_shape", al1_shape
                )
                if status_ori == Compare.EQUAL:
                    # al1 full load but tiling.get("AL1_shape") is not []
                    pass
                elif status == Compare.EQUAL:
                    sch_agent.same_attach(a_l1, c_l0c)
                elif status == Compare.LESS_EQ:
                    sch_agent.attach_at(a_l1, c_l0c, affine_shape=l1a2l0c_affine_shape)
                elif status == Compare.GREATE_EQ:
                    l1a2out_affine_shape = [l1_ma * al0_tiling_m0, None]
                    sch_agent.attach_at(a_l1, c_gm, affine_shape=l1a2out_affine_shape)
                else:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "a_l1 atach error.",
                        "value": "compare status = {}".format(status)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

            def _bl1_process():
                _, transpose_b = self._get_transpose()
                if tiling.get("BL1_shape") != []:
                    l1_nb = bl1_tiling_n * bl0_tiling_nb
                    l1_kb = (bl1_tiling_k + bl0_tiling_k0 - 1) // bl0_tiling_k0
                    l1b2l0c_affine_shape = (
                        l1_nb,
                        None,
                        cl0_tiling_m0,
                        bl0_tiling_n0,
                        l1_kb,
                        bl0_tiling_k0
                    )
                    if self.gemm_params.ops_mode == "int8int32":
                        tiling_ori_bl1 = l1_kb, l1_nb, bl0_tiling_n0, bl0_tiling_k0
                    elif transpose_b:
                        tiling_ori_bl1 = l1_nb, l1_kb * bl0_tiling_k0, bl0_tiling_n0
                    else:
                        tiling_ori_bl1 = l1_kb, l1_nb * bl0_tiling_n0, bl0_tiling_k0
                    bl1_shape = list(i.value for i in b_l1.shape)
                    if self.gemm_params.ops_mode != "int8int32" and transpose_b:
                        bl1_shape[0] = bl1_shape[0] // tiling.get("block_dim")[1]
                    else:
                        bl1_shape[1] = bl1_shape[1] // tiling.get("block_dim")[1]
                    status_ori = Compare.compare(tiling_ori_bl1, bl1_shape)
                    status = Compare.compare(
                        [l1_nb, bl0_tiling_n0, l1_kb, bl0_tiling_k0],
                        [cl0_tiling_nc, cl0_tiling_n0, c_col_k1, c_col_k0]
                    )
                    self.gemm_params.print_debug("bl1_status: ", status_ori, status)
                    self.gemm_params.print_debug(
                        "tiling_ori_bl1", tiling_ori_bl1, "bl1_shape", bl1_shape
                    )
                    if status_ori == Compare.EQUAL:
                        # bl1 full load but tiling.get("BL1_shape") is not []
                        pass
                    elif status == Compare.EQUAL:
                        sch_agent.same_attach(b_l1, c_l0c)
                    elif status == Compare.LESS_EQ:
                        sch_agent.attach_at(b_l1, c_l0c, affine_shape=l1b2l0c_affine_shape)
                    elif status == Compare.GREATE_EQ:
                        l1b2out_affine_shape = [None, l1_nb * bl0_tiling_n0]
                        sch_agent.attach_at(b_l1, c_gm, affine_shape=l1b2out_affine_shape)
                    else:
                        args_dict = {
                            "errCode": "E60114",
                            "reason": "b_l1 attach error.",
                            "value": "compare status = {}".format(status)
                        }
                        raise RuntimeError(
                            args_dict, error_manager_util.get_error_message(args_dict)
                        )

            def _aub_process():
                if self.gemm_params.cube_vector_split:
                    return
                transpose_a, _ = self._get_transpose()
                l1_ma = al1_tiling_m * al0_tiling_ma
                l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0

                aub_tiling_k0 = tbe_platform.CUBE_MKN[a_fract_k_ub.dtype]["mac"][1]
                aub_tiling_m0 = 16

                a_ub_ori_shape = list(i.value for i in a_fract_k_ub.shape)
                if self.gemm_params.ops_mode != "int8int32" and transpose_a:
                    a_ub_ori_shape[1] = a_ub_ori_shape[1] // tiling.get("block_dim")[2]
                else:
                    a_ub_ori_shape[0] = a_ub_ori_shape[0] // tiling.get("block_dim")[2]
                if self.gemm_params.ops_mode == "int8int32":
                    tiling_ori_aub = (
                        (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
                        aub_tiling_m,
                        aub_tiling_m0,
                        al0_tiling_k0
                    )
                    aub_l1_shape = [
                        aub_tiling_m,
                        (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
                        aub_tiling_m0,
                        aub_tiling_k0
                    ]
                    status_l1 = Compare.compare(
                        [
                            aub_tiling_m,
                            aub_tiling_m0,
                            (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
                            aub_tiling_k0
                        ],
                        [l1_ma, al0_tiling_m0, l1_ka, al0_tiling_k0]
                    )
                elif transpose_a:
                    tiling_ori_aub = (
                        ((aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0),
                        aub_tiling_m * aub_tiling_m0,
                        aub_tiling_k0
                    )
                    aub_l1_shape = [
                        ((aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0),
                        aub_tiling_m * aub_tiling_m0,
                        aub_tiling_k0
                    ]
                    status_l1 = Compare.compare(
                        [
                            ((aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0),
                            aub_tiling_m * aub_tiling_m0,
                            aub_tiling_k0
                        ],
                        [l1_ka, l1_ma * al0_tiling_m0, al0_tiling_k0]
                    )
                else:
                    tiling_ori_aub = (
                        aub_tiling_m,
                        ((aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0)
                        * al0_tiling_k0,
                        aub_tiling_m0
                    )
                    aub_l1_shape = [
                        aub_tiling_m,
                        ((aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0)
                        * al0_tiling_k0,
                        aub_tiling_m0
                    ]
                    status_l1 = Compare.compare(
                        [
                            aub_tiling_m,
                            ((aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0)
                            * al0_tiling_k0,
                            aub_tiling_m0
                        ],
                        [l1_ma, l1_ka * al0_tiling_k0, al0_tiling_m0]
                    )

                status_ori = Compare.compare(tiling_ori_aub, a_ub_ori_shape)
                status_l0c = Compare.compare(
                    [
                        aub_tiling_m,
                        aub_tiling_m0,
                        (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
                        aub_tiling_k0
                    ],
                    [cl0_tiling_mc, cl0_tiling_m0, c_col_k1, c_col_k0]
                )
                self.gemm_params.print_debug("aub_status: ", status_ori, status_l1, status_l0c)
                self.gemm_params.print_debug(
                    "tiling_ori_aub", tiling_ori_aub, "a_ub_ori_shape", a_ub_ori_shape
                )

                if status_ori == Compare.EQUAL:
                    pass
                elif status_l1 == Compare.EQUAL:
                    sch_agent.same_attach(a_fract_k_ub, a_l1)
                elif status_l1 == Compare.LESS_EQ:
                    sch_agent.attach_at(a_fract_k_ub, a_l1, aub_l1_shape)
                else:
                    if status_l0c == Compare.EQUAL:
                        sch_agent.same_attach(a_fract_k_ub, c_l0c)
                    elif status_l0c == Compare.LESS_EQ:
                        aub_l0c_affine_shape = (
                            None,
                            aub_tiling_m,
                            al0_tiling_m0,
                            cl0_tiling_n0,
                            (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
                            aub_tiling_k0
                        )
                        sch_agent.attach_at(
                            a_fract_k_ub, c_l0c, affine_shape=aub_l0c_affine_shape
                        )
                    else:
                        aub_out_affine_shape = [aub_tiling_m * aub_tiling_m0, None]
                        sch_agent.attach_at(
                            a_fract_k_ub, c_gm, affine_shape=aub_out_affine_shape
                        )
                sch_agent.same_attach(a_normalize_ub, a_fract_k_ub)

            def _bub_process():
                if self.gemm_params.cube_vector_split:
                    return
                _, transpose_b = self._get_transpose()
                l1_nb = bl1_tiling_n * bl0_tiling_nb
                l1_kb = (bl1_tiling_k + bl0_tiling_k0 - 1) // bl0_tiling_k0

                bub_tiling_k0 = tbe_platform.CUBE_MKN[b_fract_ub.dtype]["mac"][1]
                bub_tiling_n0 = 16

                b_ub_ori_shape = list(i.value for i in b_fract_ub.shape)
                if self.gemm_params.ops_mode != "int8int32" and transpose_b:
                    b_ub_ori_shape[0] = b_ub_ori_shape[0] // tiling.get("block_dim")[1]
                else:
                    b_ub_ori_shape[1] = b_ub_ori_shape[1] // tiling.get("block_dim")[1]

                if self.gemm_params.ops_mode == "int8int32":
                    tiling_ori_bub = (
                        (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                        bub_tiling_n,
                        bub_tiling_n0,
                        bub_tiling_k0
                    )
                    status_l1 = Compare.compare(
                        [
                            bub_tiling_n,
                            bub_tiling_n0,
                            (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                            bub_tiling_k0
                        ],
                        [l1_nb, bl0_tiling_n0, l1_kb, bl0_tiling_k0]
                    )
                    bub_shape = [
                        (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                        bub_tiling_n,
                        bub_tiling_n0,
                        bub_tiling_k0
                    ]
                elif transpose_b:
                    tiling_ori_bub = bub_tiling_n, bub_tiling_k, bub_tiling_n0
                    status_l1 = Compare.compare(
                        [bub_tiling_n, bub_tiling_k, bub_tiling_n0],
                        [l1_nb, l1_kb * bl0_tiling_k0, bl0_tiling_n0]
                    )
                    bub_shape = [bub_tiling_n, bub_tiling_k, bub_tiling_n0]
                else:
                    tiling_ori_bub = (
                        (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                        bub_tiling_n * bub_tiling_n0,
                        bub_tiling_k0
                    )
                    status_l1 = Compare.compare(
                        [
                            (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                            bub_tiling_n * bub_tiling_n0,
                            bub_tiling_k0
                        ],
                        [l1_kb, l1_nb * bl0_tiling_n0, bl0_tiling_k0]
                    )
                    bub_shape = [
                        ((bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0),
                        bub_tiling_n * bub_tiling_n0,
                        bub_tiling_k0
                    ]

                status_ori = Compare.compare(tiling_ori_bub, b_ub_ori_shape)
                status_l0c = Compare.compare(
                    [
                        bub_tiling_n,
                        bub_tiling_n0,
                        (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                        bub_tiling_k0
                    ],
                    [cl0_tiling_nc, cl0_tiling_n0, c_col_k1, c_col_k0]
                )

                self.gemm_params.print_debug("the status_ori is ", status_ori)
                self.gemm_params.print_debug(tiling_ori_bub)
                self.gemm_params.print_debug(b_ub_ori_shape)
                self.gemm_params.print_debug("the status l1 is ", status_l1)
                self.gemm_params.print_debug(
                    [
                        bub_tiling_n,
                        bub_tiling_n0,
                        (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                        bub_tiling_k0
                    ]
                )
                self.gemm_params.print_debug([l1_nb, bl0_tiling_n0, l1_kb, bl0_tiling_k0])
                self.gemm_params.print_debug("the status l0c is ", status_l0c)
                self.gemm_params.print_debug(
                    [
                        bub_tiling_n,
                        bub_tiling_n0,
                        (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                        bub_tiling_k0
                    ]
                )
                self.gemm_params.print_debug([cl0_tiling_nc, cl0_tiling_n0, c_col_k1, c_col_k0])

                if status_ori == Compare.EQUAL:
                    pass
                elif status_l1 == Compare.EQUAL:
                    sch_agent.same_attach(b_fract_ub, b_l1)
                elif status_l1 == Compare.LESS_EQ:
                    sch_agent.attach_at(b_fract_ub, b_l1, bub_shape)
                else:
                    if status_l0c == Compare.EQUAL:
                        sch_agent.same_attach(b_fract_ub, c_l0c)
                    elif status_l0c == Compare.LESS_EQ:
                        bub_l0c_affine_shape = (
                            bub_tiling_n,
                            None,
                            cl0_tiling_m0,
                            bl0_tiling_n0,
                            (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                            bub_tiling_k0
                        )
                        sch_agent.attach_at(
                            b_fract_ub, c_l0c, affine_shape=bub_l0c_affine_shape
                        )
                    else:
                        bub_out_affine_shape = [None, bub_tiling_n * bub_tiling_n0]
                        sch_agent.attach_at(
                            b_fract_ub, c_gm, affine_shape=bub_out_affine_shape
                        )
                sch_agent.same_attach(b_normalize_ub, b_fract_ub)

            def _do_padding_ub_process():
                if self.gemm_params.cube_vector_split:
                    return
                if a_zero is not None:
                    sch_agent.same_attach(a_zero, a_fract_k_ub)
                if b_zero is not None:
                    sch_agent.same_attach(b_zero, b_fract_ub)
                if b_transpose_only is not None:
                    tensor_list = [b_transpose_only, b_transpose_zero, b_after_process]
                    for i in tensor_list:
                        sch_agent.same_attach(i, b_fract_ub)
                if self.gemm_params.ops_mode == "int8int32":
                    if b_matrix_transpose is not None:
                        sch_agent.same_attach(b_matrix_transpose, b_fract_ub)
                    if a_matrix_transpose is not None:
                        sch_agent.same_attach(a_matrix_transpose, a_fract_k_ub)
                if self._is_int82fp32_nd():
                    sch_agent.same_attach(a_float16, a_fract_k_ub)
                    sch_agent.same_attach(b_float16, b_fract_ub)

            def _do_l1_process():
                if al1_tiling_k < bl1_tiling_k:
                    _al1_process()
                    _bl1_process()
                else:
                    _bl1_process()
                    _al1_process()

            def _do_ub_process():
                if aub_tiling_k < bub_tiling_k:
                    _aub_process()
                    _bub_process()
                else:
                    _bub_process()
                    _aub_process()

            def _do_l1_ub_process():
                if self.gemm_params.cube_vector_split:
                    return
                # get order
                k_dict = {
                    "aub": aub_tiling_k // self.gemm_params.block_reduce,
                    "bub": bub_tiling_k // self.gemm_params.block_reduce,
                    "al1": al1_tiling_k // int(al0_tiling_k0),
                    "bl1": bl1_tiling_k // int(bl0_tiling_k0)
                }

                tmp_order = sorted(k_dict.items(), key=lambda d: d[1], reverse=True)
                axis_order = [i[0] for i in tmp_order]
                self.gemm_params.print_debug("axis_order: ", axis_order)

                def _adjust_order(axis_order, ub_tag, l1_tag):
                    if axis_order.index(ub_tag) > axis_order.index(l1_tag) and k_dict.get(
                        ub_tag
                    ) == k_dict.get(l1_tag):
                        index_ub = axis_order.index(ub_tag)
                        index_l1 = axis_order.index(l1_tag)
                        axis_order[index_ub] = l1_tag
                        axis_order[index_l1] = ub_tag

                _adjust_order(axis_order, "aub", "al1")
                _adjust_order(axis_order, "bub", "bl1")

                for tag in axis_order[::-1]:
                    if tag == "bl1":
                        _bl1_process()
                    elif tag == "al1":
                        _al1_process()
                    elif tag == "bub":
                        _bub_process()
                    else:
                        _aub_process()

            def _double_buffer_aub():
                if self.gemm_params.cube_vector_split:
                    return
                sch[a_normalize_ub].double_buffer()
                sch[a_normalize_ub].preload()
                sch[a_fract_k_ub].double_buffer()
                if a_zero is not None:
                    sch[a_zero].double_buffer()
                if self.gemm_params.ops_mode == "int8int32":

                    if a_matrix_transpose is not None:
                        sch[a_matrix_transpose].double_buffer()
                if self._is_int82fp32_nd():
                    sch[a_float16].double_buffer()

            def _double_buffer_bub():
                if self.gemm_params.cube_vector_split:
                    return
                sch[b_normalize_ub].double_buffer()
                sch[b_normalize_ub].preload()
                sch[b_fract_ub].double_buffer()
                if b_zero is not None:
                    sch[b_zero].double_buffer()
                if b_transpose_only is not None:
                    tensor_list = [b_transpose_only, b_transpose_zero, b_after_process]
                    for i in tensor_list:
                        sch[i].double_buffer()
                if self.gemm_params.ops_mode == "int8int32":
                    if b_matrix_transpose is not None:
                        sch[b_matrix_transpose].double_buffer()

                if self._is_int82fp32_nd():
                    sch[b_float16].double_buffer()

            def _double_buffer_cub():
                if self.gemm_params.cube_vector_split:
                    return
                sch[c_before_mul_ub].double_buffer()
                sch[alpha_c_ub].double_buffer()
                sch[bias_ub].double_buffer()
                if c_ub is not None:
                    sch[c_ub].double_buffer()
                if self.gemm_params.ops_mode == "fp16fp16":
                    sch[float32_bias_ub].double_buffer()

                sch[beta_bias_ub].double_buffer()
                sch[c_ub_temp].double_buffer()

            def _double_buffer():
                if tiling.get("manual_pingpong_buffer").get("AUB_pbuffer") == 2:
                    _double_buffer_aub()
                if tiling.get("manual_pingpong_buffer").get("BUB_pbuffer") == 2:
                    _double_buffer_bub()
                if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == 2:
                    sch[a_l1].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == 2:
                    sch[b_l1].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == 2:
                    sch[a_l0a].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == 2:
                    sch[b_l0b].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
                    sch[c_l0c].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") == 2:
                    _double_buffer_cub()

            def _buffer_align():
                if self.gemm_params.cube_vector_split:
                    return
                if self._is_int82fp32_nd():
                    sch[c_ub_temp].buffer_align((1, 16), (1, 16))
                    sch[bias_ub].buffer_align((1, 32), (1, 32))
                    sch[beta_bias_ub].buffer_align((1, 32), (1, 32))
                    sch[a_normalize_ub].buffer_align((1, 32), (1, 32))
                    sch[b_normalize_ub].buffer_align((1, 32), (1, 32))
                    sch[a_float16].buffer_align((1, 32), (1, 32))
                    sch[b_float16].buffer_align((1, 32), (1, 32))
                    if a_zero is not None:
                        sch[a_zero].buffer_align((1, 32), (1, 32))
                    if b_zero is not None:
                        sch[b_zero].buffer_align((1, 32), (1, 32))
                else:
                    sch[c_ub_temp].buffer_align((1, 16), (1, 16))
                    sch[bias_ub].buffer_align((1, 16), (1, 16))
                    sch[beta_bias_ub].buffer_align((1, 16), (1, 16))
                if self.gemm_params.ops_mode == "fp16fp16":
                    sch[float32_bias_ub].buffer_align((1, 16), (1, 16))
                    sch[c_ub].buffer_align((1, 16), (1, 16))
                if self.gemm_params.ops_mode == "int8int32":
                    if b_matrix_transpose is not None:
                        sch[b_matrix_transpose].buffer_align((1, 32), (1, 32))
                    if a_matrix_transpose is not None:
                        sch[a_matrix_transpose].buffer_align((1, 32), (1, 32))
                if b_transpose_only is not None:
                    tensor_list = [b_transpose_only, b_transpose_zero, b_after_process]
                    for i in tensor_list:
                        sch[i].buffer_align((1, 32), (1, 32))

            def _emit_insn_common():  # pylint:disable=too-many-locals,too-many-statements
                scopes_intrins = sch_agent[c_l0c].intrin_scopes(6)
                scope_insn = scopes_intrins[0]

                scopes_intrins_c = sch_agent[c_gm].intrin_scopes(2)
                scope_c = scopes_intrins_c[0]

                if self.gemm_params.cube_vector_split:
                    scope_c_inner = scopes_intrins_c[1]
                    scope_c_inner_outer, scope_c_inner_inner = sch[c_gm].split(
                        scope_c_inner, factor=16)
                sch_agent[b_l0b].emit_insn(sch_agent[b_l0b].op.axis[0], "dma_copy")

                if not self.gemm_params.cube_vector_split:
                    sch_agent[b_normalize_ub].emit_insn(
                        sch_agent[b_normalize_ub].op.axis[0], "dma_copy"
                    )
                    scopes_intrins_cub = sch_agent[c_before_mul_ub].intrin_scopes(4)
                    scope_cub = scopes_intrins_cub[0]
                    sch_agent[a_normalize_ub].emit_insn(
                        sch_agent[a_normalize_ub].op.axis[0], "dma_copy"
                    )

                if self._is_int82fp32_nd() or (self.gemm_params.ops_mode in ("fp16fp32", "fp16fp16")):
                    nlast = 3
                else:
                    nlast = 4

                al1_scopes_intrins = sch_agent[a_l1].intrin_scopes(nlast)
                al1_scope_insn = al1_scopes_intrins[0]

                if self.gemm_params.cube_vector_split:
                    dma_dict = {"layout_transform", "nd2nz"}
                    sch_agent[a_l1].emit_insn(al1_scope_insn, "dma_copy", dma_dict)
                else:
                    sch_agent[a_l1].emit_insn(al1_scope_insn, "dma_copy")

                bl1_intrins = sch_agent[b_l1].intrin_scopes(nlast)
                bl1_fract_insn = bl1_intrins[0]

                if self.gemm_params.cube_vector_split:
                    dma_dict = {"layout_transform", "nd2nz"}
                    sch_agent[b_l1].emit_insn(bl1_fract_insn, "dma_copy", dma_dict)
                else:
                    sch_agent[b_l1].emit_insn(bl1_fract_insn, "dma_copy")

                sch_agent[a_l0a].emit_insn(sch_agent[a_l0a].op.axis[0], "dma_copy")

                inner_k_axis = sch_agent[c_l0c].get_relate_scope(
                    c_l0c.op.reduce_axis[0], scope_insn
                )
                if inner_k_axis:
                    mad_dict = {
                        "mad_pattern": 0,
                        "k_outer": sch_agent[c_l0c].get_relate_scope(
                            c_l0c.op.reduce_axis[0], scope_insn
                        )
                    }
                else:
                    (
                        inner_nb,
                        inner_mb,
                        inner_mp,
                        inner_np,
                        inner_kb,
                        inner_kp
                    ) = scopes_intrins
                    inner_ko, inner_ki = sch_agent[c_l0c].split(inner_kb, nparts=1)
                    sch_agent[c_l0c].reorder(
                        inner_ko, inner_nb, inner_mb, inner_mp, inner_np, inner_ki, inner_kp
                    )
                    mad_dict = {"mad_pattern": 0, "k_outer": [inner_ko]}
                sch_agent[c_l0c].emit_insn(scope_insn, "mad", mad_dict)
                if not self.gemm_params.cube_vector_split:
                    sch_agent[c_before_mul_ub].emit_insn(scope_cub, "dma_copy")
                    sch_agent[alpha_c_ub].emit_insn(
                        sch_agent[alpha_c_ub].op.axis[0], "vector_muls"
                    )
                    sch_agent[bias_ub].emit_insn(sch_agent[bias_ub].op.axis[0], "dma_copy")
                    sch_agent[beta_bias_ub].emit_insn(
                        sch_agent[beta_bias_ub].op.axis[0], "vector_muls"
                    )
                    outer1, inner1 = sch_agent[c_ub_temp].split(c_ub_temp.op.axis[0], 16)
                    outer2, inner2 = sch_agent[c_ub_temp].split(c_ub_temp.op.axis[1], 16)
                    sch_agent[c_ub_temp].reorder(outer1, outer2, inner1, inner2)
                    sch_agent[c_ub_temp].emit_insn(inner1, "vector_add")

                if self.gemm_params.cube_vector_split:
                    dma_dict = {"layout_transform", "nz2nd"}
                    sch_agent[c_gm].emit_insn(scope_c, "dma_copy", dma_dict)
                else:
                    sch_agent[c_gm].emit_insn(scope_c, "dma_copy", {"no_overlap": 1})

                if a_zero is not None:
                    sch_agent[a_zero].emit_insn(sch_agent[a_zero].op.axis[0], "vector_dup")
                if b_zero is not None:
                    sch_agent[b_zero].emit_insn(sch_agent[b_zero].op.axis[0], "vector_dup")

                if self._is_int82fp32_nd():
                    sch_agent[a_float16].emit_insn(
                        sch_agent[a_float16].op.axis[0], "dma_copy"
                    )
                    sch_agent[b_float16].emit_insn(
                        sch_agent[b_float16].op.axis[0], "dma_copy"
                    )
                if b_transpose_only is not None:
                    tensor_list = [b_transpose_only, b_after_process]
                    for i in tensor_list:
                        k_outer, k_inner = sch_agent[i].split(i.op.axis[1], factor=32)
                        sch_agent[i].reorder(k_outer, i.op.axis[0], k_inner)
                        sch_agent[i].emit_insn(sch_agent[i].op.axis[0], "vnchwconv")
                    sch_agent[b_transpose_zero].emit_insn(
                        sch_agent[b_transpose_zero].op.axis[0], "dma_copy"
                    )


            def _slove_bank_conflict():
                """slove bank conflict by storage_align
                if aub_k or bub_n bigger than threshold_data_num,
                use storage_align to slove bank conflict of aub/bub

                c_ub always conflict, must be use storage_align
                Input: None
                ---------------------------------
                Return: None
                """
                if self.gemm_params.cube_vector_split:
                    return
                a_trans, b_trans = self._get_transpose()
                (a_ub_storage_align,
                b_ub_storage_align,
                c_ub_storage_align) = _check_exceed_ub(a_trans, b_trans)

                # the data gap in ub
                gap_value = self.gemm_params.block_reduce
                c_gap_value = (self.gemm_params.block_out + 1) * self.gemm_params.block_in
                aub_k, aub_m, _, _ = tiling.get("AUB_shape")
                bub_k, bub_n, _, _ = tiling.get("BUB_shape")
                aub_m *= tbe_platform.BLOCK_IN
                bub_n *= tbe_platform.BLOCK_OUT

                # the data stride in ub
                a_align_value = (aub_m + gap_value) if a_trans else (aub_k + gap_value)
                b_align_value = (bub_k + gap_value) if b_trans else (bub_n + gap_value)

                # slove bank conflict in aub/bub
                if self._is_int82fp32_nd():
                    if a_ub_storage_align:
                        sch[a_float16].storage_align(a_float16.op.axis[0], a_align_value, 0)
                    if b_ub_storage_align:
                        sch[b_float16].storage_align(b_float16.op.axis[0], b_align_value, 0)
                else:
                    if (self.gemm_params.TENSOR_MAP["a_placehold"].dtype == "float16") and a_ub_storage_align:
                        if a_zero is not None:
                            sch[a_zero].storage_align(a_zero.op.axis[0], a_align_value, 0)
                        sch[a_normalize_ub].storage_align(a_normalize_ub.op.axis[0], a_align_value, 0)

                    if b_ub_storage_align:
                        if b_zero is not None:
                            sch[b_zero].storage_align(b_zero.op.axis[0], b_align_value, 0)
                        sch[b_normalize_ub].storage_align(b_normalize_ub.op.axis[0], b_align_value, 0)

                # slove bank conflict in cub
                if c_ub_storage_align:
                    sch[c_before_mul_ub].storage_align(c_before_mul_ub.op.axis[1], c_gap_value, 0)
                    sch[alpha_c_ub].storage_align(alpha_c_ub.op.axis[1], c_gap_value, 0)


            def _check_exceed_ub(a_trans, b_trans):
                """
                if storage_align is used, more UB space is used.
                Therefore, check the UB space usage after storage_align is used.
                Input:
                    a_trans: bool, Indicates whether matrix A is transposed.
                    b_trans: bool, Indicates whether matrix B is transposed.
                -----------------------
                Return:
                    a_ub_storage_align: bool, Matrix A uses storage_align.
                    b_ub_storage_align: bool, Matrix B uses storage_align.
                    c_ub_storage_align: bool, Matrix C uses storage_align.
                """
                threshold_data_num = 64
                gap_value = self.gemm_params.block_reduce
                ub_buffer = self.gemm_params.UB_SPACE_SIZE

                a_ub_storage_align = False
                b_ub_storage_align = False
                c_ub_storage_align = False
                aub_k, aub_m = tiling.get("AUB_shape")[0:2]
                bub_k, bub_n = tiling.get("BUB_shape")[0:2]
                cub_n, cub_m = tiling.get("CUB_matrix")[0:2]
                all_double_buffer = tiling.get("manual_pingpong_buffer")
                a_db = all_double_buffer.get("AUB_pbuffer")
                b_db = all_double_buffer.get("BUB_pbuffer")
                c_db = all_double_buffer.get("CUB_pbuffer")
                aub_m *= tbe_platform.BLOCK_IN
                bub_n *= tbe_platform.BLOCK_OUT

                # get fused num for compute use UB size
                a_fused_num, b_fused_num, c_fused_num = self._get_tiling_params(a_trans, b_trans)
                a_fused_num = a_fused_num / 10.0 + 1
                b_fused_num = b_fused_num / 10.0 + 1
                c_fused_num += 1

                # compute before storage_align used UB size
                base_buffer_size = 0
                base_buffer_size += (aub_m * aub_k * a_fused_num *
                                    self.gemm_params.INPUT_SIZE.get(self.gemm_params.ops_mode) * a_db)
                base_buffer_size += (bub_k * bub_n * b_fused_num *
                                    self.gemm_params.INPUT_SIZE.get(self.gemm_params.ops_mode) * b_db)
                base_buffer_size += (cub_n * cub_m * self.gemm_params.block_in * self.gemm_params.block_out *
                                    c_fused_num * self.gemm_params.OUTPUT_SIZE.get(self.gemm_params.ops_mode) * c_db)

                float32_int32_size = 4
                # if use storage_align, need UB size
                a_add_size = (gap_value * (aub_k if a_trans else aub_m) *
                            self.gemm_params.INPUT_SIZE.get(self.gemm_params.ops_mode) * a_db)
                b_add_size = (gap_value * (bub_n if b_trans else bub_k) *
                            self.gemm_params.INPUT_SIZE.get(self.gemm_params.ops_mode) * b_db)
                c_add_size = self.gemm_params.block_out * cub_n * cub_m * float32_int32_size * c_db

                if base_buffer_size + c_add_size <= ub_buffer:
                    base_buffer_size += c_add_size
                    c_ub_storage_align = True

                judge_value = aub_m if a_trans else aub_k
                if judge_value % threshold_data_num == 0 and (base_buffer_size + a_add_size) <= ub_buffer:
                    base_buffer_size += a_add_size
                    a_ub_storage_align = True

                judge_value = bub_k if b_trans else bub_n
                if judge_value % threshold_data_num == 0 and (base_buffer_size + b_add_size) <= ub_buffer:
                    base_buffer_size += b_add_size
                    b_ub_storage_align = True

                return a_ub_storage_align, b_ub_storage_align, c_ub_storage_align


            def _emit_insn_int8int32():
                if self.gemm_params.cube_vector_split:
                    return
                sch_agent[b_fract_ub].emit_insn(
                    sch_agent[b_fract_ub].op.axis[0], "dma_copy"
                )
                sch_agent[a_fract_k_ub].emit_insn(
                    sch_agent[a_fract_k_ub].op.axis[0], "dma_copy"
                )

                if b_matrix_transpose is not None:
                    k_outer, k_inner = sch_agent[b_matrix_transpose].split(
                        b_matrix_transpose.op.axis[1], factor=32
                    )
                    sch_agent[b_matrix_transpose].reorder(
                        k_outer, b_matrix_transpose.op.axis[0], k_inner
                    )
                    sch_agent[b_matrix_transpose].emit_insn(
                        sch_agent[b_matrix_transpose].op.axis[0], "vnchwconv"
                    )
                if a_matrix_transpose is not None:
                    m_outer, m_inner = sch_agent[a_matrix_transpose].split(
                        a_matrix_transpose.op.axis[1], factor=32
                    )
                    sch_agent[a_matrix_transpose].reorder(
                        m_outer, a_matrix_transpose.op.axis[0], m_inner
                    )
                    sch_agent[a_matrix_transpose].emit_insn(
                        sch_agent[a_matrix_transpose].op.axis[0], "vnchwconv"
                    )
                sch_agent[alpha_ub].emit_insn(sch_agent[alpha_ub].op.axis[0], "dma_copy")
                sch_agent[beta_ub].emit_insn(sch_agent[beta_ub].op.axis[0], "dma_copy")

            def _emit_insn_fp16fp16():
                if self.gemm_params.cube_vector_split:
                    return
                a_fract_intrins = sch_agent[a_fract_k_ub].intrin_scopes(3)
                a_fract_insn = a_fract_intrins[1]
                sch_agent[a_fract_k_ub].emit_insn(a_fract_insn, "vnchwconv")
                b_fract_intrins = sch_agent[b_fract_ub].intrin_scopes(3)
                b_fract_insn = b_fract_intrins[1]
                sch_agent[b_fract_ub].emit_insn(b_fract_insn, "vnchwconv")
                if self.gemm_params.ops_mode == "fp16fp16":
                    sch_agent[alpha_temp_ub].emit_insn(
                        sch_agent[alpha_temp_ub].op.axis[0], "dma_copy"
                    )
                    sch_agent[beta_temp_ub].emit_insn(
                        sch_agent[beta_temp_ub].op.axis[0], "dma_copy"
                    )
                    sch_agent[alpha_ub].emit_insn(
                        sch_agent[alpha_ub].op.axis[0], "vector_conv"
                    )
                    sch_agent[beta_ub].emit_insn(
                        sch_agent[beta_ub].op.axis[0], "vector_conv"
                    )
                    sch_agent[float32_bias_ub].emit_insn(
                        sch_agent[float32_bias_ub].op.axis[0], "vector_conv"
                    )
                    sch_agent[c_ub].emit_insn(sch_agent[c_ub].op.axis[0], "vector_conv")
                else:
                    sch_agent[alpha_ub].emit_insn(
                        sch_agent[alpha_ub].op.axis[0], "dma_copy"
                    )
                    sch_agent[beta_ub].emit_insn(sch_agent[beta_ub].op.axis[0], "dma_copy")

            def _buffer_reuse_fp16fp16():
                if self.gemm_params.cube_vector_split:
                    return
                sch_agent[c_before_mul_ub].reused_by(alpha_c_ub)
                if self.gemm_params.ops_mode == "fp16fp16":
                    sch_agent[float32_bias_ub].reused_by(beta_bias_ub, c_ub_temp)
                else:
                    sch_agent[bias_ub].reused_by(beta_bias_ub)

            def _buffer_reuse_int8int32():
                if self.gemm_params.cube_vector_split:
                    return
                sch_agent[c_before_mul_ub].reused_by(alpha_c_ub)

                sch[beta_bias_ub].reused_by(c_ub_temp)
                sch[bias_ub].reused_by(beta_bias_ub)

            def _renew_block_dim(tiling):
                """
                if tail data small then 16(output=fp16) or 32(output=int32)
                close multi core
                """

                if self.gemm_params.ops_mode == "int8int32":
                    multi_core_min_slice = 32
                else:
                    multi_core_min_slice = 16

                if (
                    c_gm.shape[1].value * self.gemm_params.OUTPUT_SIZE.get(self.gemm_params.ops_mode)
                    < multi_core_min_slice
                ):
                    tiling["block_dim"] = [1, 1, 1, 1]

                return tiling

            # -------------------------------------boost_schedule_kit end------#

            tiling = self._get_tiling_result_nd(kernel_name)
            tiling = _renew_block_dim(tiling)
            # tiling cub
            (
                cub_tiling_nc_factor,
                cub_tiling_mc_factor,
                cub_tiling_m0,
                cub_tiling_n0,
                _,
                _
            ) = tiling.get("CUB_matrix")

            # tiling l0c
            cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0, _, _ = tiling.get(
                "CL0_matrix"
            )

            # tiling l0a
            al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, al0_tiling_k0, _, _ = tiling.get(
                "AL0_matrix"
            )

            # tiling l0b
            (
                bl0_tiling_kb,
                bl0_tiling_nb,
                bl0_tiling_n0,
                bl0_tiling_k0
            ) = _tiling_l0_process()

            # tiling al1, bl1
            al1_tiling_k, al1_tiling_m, bl1_tiling_k, bl1_tiling_n = _tiling_l1_process()

            # tiling aub, bub
            if tiling.get("AUB_shape") is not None:
                aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")

            if tiling.get("BUB_shape") is not None:
                bub_tiling_k, bub_tiling_n, _, _ = tiling.get("BUB_shape")

            _tiling_check_equal()
            _tiling_check_factor()

            c_col_k1, c_col_k0 = list(ax.dom.extent.value for ax in c_l0c.op.reduce_axis)

            sch_agent = ScheduleAgent(sch)
            affine_cub = _cub_process()
            _cl0_process(affine_cub)
            _l0a_process()
            _l0b_process()
            _do_l1_ub_process()
            _do_padding_ub_process()
            _buffer_align()
            _double_buffer()

            ax_m, ax_n = sch_agent[c_gm].get_active_scopes()
            _, n_dim, m_dim, _ = tiling.get("block_dim")
            ax_core = sch_agent[c_gm].bind_core([ax_m, ax_n], [m_dim, n_dim])
            sch_agent.root_stage_at(c_gm, ax_core)
            if self.gemm_params.ops_mode == "int8int32":
                _buffer_reuse_int8int32()
                _emit_insn_int8int32()
            else:
                _buffer_reuse_fp16fp16()
                _emit_insn_fp16fp16()
            _emit_insn_common()
            _slove_bank_conflict()
            if b_transpose_only is not None:
                sch_agent[b_transpose_zero].reused_by(b_transpose_only)
            if b_zero is not None:
                sch_agent[b_normalize_ub].reused_by(b_zero)

            if a_zero is not None:
                sch_agent[a_normalize_ub].reused_by(a_zero)

            sch_agent.apply()
            self.gemm_params.print_ir_matmul("orgin_end", sch)

        def _nz_process():  # pylint: disable=R0914,R0915
            """
            nz schedule process enter
            """
            # --------------get factor and parts from tiling----------------
            if not self.gemm_params.is_dynamic:
                (
                    l0c_factor,
                    l0c_ub_parts,
                    al1_parts,
                    bl1_parts,
                    aub_parts,
                    bub_parts
                ) = self._get_aicore_tiling_factor()
            else:
                (
                    l0c_factor,
                    l0c_ub_parts,
                    al1_factor,
                    bl1_factor
                ) = self._get_aicore_tiling_factor_dynamic()
            al0_axis_factor, bl0_axis_factor, reduce_axis_factor = self._get_mmad_factor()
            # -----------split and get axis of l0c, al1, bl1 or may aub , bub----------------
            small_ub_flag = self._get_ub_pos()
            fix_pipe_bias = self.gemm_params.TENSOR_MAP.get("fix_pipe_bias")
            is_with_fix_pipe_bias = fix_pipe_bias is not None

            if self.gemm_params.ops_mode == "int8fp32" and not small_ub_flag and not self.gemm_params.cube_vector_split:
                (
                    batch_in_out_axis,
                    bub_at_c_axis,
                    aub_at_c_axis,
                    c_slice_axis,
                    l0c_n_inner,
                    l0c_m_inner,
                    _,
                    c_in,
                    _
                ) = self._get_l0c_and_l1_axis(sch, c_gm, l0c_factor, aub_parts, bub_parts)
                bl1_at_c_axis, al1_at_c_axis, c_slice_axis = self._get_l1_at_c_gm_axis(
                    sch,
                    c_gm,
                    aub_parts,
                    bub_parts,
                    al1_parts,
                    bl1_parts,
                    c_in,
                    c_slice_axis
                )
            else:
                if self.gemm_params.is_dynamic:
                    (
                        batch_in_out_axis,
                        bl1_at_c_axis,
                        al1_at_c_axis,
                        c_slice_axis,
                        l0c_n_inner,
                        l0c_m_inner,
                    ) = self._get_l0c_and_l1_axis_dynamic(sch, c_gm, l0c_factor, al1_factor, bl1_factor)
                else:
                    (
                        batch_in_out_axis,
                        bl1_at_c_axis,
                        al1_at_c_axis,
                        c_slice_axis,
                        l0c_n_inner,
                        l0c_m_inner,
                        _,
                        _,
                        _
                    ) = self._get_l0c_and_l1_axis(sch, c_gm, l0c_factor, al1_parts, bl1_parts)

            # -----------attach tensor of CUB----------------
            def _get_cub_at_axis(sch, c_gm, l0c_n_inner, l0c_m_inner, l0c_ub_parts):
                """
                get_cub_at_axis
                param: sch:schedule
                param: c_gm:tensor
                param: l0c_n_inner:axis
                param: l0c_m_inner:axis
                param: l0c_ub_parts: spilt parts
                return c_gm_at_axis, l0c_n_inner_inner
                """
                if not self.gemm_params.cube_vector_split:
                    l0c_n_inner_outer, l0c_n_inner_inner = sch[c_gm].split(
                        l0c_n_inner, nparts=l0c_ub_parts[0]
                    )
                    l0c_m_inner_outer, l0c_m_inner_inner = sch[c_gm].split(
                        l0c_m_inner, nparts=l0c_ub_parts[1]
                    )
                    sch[c_gm].reorder(
                        l0c_n_inner_outer,
                        l0c_m_inner_outer,
                        l0c_n_inner_inner,
                        l0c_m_inner_inner
                    )
                    c_gm_at_axis = l0c_m_inner_outer
                else:
                    c_gm_at_axis = None
                    l0c_n_inner_inner = l0c_n_inner
                return c_gm_at_axis, l0c_n_inner_inner

            c_gm_at_axis, l0c_n_inner_inner = _get_cub_at_axis(
                sch, c_gm, l0c_n_inner, l0c_m_inner, l0c_ub_parts
            )

            def _fix_pipe_bias_process():
                bias_l1 = sch.cache_read(
                    fix_pipe_bias, tbe_platform_info.scope_cbuf, [c_l0c])
                bias_fix_pipe = sch.cache_read(bias_l1, "local.FB", [c_l0c])
                sch[bias_fix_pipe].compute_at(sch[c_l0c], bl0_n_outer)
                sch[bias_l1].compute_at(sch[c_l0c], bl0_n_outer)
                sch[bias_l1].emit_insn(bias_l1.op.axis[0], "dma_copy")
                sch[bias_fix_pipe].emit_insn(bias_fix_pipe.op.axis[0], "dma_copy")

            def _attach_ub(sch, c_gm, at_axis):
                """
                tensor cub
                """
                if not self.gemm_params.MAT_MUL:
                    tensor_cub_list = [
                        c_ub,
                        c_before_mul_ub,
                        alpha_c_ub,
                        bias_ub,
                        c_ub_temp,
                        beta_bias_ub
                    ]
                    if self.gemm_params.ops_mode == "fp16fp16":
                        tensor_cub_list += [float32_bias_ub]
                    elif self.gemm_params.ops_mode == "int8int32":
                        tensor_cub_list += [bias_ub_fract]
                else:
                    tensor_cub_list = [c_ub]
                for tensor in tensor_cub_list:
                    sch[tensor].compute_at(sch[c_gm], at_axis)
                for input_tensor in self.gemm_params.TENSOR_MAP.get("fusion_input", []):
                    sch[input_tensor].compute_at(sch[c_gm], at_axis)
                for ub_tensor in self.gemm_params.TENSOR_MAP.get("fusion_ub", []):
                    sch[ub_tensor].compute_at(sch[c_gm], at_axis)

            if not self.gemm_params.cube_vector_split:
                _attach_ub(sch, c_gm, c_gm_at_axis)

            if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
                sch[ub_after_reduce].compute_at(sch[c_gm], c_gm_at_axis)

            # -----------attach tensor of l0c----------------
            if bias_ub is not None and self.gemm_params.MAT_MUL:
                sch[bias_ub].compute_at(sch[c_gm], c_slice_axis)
                sch[bias_l0c].compute_at(sch[c_gm], c_slice_axis)
                sch[c_add_bias].compute_at(sch[c_gm], c_slice_axis)
            sch[c_l0c].compute_at(sch[c_gm], c_slice_axis)

            new_c_col_axis = sch[c_l0c].op.axis[-4:]

            al0_m_outer, bl0_n_outer, k_outer_outer, bl0_n_inner = self._get_l0a_and_l0b_axis(
                sch,
                c_l0c,
                new_c_col_axis,
                al0_axis_factor,
                bl0_axis_factor,
                reduce_axis_factor
            )
            l0a_at_axis = al0_m_outer
            l0b_at_axis = bl0_n_outer
            # ---split and get axis of al1_at_l0c_axis, bl1_at_l0c_axis---
            if self.gemm_params.ops_mode == "int8fp32" and not small_ub_flag and not self.gemm_params.cube_vector_split:
                (
                    bl0_n_outer_outer,
                    al0_m_outer_outer,
                    bl0_n_outer_inner,
                    al0_m_outer_inner
                ) = self._get_l1_mn_axis_at_l0c(
                    sch,
                    c_l0c,
                    al0_m_outer,
                    bl0_n_outer,
                    al1_parts,
                    bl1_parts,
                    aub_parts,
                    bub_parts
                )

                reduce_axis_serial, axis_order = self._get_ub_k_axis_at_l0c(
                    sch, c_l0c, al1_parts, bl1_parts, aub_parts, bub_parts, k_outer_outer
                )
                aub_at_l0c_axis = reduce_axis_serial[axis_order.index("aub") + 1]
                bub_at_l0c_axis = reduce_axis_serial[axis_order.index("bub") + 1]
                al1_at_l0c_axis = reduce_axis_serial[axis_order.index("al1") + 1]
                bl1_at_l0c_axis = reduce_axis_serial[axis_order.index("bl1") + 1]

                def _reorder_mkn_axis():
                    if (axis_order.index("aub") in [2, 3]) and (
                        axis_order.index("bub") in [2, 3]
                    ):
                        if axis_order.index("al1") > axis_order.index("bl1"):
                            self.gemm_params.print_debug("order 1-1")
                            sch[c_l0c].reorder(
                                reduce_axis_serial[4],
                                reduce_axis_serial[3],
                                al0_m_outer_outer,
                                reduce_axis_serial[2],
                                bl0_n_outer_outer,
                                reduce_axis_serial[1],
                                reduce_axis_serial[0],
                                bl0_n_outer_inner,
                                al0_m_outer_inner
                            )
                        else:
                            self.gemm_params.print_debug("order 1-2")
                            sch[c_l0c].reorder(
                                reduce_axis_serial[4],
                                reduce_axis_serial[3],
                                bl0_n_outer_outer,
                                reduce_axis_serial[2],
                                al0_m_outer_outer,
                                reduce_axis_serial[1],
                                reduce_axis_serial[0],
                                bl0_n_outer_inner,
                                al0_m_outer_inner
                            )
                    elif axis_order.index("aub") == 3:
                        self.gemm_params.print_debug("order 2")
                        sch[c_l0c].reorder(
                            reduce_axis_serial[4],
                            al0_m_outer_outer,
                            reduce_axis_serial[3],
                            reduce_axis_serial[2],
                            bl0_n_outer_outer,
                            reduce_axis_serial[1],
                            reduce_axis_serial[0],
                            bl0_n_outer_inner,
                            al0_m_outer_inner
                        )
                    elif axis_order.index("bub") == 3:
                        self.gemm_params.print_debug("order 3")
                        sch[c_l0c].reorder(
                            reduce_axis_serial[4],
                            bl0_n_outer_outer,
                            reduce_axis_serial[3],
                            reduce_axis_serial[2],
                            al0_m_outer_outer,
                            reduce_axis_serial[1],
                            reduce_axis_serial[0],
                            bl0_n_outer_inner,
                            al0_m_outer_inner
                        )

                _reorder_mkn_axis()
                l0a_at_axis = al0_m_outer_inner
                l0b_at_axis = bl0_n_outer_inner

            else:
                if self.gemm_params.is_dynamic:
                    (
                        al1_at_l0c_axis,
                        bl1_at_l0c_axis,
                        reduce_axis_serial
                    ) = self._get_al1_and_bl1_axis_dynamic(sch, c_l0c, al1_factor, bl1_factor, k_outer_outer, reduce_axis_factor)
                else:
                    (
                        al1_at_l0c_axis,
                        bl1_at_l0c_axis,
                        reduce_axis_serial
                    ) = self._get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer)

            # -----------attach tensor of a_l0a----------------
            sch[a_l0a].compute_at(sch[c_l0c], l0a_at_axis)
            sch[b_l0b].compute_at(sch[c_l0c], l0b_at_axis)
            if is_with_fix_pipe_bias:
                _fix_pipe_bias_process()
            self.gemm_params.print_ir_matmul("before attach aub bub", sch)

            def _attach_aub_bub(al1_at_tensor, bl1_at_tensor):
                if self.gemm_params.ops_mode == "int8fp32" and not small_ub_flag:
                    sch[zz_a_ub].split(zz_a_ub.op.axis[1], factor=2)
                    zn_b_ub_k_out, _ = sch[zn_b_ub].split(zn_b_ub.op.axis[0], factor=2)
                    tensor_list = [a_ub, float16_a_ub, zz_a_ub]
                    if self.gemm_params.TILING["AUB_shape"]:
                        if aub_parts[0] != 1 or al1_at_tensor == "c_l0c":
                            self.gemm_params.print_debug("a_ub at c_l0c")
                            at_tensor = c_l0c
                            at_axis = aub_at_l0c_axis
                        else:
                            self.gemm_params.print_debug("a_ub at c_gm")
                            at_tensor = c_gm
                            at_axis = aub_at_c_axis
                    else:
                        self.gemm_params.print_debug("a_ub at c_gm2")
                        at_tensor = c_gm
                        at_axis = batch_in_out_axis

                    for tensor in tensor_list:
                        sch[tensor].compute_at(sch[at_tensor], at_axis)

                    tensor_list = [b_ub, float16_b_ub, zn_b_ub]
                    if self.gemm_params.TILING["BUB_shape"]:
                        if bub_parts[0] != 1 or bl1_at_tensor == "c_l0c":
                            self.gemm_params.print_debug("b_ub at c_l0c")
                            at_tensor = c_l0c
                            at_axis = bub_at_l0c_axis
                        else:
                            self.gemm_params.print_debug("b_ub at c_gm")
                            at_tensor = c_gm
                            at_axis = bub_at_c_axis
                    else:
                        self.gemm_params.print_debug("b_ub at c_gm 2")
                        at_tensor = c_gm
                        at_axis = batch_in_out_axis
                    for tensor in tensor_list:
                        sch[tensor].compute_at(sch[at_tensor], at_axis)
                    return zn_b_ub_k_out
                return None

            def _attach_aub_bub_small_ub(zn_b_ub_k_out):
                al1_axis = None
                bl1_axis = None
                if small_ub_flag:
                    al1_m_outer, al1_m_inner = sch[a_l1].split(
                        a_l1.op.axis[0], self.gemm_params.TILING.get("AUB_shape")[1]
                    )
                    al1_k_outer, al1_k_inner = sch[a_l1].split(
                        a_l1.op.axis[1], self.gemm_params.TILING.get("AUB_shape")[0] / 16
                    )
                    sch[a_l1].reorder(al1_m_outer, al1_k_outer, al1_m_inner, al1_k_inner)
                    al1_axis = al1_m_inner

                    bl1_k_outer, bl1_k_inner = sch[b_l1].split(
                        b_l1.op.axis[0], self.gemm_params.TILING.get("BUB_shape")[0] / 16
                    )
                    bl1_n_outer, bl1_n_inner = sch[b_l1].split(
                        b_l1.op.axis[1], self.gemm_params.TILING.get("BUB_shape")[1]
                    )
                    sch[b_l1].reorder(bl1_k_outer, bl1_n_outer, bl1_k_inner, bl1_n_inner)
                    bl1_axis = bl1_k_inner

                    sch[zz_a_ub].split(zz_a_ub.op.axis[1], factor=2)
                    zn_b_ub_k_out, _ = sch[zn_b_ub].split(zn_b_ub.op.axis[0], factor=2)

                    tensor_list = [a_ub, float16_a_ub, zz_a_ub]
                    for tensor in tensor_list:
                        sch[tensor].compute_at(sch[a_l1], al1_k_outer)

                    tensor_list = [b_ub, float16_b_ub, zn_b_ub]
                    for tensor in tensor_list:
                        sch[tensor].compute_at(sch[b_l1], bl1_n_outer)

                return al1_axis, bl1_axis, zn_b_ub_k_out

            def _attach_al1_bl1():
                if not self.gemm_params.is_dynamic:
                    if self.gemm_params.TILING["AL1_shape"]:
                        if al1_parts[0] != 1:
                            self.gemm_params.print_debug("a_l1 at c_l0c")
                            sch[a_l1].compute_at(sch[c_l0c], al1_at_l0c_axis)
                            al1_at_tensor = "c_l0c"
                        else:
                            self.gemm_params.print_debug("a_l1 at c_gm")
                            sch[a_l1].compute_at(sch[c_gm], al1_at_c_axis)
                            al1_at_tensor = "c_gm"
                    else:
                        self.gemm_params.print_debug("a_l1 at c_gm 2")
                        sch[a_l1].compute_at(sch[c_gm], batch_in_out_axis)
                        al1_at_tensor = "c_gm"

                    if self.gemm_params.TILING["BL1_shape"]:
                        if bl1_parts[0] != 1:
                            self.gemm_params.print_debug("b_l1 at c_l0c")
                            sch[b_l1].compute_at(sch[c_l0c], bl1_at_l0c_axis)
                            bl1_at_tensor = "c_l0c"
                        else:
                            self.gemm_params.print_debug("b_l1 at c_gm")
                            sch[b_l1].compute_at(sch[c_gm], bl1_at_c_axis)
                            bl1_at_tensor = "c_gm"
                    else:
                        self.gemm_params.print_debug("b_l1 at c_gm 2")
                        sch[b_l1].compute_at(sch[c_gm], batch_in_out_axis)
                        bl1_at_tensor = "c_gm"
                else:
                    if self.gemm_params.TILING["AL1_shape"]:
                        self.gemm_params.print_debug("a_l1 at c_l0c")
                        sch[a_l1].compute_at(sch[c_l0c], al1_at_l0c_axis)
                        al1_at_tensor = "c_l0c"
                    else:
                        self.gemm_params.print_debug("a_l1 at c_gm2")
                        sch[a_l1].compute_at(sch[c_gm], batch_in_out_axis)
                        al1_at_tensor = "c_gm"
                    if self.gemm_params.TILING["BL1_shape"]:
                        self.gemm_params.print_debug("b_l1 at c_l0c")
                        sch[b_l1].compute_at(sch[c_l0c], bl1_at_l0c_axis)
                        bl1_at_tensor = "c_l0c"
                    else:
                        self.gemm_params.print_debug("b_l1 at c_gm 2")
                        sch[b_l1].compute_at(sch[c_gm], batch_in_out_axis)
                        bl1_at_tensor = "c_gm"

                return al1_at_tensor, bl1_at_tensor

            def _do_reused_by():
                # reused_by
                if not self.gemm_params.MAT_MUL:
                    if self.gemm_params.ops_mode == "fp16fp16":
                        sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)

                        sch[float32_bias_ub].reused_by(beta_bias_ub)

                    elif self.gemm_params.ops_mode == "fp16fp32":
                        sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)
                        sch[bias_ub].reused_by(beta_bias_ub)
                    elif self.gemm_params.ops_mode == "int8int32":
                        sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)
                        sch[bias_ub].reused_by(beta_bias_ub)
                    elif self.gemm_params.ops_mode == "int8fp32":
                        sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)
                        sch[bias_ub].reused_by(beta_bias_ub)
                else:
                    if bias_ub is not None:
                        sch[bias_l0c].reused_by(c_add_bias, c_l0c)
                        sch[c_add_bias].emit_insn(c_add_bias.op.axis[0], "phony_insn")
                        sch[bias_l0c].emit_insn(bias_l0c.op.axis[0], "dma_copy")
                        sch[bias_ub].emit_insn(bias_ub.op.axis[0], "dma_copy")


            def _do_double_buffer_common():
                # double buffer
                # a_l1 b_l1
                temp_tensor_list = list()
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["AL1_pbuffer"] == 2 and (
                    self.gemm_params.TILING["AL1_shape"] != []
                ):
                    temp_tensor_list += [a_l1]
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["BL1_pbuffer"] == 2 and (
                    self.gemm_params.TILING["BL1_shape"] != []
                ):
                    temp_tensor_list += [b_l1]

                # L0A L0B
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["AL0_pbuffer"] == 2:
                    temp_tensor_list += [a_l0a]
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["BL0_pbuffer"] == 2:
                    temp_tensor_list += [b_l0b]

                # c_l0c C_UB
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["CL0_pbuffer"] == 2:
                    temp_tensor_list += [c_l0c]
                    if bias_ub is not None and self.gemm_params.MAT_MUL:
                        temp_tensor_list += [bias_l0c, c_add_bias]
                        sch[bias_l0c].preload()

                for temp_tensor in temp_tensor_list:
                    sch[temp_tensor].double_buffer()


            def _do_double_buffer_ub():
                """double buffer for tensor in ub
                INPUT: None
                ----------------------------
                RETURN: None
                """
                if self.gemm_params.cube_vector_split:
                    return None

                temp_tensor_list = list()
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["CUB_pbuffer"] == 2:
                    if not self.gemm_params.MAT_MUL:
                        temp_tensor_list += [
                            alpha_ub,
                            alpha_c_ub,
                            beta_ub,
                            bias_ub,
                            c_ub_temp,
                            beta_bias_ub,
                            c_ub,
                            c_before_mul_ub
                        ]
                    else:
                        for input_tensor in self.gemm_params.TENSOR_MAP.get("fusion_input", []):
                            temp_tensor_list.append(input_tensor)
                            sch[input_tensor].preload()
                        for ub_tensor in self.gemm_params.TENSOR_MAP.get("fusion_ub", []):
                            temp_tensor_list.append(ub_tensor)
                        if bias_ub is not None:
                            temp_tensor_list += [c_ub, bias_ub]
                            sch[bias_ub].preload()
                        else:
                            temp_tensor_list += [c_ub]
                    if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
                        temp_tensor_list += [ub_after_reduce]
                    if not self.gemm_params.MAT_MUL:
                        if self.gemm_params.ops_mode == "fp16fp16":
                            temp_tensor_list += [alpha_temp_ub, beta_temp_ub, float32_bias_ub]
                        elif self.gemm_params.ops_mode == "int8int32":
                            temp_tensor_list += [bias_ub_fract]

                if self.gemm_params.TILING.get("manual_pingpong_buffer")["AUB_pbuffer"] == 2:
                    temp_tensor_list += [float16_a_ub, a_ub, zz_a_ub]
                if self.gemm_params.TILING.get("manual_pingpong_buffer")["BUB_pbuffer"] == 2:
                    temp_tensor_list += [float16_b_ub, b_ub, zn_b_ub]

                for temp_tensor in temp_tensor_list:
                    sch[temp_tensor].double_buffer()


            def _do_intrin_mapping_common(al1_emit_axis, bl1_emit_axis):
                # intrin mapping
                temp_tensor_list = [
                    a_l0a,
                    b_l0b,
                ]
                if small_ub_flag:
                    sch[a_l1].emit_insn(al1_emit_axis, "dma_copy")
                    sch[b_l1].emit_insn(bl1_emit_axis, "dma_copy")
                elif self.gemm_params.cv_split_nd_in_flag:
                    dma_dict = {"layout_transform": "nd2nz"}
                    sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy", dma_dict)
                    sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy", dma_dict)
                else:
                    temp_tensor_list.append(a_l1)
                    temp_tensor_list.append(b_l1)

                for temp_tensor in temp_tensor_list:
                    sch[temp_tensor].emit_insn(temp_tensor.op.axis[0], "dma_copy")

                sch[c_gm].emit_insn(l0c_n_inner_inner, "dma_copy")

                if (self.gemm_params.ops_mode == "int8fp32"
                    and not small_ub_flag
                    and not self.gemm_params.cube_vector_split):
                    mad_dict = {
                        "mad_pattern": tbe_platform.GEMM_MODE,
                        "k_outer": [
                            reduce_axis_serial[0],
                            reduce_axis_serial[1],
                            reduce_axis_serial[2],
                            reduce_axis_serial[3],
                            reduce_axis_serial[4]
                        ]
                    }
                else:
                    mad_dict = {
                        "mad_pattern": tbe_platform.GEMM_MODE,
                        "k_outer": [
                            reduce_axis_serial[0],
                            reduce_axis_serial[1],
                            reduce_axis_serial[2]
                        ]
                    }
                if bias_ub is not None and self.gemm_params.MAT_MUL:
                    mad_dict["init_bias"] = 1
                sch[c_l0c].emit_insn(bl0_n_inner, "mad", mad_dict)


            def _do_intrin_mapping_ub(zn_b_ub_k_out):
                """intrin mapping for tensor in ub
                INPUT:
                    zn_b_ub_k_out, axis
                ---------------------------
                RETURN None
                """
                if self.gemm_params.cube_vector_split:
                    return None
                if not self.gemm_params.MAT_MUL:
                    temp_tensor_list = [
                        c_ub,
                        c_before_mul_ub,
                        alpha_ub,
                        beta_ub,
                        bias_ub
                    ]
                else:
                    temp_tensor_list = [c_ub]
                temp_tensor_list += self.gemm_params.TENSOR_MAP.get("fusion_input", [])
                if not self.gemm_params.MAT_MUL:
                    if self.gemm_params.ops_mode == "fp16fp16":
                        temp_tensor_list += [alpha_temp_ub, beta_temp_ub, float32_bias_ub]
                    elif self.gemm_params.ops_mode == "int8int32":
                        sch[bias_ub_fract].emit_insn(bias_ub_fract.op.axis[0], "vector_adds")
                    elif self.gemm_params.ops_mode == "int8fp32":
                        sch[zz_a_ub].emit_insn(zz_a_ub.op.axis[0], "vector_auto")
                        sch[zn_b_ub].emit_insn(zn_b_ub_k_out, "vector_auto")
                        temp_tensor_list += [float16_a_ub, float16_b_ub, a_ub, b_ub]
                for temp_tensor in temp_tensor_list:
                    sch[temp_tensor].emit_insn(temp_tensor.op.axis[0], "dma_copy")

                if not self.gemm_params.MAT_MUL:
                    sch[alpha_c_ub].emit_insn(alpha_c_ub.op.axis[0], "vector_muls")
                    if alpha_c_ub is not None:
                        sch[c_ub_temp].emit_insn(c_ub_temp.op.axis[0], "vector_add")
                    else:
                        sch[c_ub_temp].emit_insn(c_ub_temp.op.axis[0], "vector_muls")

                    sch[beta_bias_ub].emit_insn(beta_bias_ub.op.axis[0], "vector_muls")
                for tensor_ub in self.gemm_params.TENSOR_MAP.get("fusion_ub", []):
                    sch[tensor_ub].emit_insn(tensor_ub.op.axis[0], "vector_auto")
                if self.gemm_params.fusion_type == FusionType.REDUCE_FUSION:
                    sch[ub_after_reduce].emit_insn(ub_after_reduce.op.axis[0], "dma_copy")


            al1_at_tensor, bl1_at_tensor = _attach_al1_bl1()
            if not self.gemm_params.cube_vector_split:
                zn_b_ub_k_out = _attach_aub_bub(al1_at_tensor, bl1_at_tensor)
                al1_axis, bl1_axis, zn_b_ub_k_out = _attach_aub_bub_small_ub(zn_b_ub_k_out)
                _do_reused_by()
            else:
                zn_b_ub_k_out = None
                al1_axis = None
                bl1_axis = None
            _do_double_buffer_common()
            _do_double_buffer_ub()
            _do_intrin_mapping_common(al1_axis, bl1_axis)
            _do_intrin_mapping_ub(zn_b_ub_k_out)
            self.gemm_params.print_ir_matmul("finish", sch)

        if self.gemm_params.ops_format_mode == "ND":
            _nd_process()
        else:
            _nz_process()

        def _mem_process():
            def _get_al1_bound():
                if self.gemm_params.TILING["AL1_shape"]:
                    m_bound = self.gemm_params.TILING["AL1_shape"][1] * self.gemm_params.TILING["CL0_matrix"][1] * self.gemm_params.block_in
                    k_bound = self.gemm_params.TILING["AL1_shape"][0]
                    al1_bound = m_bound * k_bound
                else:
                    k_bound = self.gemm_params.DIM_MAP["A_matrix_dim"][-3] * self.gemm_params.block_reduce
                    if self.gemm_params.TILING["block_dim"][2] == 1:
                        m_bound = self.gemm_params.DIM_MAP["A_matrix_dim"][-4] * self.gemm_params.block_in
                    else:
                        m_parts = self._int_ceil_div(self.gemm_params.DIM_MAP["A_matrix_dim"][-4], self.gemm_params.TILING["CL0_matrix"][1])
                        m_factors = self._int_ceil_div(m_parts, self.gemm_params.TILING["block_dim"][2])
                        m_bound = m_factors * self.gemm_params.TILING["CL0_matrix"][1] * self.gemm_params.block_in
                    al1_bound = m_bound * k_bound
                return al1_bound

            def _get_bl1_bound():
                if self.gemm_params.TILING["BL1_shape"]:
                    n_bound = self.gemm_params.TILING["BL1_shape"][1] * self.gemm_params.TILING["CL0_matrix"][0] * self.gemm_params.block_out
                    k_bound = self.gemm_params.TILING["BL1_shape"][0]
                    bl1_bound = n_bound * k_bound
                else:
                    k_bound = self.gemm_params.DIM_MAP["B_matrix_dim"][-4] * self.gemm_params.block_reduce
                    if self.gemm_params.TILING["block_dim"][1] == 1:
                        n_bound = self.gemm_params.DIM_MAP["B_matrix_dim"][-3] * self.gemm_params.block_out
                    else:
                        n_parts = self._int_ceil_div(self.gemm_params.DIM_MAP["B_matrix_dim"][-3], self.gemm_params.TILING["CL0_matrix"][0])
                        n_factors = self._int_ceil_div(n_parts, self.gemm_params.TILING["block_dim"][1])
                        n_bound = n_factors * self.gemm_params.TILING["CL0_matrix"][0] * self.gemm_params.block_out
                    bl1_bound = n_bound * k_bound
                return bl1_bound

            if self.gemm_params.is_dynamic:
                sch.disable_allocate(tbe_platform_info.scope_cbuf)
                sch.disable_allocate(tbe_platform_info.scope_ca)
                sch.disable_allocate(tbe_platform_info.scope_cb)
                sch.disable_allocate(tbe_platform_info.scope_cc)
                sch.disable_allocate(tbe_platform_info.scope_ubuf)

                # get l1 bound
                sch[a_l1].set_storage_bound(_get_al1_bound())
                sch[b_l1].set_storage_bound(_get_bl1_bound())

                # mem_unique
                sch[a_l1].mem_unique()
                sch[b_l1].mem_unique()
                sch[a_l0a].mem_unique()
                sch[b_l0b].mem_unique()
                sch[c_ub].mem_unique()
                if bias_ub is not None:
                    sch[bias_ub].mem_unique()
                else:
                    sch[c_l0c].mem_unique()

        _mem_process()
        # clear global cache
        self.gemm_params.TILING.clear()
        self.gemm_params.DIM_MAP.clear()
        self.gemm_params.TENSOR_MAP.clear()
        return True


def gemm_schedule(res, sch_list, dynamic_para=None):
    """
    schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    """
    gemm_sch = GEMM_Schedule()
    return gemm_sch.gemm_schedule(res, sch_list, dynamic_para)
