# Copyright 2020 Huawei Technologies Co., Ltd
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
drop_out_do_mask.py
"""
# 'pylint: disable=too-many-arguments,too-few-public-methods,too-many-instance-attributes
from impl.util import util_select_op_base
from impl.util.util_tensor_dict import TensorClass
from impl.util.util_tensor_dict import get_format_for_format_ignore
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    # ting param num
    TILING_ARG_NUM = 16


# 'pylint: disable=too-many-locals,unused-argument,invalid-name
def op_select_format(input_tensor,
                     input_mask,
                     input_keep_prob,
                     output,
                     kernel_name="dropout_do_mask"):
    """
    Returns the dtype and format for DropoutDoMask
    """
    tensor_cls = TensorClass(input_tensor)

    dtype_base = ["float16", "float"]
    format_base = ["ND"]
    format_base += get_format_for_format_ignore(tensor_cls, need_align=True)

    tensor_dtype = []
    tensor_format = []
    for _format in format_base:
        tensor_dtype = tensor_dtype + dtype_base
        tensor_format = tensor_format + [_format] * len(dtype_base)

    input_tensor_dtype_str = ",".join(tensor_dtype)
    input_tensor_format_str = ",".join(tensor_format)
    input_mask_dtype_str = ",".join(["uint8"] * len(tensor_dtype))
    input_mask_format_str = ",".join(["ND"] * len(tensor_dtype))
    input_keep_prob_dtype_str = input_tensor_dtype_str
    input_keep_prob_format_str = input_mask_format_str

    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                           datatype=input_tensor_dtype_str,
                                           format=input_tensor_format_str,
                                           unknownshape_format=input_tensor_format_str)
    input1 = util_select_op_base.gen_param(classify="input1", name="mask",
                                           datatype=input_mask_dtype_str,
                                           format=input_mask_format_str,
                                           unknownshape_format=input_mask_format_str)
    input2 = util_select_op_base.gen_param(classify="input2", name="keep_prob",
                                           datatype=input_keep_prob_dtype_str,
                                           format=input_keep_prob_format_str,
                                           unknownshape_format=input_keep_prob_format_str)
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=input_tensor_dtype_str,
                                            format=input_tensor_format_str,
                                            unknownshape_format=input_tensor_format_str)

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


class DropOutDoMask:
    """
    Function: use to store dropoutdomask base parameters
    Modify: 2020-11-16
    """

    def __init__(self, var, mask, keep_prob, var_out, kernel_name):
        # reserved ub size
        reserved_ub_size = 8 * 1024
        # MAX REPEAT NUM
        max_repeat_num = 254
        self.tik_instance = tik.Tik(tik.Dprofile)
        self.var_dtype = var.get("dtype").lower()
        self.mask_dtype = mask.get("dtype").lower()
        self.keep_prob_dtype = keep_prob.get("dtype").lower()
        self.out_dtype = var_out.get("dtype").lower()

        # check dtype
        para_check.check_dtype(self.mask_dtype, ("uint8",), param_name="mask")
        para_check.check_dtype(self.var_dtype, ("float32", "float16"), param_name="var")
        if self.keep_prob_dtype != self.var_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("DropOutDoMask", "keep_prob", "var",
                                                                  self.keep_prob_dtype, self.var_dtype)
        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - reserved_ub_size)
        self.elememts_vector_fp16 = tbe_platform.ELEMENTS_VECTOR_OP_FP16

        self.mask_pre_core = 16 if self.var_dtype == "float16" else 8
        self.mask_value = 128 if self.var_dtype == "float16" else 64
        self.block_num = 16 if self.var_dtype == "float16" else 8
        self.vcetor_num = self.block_num * 8
        self.max_process_num = max_repeat_num * self.vcetor_num
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="ting_gm", \
        scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT32,), name="var_gm", scope=tik.scope_gm)
        self.keep_prob_gm = self.tik_instance.Tensor(self.keep_prob_dtype, (Constant.MAX_INT32,),
                                                     name="keep_prob_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_instance.Tensor(self.mask_dtype, (Constant.MAX_INT32,), name="mask_gm", \
        scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT32,), name="out_gm", scope=tik.scope_gm)

        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", self.keep_prob_dtype)
        # init ub
        self.var_ub = None
        self.tiling_ub = None
        self.prob_rec = None
        self.core_used_num = None
        self.do_num_per_core = None
        self.do_num_tail_core = None
        self.mask_ub = None
        self.zero_ub = None

        if self.var_dtype == "float32":
            self.one_ub = None
            self.sel_fp16_ub = None
            self.sel_fp32_ub = None

    def _tiling_args(self):
        """
        get runtime tiling parameters from tiling
        """
        self.core_used_num = self.tik_instance.Scalar("int64", name="core_used_num")
        self.do_num_per_core = self.tik_instance.Scalar("int64", name="do_num_per_core")
        self.do_num_tail_core = self.tik_instance.Scalar("int64", name="do_num_tail_core")
        self.core_used_num.set_as(self.tiling_ub[0])
        self.do_num_per_core.set_as(self.tiling_ub[1])
        self.do_num_tail_core.set_as(self.tiling_ub[2])

    def _init_ub_tensor(self):
        """
        compute the ub size of tensors
        """
        self.var_ub = self.tik_instance.Tensor(self.var_dtype, (self.max_process_num,),
                                               name="var_ub", scope=tik.scope_ubuf)
        self.mask_ub = self.tik_instance.Tensor(self.mask_dtype, (self.max_process_num + 7 // 8,),
                                                name="mask_ub", scope=tik.scope_ubuf)
        self.zero_ub = self.tik_instance.Tensor("float16", (self.elememts_vector_fp16,),
                                                name="zero_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.elememts_vector_fp16, self.zero_ub,
                                     0.0, 1, 1, 8)

        if self.var_dtype == "float32":
            self.one_ub = self.tik_instance.Tensor("float16", (self.elememts_vector_fp16,),
                                                   name="one_ub", scope=tik.scope_ubuf)
            self.sel_fp16_ub = self.tik_instance.Tensor("float16", (self.elememts_vector_fp16,),
                                                        name="sel_fp16_ub", scope=tik.scope_ubuf)
            self.sel_fp32_ub = self.tik_instance.Tensor(self.var_dtype, (self.elememts_vector_fp16,),
                                                        name="sel_fp32_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(self.elememts_vector_fp16, self.one_ub,
                                         1.0, 1, 1, 8)

    def _init_prob_scalar(self):
        """
        _init_prob_scalar
        """
        self.prob_rec = self.tik_instance.Scalar(self.keep_prob_dtype, name="keep_prob_scaler")
        with self.tik_instance.new_stmt_scope():
            keep_prob_ub = self.tik_instance.Tensor(self.keep_prob_dtype, (self.vcetor_num,),
                                                    name="keep_prob_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(keep_prob_ub, self.keep_prob_gm, 0, 1, 1, 0, 0)

            if self.is_suport_vdiv:
                one_ub_vdiv = self.tik_instance.Tensor(self.keep_prob_dtype, (self.vcetor_num,),
                                                       name="one_ub_vdiv", scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(1, one_ub_vdiv, 1.0, 1, 1, 8)
                self.tik_instance.vdiv(1, keep_prob_ub, one_ub_vdiv, keep_prob_ub, 1, 1, 1, 1, 8, 8, 8)
                self.prob_rec.set_as(keep_prob_ub[0])
            else:
                # when donot support vdiv, will use vrec
                keep_prob_ub_out = self.tik_instance.Tensor(self.keep_prob_dtype, (self.vcetor_num,),
                                                            name="keep_prob_ub_out", scope=tik.scope_ubuf)
                self.tik_instance.vrec(1, keep_prob_ub_out, keep_prob_ub, 1, 1, 1, 8, 8)
                _tik_fuc_vrec_newton(self.tik_instance, keep_prob_ub_out, keep_prob_ub, 1)
                self.prob_rec.set_as(keep_prob_ub_out[0])

    def _var_update(self, process_num):
        """
        _var_update
        """
        sel_loop = (process_num + self.elememts_vector_fp16 - 1) // self.elememts_vector_fp16
        with self.tik_instance.for_range(0, sel_loop) as vsel_loop_idx:
            output_offset = vsel_loop_idx * self.elememts_vector_fp16
            mask_offset = vsel_loop_idx * (self.elememts_vector_fp16 // 8)
            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(self.mask_ub[mask_offset])
            if self.var_dtype == "float32":
                self.tik_instance.vsel(self.elememts_vector_fp16, 0, self.sel_fp16_ub,
                                       cmpmask, self.one_ub, self.zero_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vconv(64, "", self.sel_fp32_ub, self.sel_fp16_ub, 2, 1, 1, 8, 4)
                self.tik_instance.vmul(64, self.var_ub[output_offset], self.sel_fp32_ub, self.var_ub[output_offset],
                                       2, 1, 1, 1, 8, 8, 8)
            else:
                self.tik_instance.vsel(self.elememts_vector_fp16, 0, self.var_ub[output_offset],
                                       cmpmask, self.var_ub[output_offset], self.zero_ub, 1, 1, 1, 1, 8, 8, 8)

    def _run_one_loop(self, gm_offset, process_num, vector_mask, prob_rec, is_last_tail_data=False):
        """
        _run_one_loop
        """
        _process_num_one_loop = process_num
        if is_last_tail_data:
            repeats = 1
        else:
            repeats = (_process_num_one_loop + self.vcetor_num - 1) // self.vcetor_num
        copy_burst_len = (_process_num_one_loop + self.block_num - 1) // self.block_num
        self.tik_instance.data_move(self.var_ub, self.var_gm[gm_offset],
                                    0, 1, copy_burst_len, 0, 0)
        # one block mean 32bype = 256 bool
        copy_burst_len = (_process_num_one_loop + 255) // 256
        self.tik_instance.data_move(self.mask_ub, self.mask_gm[gm_offset // 8], 0, 1, copy_burst_len, 0, 0)
        self.tik_instance.vmuls(vector_mask, self.var_ub, self.var_ub, prob_rec, repeats, 1, 1, 8, 8)
        self._var_update(_process_num_one_loop)
        copy_burst_len = (_process_num_one_loop + self.block_num - 1) // self.block_num
        self.tik_instance.data_move(self.out_gm[gm_offset], self.var_ub, 0, 1, copy_burst_len, 0, 0)

    def _run_one_core(self, _core_idx, process_num, prob_rec, is_tail_core=False):
        """
        _run drop out per core
        """
        copy_loop = process_num // self.max_process_num
        copy_tail = process_num % self.max_process_num

        # process algin data
        with self.tik_instance.for_range(0, copy_loop) as _copy_idx:
            _process_num_one_loop = self.max_process_num
            vector_mask = self.mask_value
            copy_gm_offset = _core_idx * self.do_num_per_core + _copy_idx * self.max_process_num
            self._run_one_loop(copy_gm_offset, _process_num_one_loop, vector_mask, prob_rec)

        _process_num_one_loop = (copy_tail // self.vcetor_num) * self.vcetor_num
        with self.tik_instance.if_scope(_process_num_one_loop > 0):
            # process tail vcetor data
            vector_mask = self.mask_value
            copy_gm_offset = _core_idx * self.do_num_per_core + copy_loop * self.max_process_num
            self._run_one_loop(copy_gm_offset, _process_num_one_loop, vector_mask, prob_rec)

        if is_tail_core:
            # process tail last vcetor data
            _process_num_one_loop = copy_tail % self.vcetor_num
            with self.tik_instance.if_scope(_process_num_one_loop > 0):
                vector_mask = _process_num_one_loop
                copy_gm_offset = \
                    _core_idx * self.do_num_per_core + \
                    copy_loop * self.max_process_num + (copy_tail // self.vcetor_num) * self.vcetor_num
                self._run_one_loop(copy_gm_offset, _process_num_one_loop, vector_mask, prob_rec, True)

    def drop_do_mask_operator(self):
        """_drop_do_mask_operator"""
        self._keep_prob_the_var()
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.mask_gm, self.keep_prob_gm),
                                   outputs=(self.out_gm,),
                                   flowtable=(self.tiling_gm,), config=opt_config)
        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": self.ub_size_bytes, "core_num": self.ai_core_num})

    def _keep_prob_the_var(self):
        """
        main process of dropout_do_mask
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as _core_idx:
            self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                      name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
            self._tiling_args()
            self._init_ub_tensor()
            self._init_prob_scalar()

            with self.tik_instance.if_scope(_core_idx < self.core_used_num - 1):
                self._run_one_core(_core_idx, self.do_num_per_core, self.prob_rec)
            with self.tik_instance.if_scope(_core_idx == self.core_used_num - 1):
                self._run_one_core(_core_idx, self.do_num_tail_core, self.prob_rec, True)


def _tik_fuc_vrec_newton(tik_instance, vrec_ub, origin_ub, do_len, newton_iteration=2, block_num=16):
    """tik_fuc_vrec_newton
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_1", scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_2", scope=tik.scope_ubuf)

        def _one_newton():
            tik_instance.vmul(1, vrec_newton_1, vrec_ub, origin_ub, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(1, vrec_newton_2, vrec_newton_1, -1, 1, 1, 1, 8, 8)
            tik_instance.vadds(1, vrec_newton_1, vrec_newton_2, 2, 1, 1, 1, 8, 8)
            tik_instance.vmul(1, vrec_ub, vrec_newton_1, vrec_ub, 1, 1, 1, 1, 8, 8, 8)

        for _ in range(newton_iteration):
            _one_newton()


@register_operator("DropOutDoMask")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def drop_out_do_mask(input_tensor, input_mask, input_keep_prob, output,
                     kernel_name="dropout_do_mask"):
    """
    algorithm: tf_dropout_do_mask
    scale_x = x*(1 / keep_prob)
    res = select(mask == 1, scale_x, 0)

    Parameters
    ----------
    input_tensor : dict,shape and dtype of input_tensor,only support float16 and float32
    input_mask : dict,shape and dtype of input_mask
        shape of mask,1D, dtype == uint8
        length=(size(shape_tensor)+tbe_platform.ELEMENTS_VECTOR_OP_FP16
        -1)/tbe_platform.ELEMENTS_VECTOR_OP_FP16*tbe_platform.ELEMENTS_VECTOR_OP_FP16/8
        eg. shape_tensor=[2,5,8] shape_mask=[16] shape_res=[2,5,8]
        shape_tensor=[15,17,19] shape_mask=[608] shape_res=[15,17,19]
    input_keep_prob : dict,shape and dtype of input_keep_prob
        shape of keep_prob, only 1 parament and equals to (1)
        prob scale (0.0,1.0] NOTICE: type same as dytpe
    output : dict,shape and dtype of output
    kernel_name : str
        cce kernel name, default value is "dropout_do_mask"

    Returns
    -------
    None
    """
    obj = DropOutDoMask(input_tensor, input_mask, input_keep_prob, output,
                        kernel_name)

    obj.drop_do_mask_operator()
