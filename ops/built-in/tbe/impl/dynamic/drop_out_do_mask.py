from te import tik
from te import platform as tbe_platform
import te.lang.dynamic
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


# max int32
MAX_INT32 = 2**31 - 1
# ting param num
TILING_ARG_NUM = 16
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# ub size
TENSOR_MAX_UB_NUM = 254*64


class DropOutDoMask:
    """
    Function: use to store dropoutdomask base parameters
    Modify: 2020-11-16
    """
    def __init__(self, var, mask, keep_prob, var_out, kernel_name):
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
        self.ai_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - RESERVED_UB_SIZE)

        self.mask_pre_core = 16 if self.var_dtype == "float16" else 8
        self.mask_value = 128 if self.var_dtype == "float16" else 64
        self.block_num = 16 if self.var_dtype == "float16" else 8
        self.vcetor_num = self.block_num*8
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="ting_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT32,), name="var_gm", scope=tik.scope_gm)
        self.keep_prob_gm = self.tik_instance.Tensor(self.keep_prob_dtype, (MAX_INT32,),
                                                     name="keep_prob_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_instance.Tensor(self.mask_dtype, (MAX_INT32,), name="mask_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT32,), name="out_gm", scope=tik.scope_gm)

        self.is_vsel_float = tbe_platform.cce_conf.api_check_support("te.lang.cce.vsel", "float32")
        # init ub
        self.var_ub = None
        self.keep_prob_ub = None
        self._core_idx = None
        self.tiling_ub = None

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
        self.var_ub = self.tik_instance.Tensor(self.var_dtype, (TENSOR_MAX_UB_NUM,), name="var_ub", scope=tik.scope_ubuf)
        self.keep_prob_ub = self.tik_instance.Tensor(self.var_dtype, (self.vcetor_num,), name="keep_prob_ub", scope=tik.scope_ubuf)
        self.mask_ub = self.tik_instance.Tensor(self.var_dtype, (TENSOR_MAX_UB_NUM,), name="mask_ub", scope=tik.scope_ubuf)
        self.out_ub = self.tik_instance.Tensor(self.var_dtype, (TENSOR_MAX_UB_NUM,), name="out_ub", scope=tik.scope_ubuf)
        self.zero_ub = self.tik_instance.Tensor(self.var_dtype, (self.vcetor_num,), name="zero_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.vcetor_num, self.zero_ub,
                                     0.0, 1, 1, 8)
        if not self.is_vsel_float and self.var_dtype == "float32":
            self.one_ub = self.tik_instance.Tensor(self.var_dtype, (self.vcetor_num,), name="one_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(self.vcetor_num, self.zero_ub,
                                         1.0, 1, 1, 8)
            self.tmp_ub = self.tik_instance.Tensor(self.var_dtype, (self.vcetor_num,), name="tmp_ub", scope=tik.scope_ubuf)



    def _var_update(self, vsel_loop):
        if not self.is_vsel_float and self.var_dtype == "float32":
            with self.tik_instance.new_stmt_scope():
                self.one_ub = self.tik_instance.Tensor(self.var_dtype, (self.vcetor_num,), name="one_ub", scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(self.vcetor_num, self.zero_ub,
                                             1.0, 1, 1, 8)
                self.tmp_ub = self.tik_instance.Tensor(self.var_dtype, (self.vcetor_num,), name="tmp_ub", scope=tik.scope_ubuf)
            pass
        else:
            with self.tik_instance.for_range(0, vsel_loop) as vsel_loop_idx:
                output_offset = vsel_loop_idx*self.vcetor_num
                mask_offset = vsel_loop_idx*(self.vcetor_num // 8)
                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(self.mask_ub[mask_offset])
                self.tik_instance.vsel(self.mask_value, 0, self.out_ub[output_offset],
                                       cmpmask, self.var_ub[output_offset], self.zero_ub, 1, 1, 1, 1, 8, 8, 8)

    def _run_one_loop(self, gm_offset, process_num, vector_mask, prob_rec, is_last_tail_data=False):
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
        self._var_update(repeats)
        copy_burst_len = (_process_num_one_loop + self.block_num - 1) // self.block_num
        self.tik_instance.data_move(self.out_gm[gm_offset], self.out_ub, 0, 1, copy_burst_len, 0, 0)

    def _run_one_core(self, _core_idx, process_num, prob_rec, is_tail_core=False):
        copy_loop = process_num // TENSOR_MAX_UB_NUM
        copy_tail = process_num % TENSOR_MAX_UB_NUM
        tail_repeats = self.tik_instance.Scalar("int64", name="tail_repeats")
        tail_repeats.set_as(copy_tail // self.vcetor_num)

        tail_last_num = self.tik_instance.Scalar("int64", name="tail_last_num")
        tail_last_num.set_as(copy_tail % self.vcetor_num)

        # process algin data
        with self.tik_instance.for_range(0, copy_loop) as _copy_idx:
            _process_num_one_loop = TENSOR_MAX_UB_NUM
            vector_mask = self.mask_value
            copy_gm_offset = _core_idx*process_num + _copy_idx*TENSOR_MAX_UB_NUM
            self._run_one_loop(copy_gm_offset, _process_num_one_loop, vector_mask, prob_rec)

        if is_tail_core:
            # process tail vcetor data
            _process_num_one_loop = (copy_tail // self.vcetor_num) * self.vcetor_num
            vector_mask = self.mask_value
            copy_gm_offset = _core_idx*process_num + copy_loop*TENSOR_MAX_UB_NUM
            self._run_one_loop(copy_gm_offset, _process_num_one_loop, vector_mask, prob_rec)

            # process tail last vcetor data
            _process_num_one_loop = copy_tail % self.vcetor_num
            vector_mask = _process_num_one_loop
            copy_gm_offset = \
                _core_idx*process_num + copy_loop*TENSOR_MAX_UB_NUM + (copy_tail // self.vcetor_num) * self.vcetor_num
            self._run_one_loop(copy_gm_offset, _process_num_one_loop, vector_mask, prob_rec,True)

    def _keep_prob_the_var(self):
        """
        main process of dropout_do_mask
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as _core_idx:
            self.tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
            self._tiling_args()
            self._init_ub_tensor()

            self.tik_instance.data_move(self.keep_prob_ub, self.keep_prob_gm, 0, 1, 1, 0, 0)
            self.tik_instance.vrec(1, self.keep_prob_ub, self.keep_prob_ub, 1, 1, 1, 8, 8)
            prob_rec = self.tik_instance.Scalar(self.var_dtype)
            prob_rec.set_as(self.keep_prob_ub[0])
            with self.tik_instance.if_scope(_core_idx < self.core_used_num - 1):
                self._run_one_core(_core_idx, self.do_num_per_core, prob_rec)
            with self.tik_instance.if_scope(_core_idx == self.core_used_num - 1):
                self._run_one_core(_core_idx, self.do_num_tail_core, prob_rec, True)

    def _drop_do_mask_operator(self):
        """_drop_do_mask_operator"""
        self._keep_prob_the_var()
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.mask_gm, self.keep_prob_gm),
                                   outputs=(self.out_gm,),
                                   flowtable=(self.tiling_gm,), config=opt_config)
        te.op.add_compile_info("vars", {"ub_size": self.ub_size_bytes, "core_num": self.ai_core_num})


@te.op.register_operator("DropOutDoMask")

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

    return obj._drop_do_mask_operator()
