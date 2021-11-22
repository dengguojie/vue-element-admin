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
assign.py
"""
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
    # tiling param num
    TILING_ARG_NUM = 16

# 'pylint: disable=too-many-instance-attributes,invalid-name
class Assign:
    """
    Class for Dynamic shape operator Assign
    """

    def __init__(self, ref, value, output, kernel_name):
        # reserved ub size
        RESERVED_UB_SIZE = 8 * 1024
        self.tik_instance = tik.Tik(tik.Dprofile)
        self.ref_dtype = ref.get("dtype").lower()
        self.value_dtype = value.get("dtype").lower()
        self.out_dtype = output.get("dtype").lower()

        # check dtype
        para_check.check_dtype(self.ref_dtype,
                               ("float16", "float32", "int8", "int32", "int64", "uint8",
                                "int16", "uint16", "uint32", "uint64"), param_name="ref")
        para_check.check_dtype(self.value_dtype,
                               ("float16", "float32", "int8", "int32", "int64", "uint8",
                                "int16", "uint16", "uint32", "uint64"), param_name="value")
        if self.ref_dtype != self.value_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("Assign", "ref", "value",
                                                                  self.ref_dtype, self.value_dtype)
        self.kernel_name = kernel_name

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE)
        self.max_burst_len = self.ub_size_bytes // (2 * 32)  # 2 means double buffer, 32 means one burst of UB is 32B

        if self.ref_dtype in ("int8", "uint8"):
            self.ele_per_block = 32
        elif self.ref_dtype in ("float16", "int16", "uint16"):
            self.ele_per_block = 16
        elif self.ref_dtype in ("float32", "int32", "uint32"):
            self.ele_per_block = 8
        else:
            self.ele_per_block = 4

        self.max_tensor_size = self.max_burst_len * self.ele_per_block

        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,), name="tiling_gm", \
        scope=tik.scope_gm)
        self.ref_gm = self.tik_instance.Tensor(self.ref_dtype, (Constant.MAX_INT32,), name="ref_gm", \
        scope=tik.scope_gm)
        self.value_gm = self.tik_instance.Tensor(self.value_dtype, (Constant.MAX_INT32,), name="value_gm", \
        scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.ref_dtype, (Constant.MAX_INT32,), name="out_gm", \
        scope=tik.scope_gm)

        self.tiling_ub = None
        self.value_ub = None
        self.out_ub = None

        self.core_used_num = self.tik_instance.Scalar("int64", name="core_used_num")
        self.block_per_core = self.tik_instance.Scalar("int64", name="block_per_core")
        self.block_tail_core = self.tik_instance.Scalar("int64", name="block_tail_core")

    def _tiling_args(self):
        """
        get runtime tiling parameters from tiling
        """
        # read tiling int64 scalar
        self.core_used_num.set_as(self.tiling_ub[0])
        self.block_per_core.set_as(self.tiling_ub[1])
        self.block_tail_core.set_as(self.tiling_ub[2])

    def _run_one_loop(self, gm_offset, burst_len, value_ub):
        self.tik_instance.data_move(value_ub, self.value_gm[gm_offset], 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.out_gm[gm_offset], value_ub, 0, 1, burst_len, 0, 0)

    def run_one_core(self, _core_idx, block_num):
        """
        run assign in one core
        """
        copy_loop = block_num // self.max_burst_len
        copy_tail = block_num % self.max_burst_len

        with self.tik_instance.for_range(0, copy_loop, thread_num=2) as _copy_idx:
            copy_gm_offset = _core_idx * self.block_per_core + _copy_idx * self.max_burst_len
            copy_gm_offset = copy_gm_offset * self.ele_per_block
            value_ub = self.tik_instance.Tensor(self.value_dtype, (self.max_tensor_size,),
                                                name="value_ub", scope=tik.scope_ubuf)
            self._run_one_loop(copy_gm_offset, self.max_burst_len, value_ub)

        with self.tik_instance.if_scope(copy_tail > 0):
            value_ub = self.tik_instance.Tensor(self.value_dtype, (self.max_tensor_size,),
                                                name="value_ub", scope=tik.scope_ubuf)
            copy_gm_offset = _core_idx * self.block_per_core + copy_loop * self.max_burst_len
            copy_gm_offset = copy_gm_offset * self.ele_per_block

            self._run_one_loop(copy_gm_offset, copy_tail, value_ub)

    def assign_compute(self):
        """
        The tik implementation of operator Assign
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as _core_idx:
            self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                      name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
            self._tiling_args()

            with self.tik_instance.if_scope(_core_idx < (self.core_used_num - 1)):
                self.run_one_core(_core_idx, self.block_per_core)
            with self.tik_instance.if_scope(_core_idx == (self.core_used_num - 1)):
                self.run_one_core(_core_idx, self.block_tail_core)

        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.ref_gm, self.value_gm),
                                   outputs=(self.out_gm,),
                                   flowtable=(self.tiling_gm,), config=opt_config)
        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": self.ub_size_bytes, "core_num": self.ai_core_num})


@register_operator("Assign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def assign(ref, value, output, kernel_name="assign"):
    """
    algorithm: assign
    calculating: update 'ref' by assigning 'value' to it

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign

    Returns
    -------
    None
    """
    obj = Assign(ref, value, output, kernel_name)
    obj.assign_compute()
