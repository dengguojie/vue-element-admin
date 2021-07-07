# Copyright 2021 Huawei Technologies Co., Ltd
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
scan_pq_codes.py
"""
from impl.util import util_tik_comm_func
from impl.util import util_common
from impl.util.util_tik_comm_func import OpBase
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

# max int64
MAX_INT64 = 2 ** 63 - 1
# ting param num
TILING_ARG_NUM = 16
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024


# pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class ScanPQCodes():
    """
    Function: use to store ScanPQCodes base parameters
    """

    def __init__(self, ivf, ivf_base_offset, ivf_cur_offset, ivf_cur_count, cur_adc_index,
                 adc_tables, pq_distance, pq_index, kernel_name):
        self.tik_instance = tik.Tik()
        self.unknown_max_shape = (MAX_INT64,)
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.opt_config = {"out_of_bound_sync_check": True,
                           "enable_const_fold": True}
        self.ivf_dtype = ivf.get("dtype").lower()
        self.ivf_shape = self.unknown_max_shape if util_common.is_unknown([ivf]) else ivf.get("shape")
        self.adc_tables_dtype = adc_tables.get("dtype").lower()
        self.adc_tables_shape = adc_tables.get("shape")
        self.ivf_base_offset_dtytpe = ivf_base_offset.get("dtype").lower()
        self.ivf_cur_offset_dtytpe = ivf_cur_offset.get("dtype").lower()
        self.ivf_cur_count_dtytpe = ivf_cur_count.get("dtype").lower()
        self.cur_adc_index_dtytpe = cur_adc_index.get("dtype").lower()
        # check dtype
        para_check.check_dtype(self.ivf_dtype, ("uint8"), param_name="ivf")
        para_check.check_dtype(self.adc_tables_dtype, ("float16"), param_name="adc_tables")
        para_check.check_dtype(self.ivf_base_offset_dtytpe, ("int32"), param_name="ivf_base_offset")
        para_check.check_dtype(self.ivf_cur_offset_dtytpe, ("int32"), param_name="ivf_cur_offset")
        para_check.check_dtype(self.ivf_cur_count_dtytpe, ("int32"), param_name="ivf_cur_count")
        para_check.check_dtype(self.cur_adc_index_dtytpe, ("int32"), param_name="cur_adc_index")

        self.kernel_name = kernel_name
        self.ub_size_bytes = self.ub_size_bytes - RESERVED_UB_SIZE
        self.elememts_vector_fp16 = tbe_platform.ELEMENTS_VECTOR_OP_FP16
        self.adc_tables_block_num = 16 if self.adc_tables_dtype in ("float16",) else 8
        # init gm addr
        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.ivf_gm = self.tik_instance.Tensor(self.ivf_dtype, self.ivf_shape,
                                               name="ivf_gm", scope=tik.scope_gm)
        self.ivf_base_offset_gm = self.tik_instance.Tensor(self.ivf_base_offset_dtytpe, [1, ],
                                                           name="ivf_base_offset_gm", scope=tik.scope_gm)
        self.ivf_cur_offset_gm = self.tik_instance.Tensor(self.ivf_cur_offset_dtytpe, [1, ],
                                                          name="ivf_cur_offset_gm", scope=tik.scope_gm)
        self.ivf_cur_count_gm = self.tik_instance.Tensor(self.ivf_cur_count_dtytpe, [1, ],
                                                         name="ivf_cur_count_gm", scope=tik.scope_gm)
        self.cur_adc_index_gm = self.tik_instance.Tensor(self.cur_adc_index_dtytpe, [1, ],
                                                         name="cur_adc_index_gm", scope=tik.scope_gm)
        self.adc_tables_gm = self.tik_instance.Tensor(self.adc_tables_dtype, self.unknown_max_shape,
                                                      name="adc_tables_gm", scope=tik.scope_gm)
        self.pq_distance_gm = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                       self.unknown_max_shape,
                                                       name="pq_distance_gm", scope=tik.scope_gm)
        self.pq_index_gm = self.tik_instance.Tensor(self.ivf_cur_offset_dtytpe,
                                                    self.unknown_max_shape,
                                                    name="pq_index_gm", scope=tik.scope_gm)
        add_index_init_list = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240] * 4
        pq_index_list = [i for i in range(0, 512)]
        self.assist_pq_index_gm = self.tik_instance.Tensor("int32", [512], name="assist_pq_index_gm",
                                                           scope=tik.scope_gm,
                                                           init_value=pq_index_list)
        self.assist_add_gm = self.tik_instance.Tensor("int32", [64], name="assist_add_gm",
                                                      scope=tik.scope_gm,
                                                      init_value=add_index_init_list)
        self.input_gm_tuple = (self.ivf_gm, self.ivf_base_offset_gm, self.ivf_cur_offset_gm,
                               self.ivf_cur_count_gm, self.cur_adc_index_gm, self.adc_tables_gm,)
        self.output_gm_tuple = (self.pq_distance_gm, self.pq_index_gm,)
        # init ub
        self.ivf_cur_process_ub = None
        self.tmp_scalar_ub = None
        self.adc_tables_ub = None
        self.pq_distance_ub = None
        self.pq_index_ub = None

        self.do_num_per_core = None
        self.core_used_num = None
        self.assist_pq_index_ub = None
        self.tiling_ub = None
        self.do_num_tail_core = None
        self.ivf_cur_process_ub_int32 = None
        self.assist_add_ub = None
        self.vgather_out_ub = None
        self.ivf_repeats_int32 = None
        self.ivf_repeats = None
        self.ivf_cur_process_ub_fp16 = None
        # init scaler for each core
        # nc1 start addr offset for per core
        self.ivf_base_offset_scalar = self.tik_instance.Scalar("int32", name="ivf_base_offset_scalar")
        # h start addr offset for per core
        self.ivf_cur_offset_scalar = self.tik_instance.Scalar("int32", name="ivf_cur_offset_scalar")
        # w start addr offset for per core
        self.ivf_cur_count_scalar = self.tik_instance.Scalar("int32", name="ivf_cur_count_scalar")
        self.cur_adc_index_scalar = self.tik_instance.Scalar("int32", name="cur_adc_index_scalar")
        self.tmp_gm_offset_scalar = self.tik_instance.Scalar("int32", name="tmp_gm_offset_scalar")

    def _tiling_args(self):
        """
        tiling_args
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,),
                                                  name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
        self.core_used_num = self.tik_instance.Scalar("int64", name="core_used_num")
        self.do_num_per_core = self.tik_instance.Scalar("int64", name="do_num_per_core")
        self.do_num_tail_core = self.tik_instance.Scalar("int64", name="do_num_tail_core")
        self.core_used_num.set_as(self.tiling_ub[0])
        self.do_num_per_core.set_as(self.tiling_ub[1])
        self.do_num_tail_core.set_as(self.tiling_ub[2])

    def _function_default(self):
        """
        _function_default, run this
        """
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_idx:
            self._tiling_args()
            with self.tik_instance.if_scope(core_idx < self.core_used_num - 1):
                self._run_one_core(core_idx)
            with self.tik_instance.if_scope(core_idx == self.core_used_num - 1):
                self._run_one_core(core_idx, True)

    def _run_one_core(self, core_idx, is_tail_core=False):
        # ub init
        process_num = self.do_num_per_core
        self.tmp_scalar_ub = self.tik_instance.Tensor(self.ivf_base_offset_dtytpe, [32, ],
                                                      name="tmp_scalar_ub", scope=tik.scope_ubuf)
        self.adc_tables_ub = self.tik_instance.Tensor(self.adc_tables_dtype, [256, 16, 16],
                                                      name="adc_tables_ub", scope=tik.scope_ubuf)
        # get input scalar
        self.tik_instance.data_move(self.tmp_scalar_ub[0:], self.ivf_base_offset_gm, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.tmp_scalar_ub[8:], self.ivf_cur_offset_gm, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.tmp_scalar_ub[16:], self.ivf_cur_count_gm, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.tmp_scalar_ub[24:], self.cur_adc_index_gm, 0, 1, 1, 0, 0)

        self.ivf_base_offset_scalar.set_as(self.tmp_scalar_ub[0])
        self.ivf_cur_offset_scalar.set_as(self.tmp_scalar_ub[8])
        self.ivf_cur_count_scalar.set_as(self.tmp_scalar_ub[16])
        self.cur_adc_index_scalar.set_as(self.tmp_scalar_ub[24])
        # move in adc_tables
        self.tmp_gm_offset_scalar.set_as(self.cur_adc_index_scalar * 65536)
        self.tik_instance.data_move(self.adc_tables_ub, self.adc_tables_gm[self.tmp_gm_offset_scalar],
                                    0, 1, 65536 // self.adc_tables_block_num, 0, 0)
        # move in ivf
        self.tmp_gm_offset_scalar.set_as((self.ivf_base_offset_scalar +
                                          self.ivf_cur_offset_scalar + core_idx * process_num) * 16)
        self.ivf_cur_process_ub = self.tik_instance.Tensor(self.ivf_dtype, [512, 16],
                                                           name="ivf_cur_process_ub", scope=tik.scope_ubuf)
        self.ivf_cur_process_ub_fp16 = self.tik_instance.Tensor("float16", [512, 16],
                                                                name="ivf_cur_process_ub_fp16", scope=tik.scope_ubuf)
        self.ivf_cur_process_ub_int32 = self.tik_instance.Tensor("int32", [512, 16],
                                                                 name="ivf_cur_process_ub_int32", scope=tik.scope_ubuf)
        if is_tail_core:
            self.tik_instance.vector_dup(128, self.ivf_cur_process_ub.reinterpret_cast_to("uint16"), 0,
                                         process_num / 16, 1, 8)
            self.tik_instance.data_move(self.ivf_cur_process_ub, self.ivf_gm[self.tmp_gm_offset_scalar],
                                        0, 1, (self.do_num_tail_core + 1) / 2, 0, 0)
        else:
            # repeat = process_num*16/self.ivf_block_num
            self.tik_instance.data_move(self.ivf_cur_process_ub, self.ivf_gm[self.tmp_gm_offset_scalar],
                                        0, 1, process_num / 2, 0, 0)
        # ivf index reprocess
        self.ivf_repeats = self.tmp_gm_offset_scalar
        self.ivf_repeats.set_as(process_num / 8)
        self.tik_instance.vconv(128, "", self.ivf_cur_process_ub_fp16, self.ivf_cur_process_ub,
                                self.ivf_repeats, 1, 1, 8, 4)
        self.ivf_repeats_int32 = self.tmp_gm_offset_scalar
        self.ivf_repeats_int32.set_as(process_num / 4)
        self.tik_instance.vconv(64, "floor", self.ivf_cur_process_ub_int32, self.ivf_cur_process_ub_fp16,
                                self.ivf_repeats_int32, 1, 1, 8, 4)
        self.tik_instance.vmuls(64, self.ivf_cur_process_ub_int32, self.ivf_cur_process_ub_int32, 256,
                                self.ivf_repeats_int32, 1, 1, 8, 8)
        self.assist_add_ub = self.tik_instance.Tensor("int32", [64], name="adc_tables_ub",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.assist_add_ub, self.assist_add_gm,
                                    0, 1, 8, 0, 0)
        self.tik_instance.vadd(64, self.ivf_cur_process_ub_int32, self.ivf_cur_process_ub_int32,
                               self.assist_add_ub, self.ivf_repeats_int32, 1, 1, 1, 8, 8, 0)
        # *2 because offset Bytes
        self.tik_instance.vmuls(64, self.ivf_cur_process_ub_int32, self.ivf_cur_process_ub_int32, 2,
                                self.ivf_repeats_int32, 1, 1, 8, 8)
        # vgather
        self.vgather_out_ub = self.ivf_cur_process_ub_fp16
        # with self.tik_instance.if_scope(process_num/4 != 0):
        self.tik_instance.vgather(process_num * 16, self.vgather_out_ub, self.adc_tables_ub,
                                  self.ivf_cur_process_ub_int32, 1, 8, 0, 0, "counter")
        # pq_index out
        self.assist_pq_index_ub = self.tik_instance.Tensor("int32", [512, ], name="assist_pq_index_ub",
                                                           scope=tik.scope_ubuf)
        self.pq_index_ub = self.tik_instance.Tensor("int32", [512, ], name="pq_index_ub",
                                                    scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.assist_pq_index_ub, self.assist_pq_index_gm, 0, 1,
                                    process_num / 8, 0, 0)
        self.tik_instance.vector_dup(64, self.pq_index_ub, self.ivf_base_offset_scalar +
                                     self.ivf_cur_offset_scalar, (process_num + 63) / 64, 1, 8)  # all dump
        self.tik_instance.vadd(64, self.pq_index_ub, self.pq_index_ub,
                               self.assist_pq_index_ub, (process_num + 63) / 64, 1, 1, 1, 8, 8, 8)
        # vcgadd
        self.pq_distance_ub = self.tik_instance.Tensor("float16", [512, ],
                                                       name="pq_distance_ub", scope=tik.scope_ubuf)
        self.tik_instance.vcgadd(128, self.pq_distance_ub, self.vgather_out_ub, process_num / 8, 1, 1, 8)
        if is_tail_core:
            self.tik_instance.data_move(self.pq_distance_gm[core_idx * process_num], self.pq_distance_ub, 0, 1,
                                        (self.do_num_tail_core + 15) / 16, 0, 0)
            self.tik_instance.data_move(self.pq_index_gm[core_idx * process_num], self.pq_index_ub, 0, 1,
                                        (self.do_num_tail_core + 7) / 8, 0, 0)
        else:
            self.tik_instance.data_move(self.pq_distance_gm[core_idx * process_num], self.pq_distance_ub, 0, 1,
                                        self.do_num_per_core / 16, 0, 0)
            self.tik_instance.data_move(self.pq_index_gm[core_idx * process_num], self.pq_index_ub, 0, 1,
                                        self.do_num_per_core / 8, 0, 0)

    def scan_pq_codes_operator(self):
        """
        scan_pq_codes_operator
        """
        # main process
        self._function_default()
        # Build CCE
        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.core_nums})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_tuple,
                                   outputs=self.output_gm_tuple,
                                   flowtable=(self.tiling_gm,), config=self.opt_config)

        return self.tik_instance


@register_operator("ScanPQCodes")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def scan_pq_codes(ivf, ivf_base_offset, ivf_cur_offset, ivf_cur_count, cur_adc_index,
                  adc_tables, pq_distance, pq_index, kernel_name="scan_pq_codes"):
    """
    ivf: input, 1D Tensor, dtype uint8.  input

    ivf_base_offset: input,  Scalar, dtype int32
        Offset of range data.
    ivf_cur_offset:input,   Scalar, dtype int32
        The offset position of the first element.
    ivf_cur_count: input,  Scalar, dtype int32
        Number of elements to process each time.
    cur_adc_index: input,  Scalar, dtype int32
        Which adc_tables to use.
    adc_tables：input,  3D Tensor, dtype fp16 and fp32, General shape is (ns, ksub, M).
        Represents the adc_tables to be queried
    pq_distance： output, 1D Tensor
        weight
    pq_index: output, 1D Tensor
        index
    """
    obj = ScanPQCodes(ivf, ivf_base_offset, ivf_cur_offset, ivf_cur_count, cur_adc_index,
                      adc_tables, pq_distance, pq_index, kernel_name)

    return obj.scan_pq_codes_operator()
