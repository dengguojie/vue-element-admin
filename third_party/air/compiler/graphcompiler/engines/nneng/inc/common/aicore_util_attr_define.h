/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_AICORE_UTIL_ATTR_DEFINE_H_
#define FUSION_ENGINE_INC_COMMON_AICORE_UTIL_ATTR_DEFINE_H_

#include <string>

namespace fe {
// sgt attr
const std::string NON_TAIL_WORKSPACE_SIZE = "_non_tail_workspace_size";

const std::string ATTR_NAME_THREAD_CUBE_VECTOR_CORE_TYPE = "_thread_cube_vector_core_type";

const std::string ATTR_NAME_THREAD_TBE_KERNEL_SIZE = "_thread_tbeKernelSize";

const std::string ATTR_NAME_THREAD_TVM_CACHE_READ_MODE = "thread_tvm_cache_read_mode";

const std::string kThreadTaskRadio = "_thread_task_ratio";

const std::string kThreadModeInArgsFirstField = "_thread_mode_in_args_first_field";

const std::string kContextId = "_context_id";

const std::string kPrefetchEnableBm = "_prefetch_enable_bm";

const std::string kInvalidateBm = "_invalidate_bm";

const std::string kWriteBackBm = "_write_back_bm";

const std::string kAutoCtxIdList = "_context_id_list";

const std::string SCOPE_ID_ATTR = "fusion_scope";

const std::string L1_SCOPE_ID_ATTR = "_l1_fusion_scope";

const std::string FAILED_L1_SCOPE_ID_ATTR = "_failed_l1_fusion_scope";

const std::string FE_IMPLY_TYPE = "_fe_imply_type";

const std::string PARENT_OP_TYPE = "parentOpType";

const std::string ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR = "task_l2_fusion_info_extend_content";

const std::string ATTR_DATA_DUMP_REF = "_datadump_ref";

const std::string ATTR_NAME_L2_FUSION_EXTEND_PTR = "l2_fusion_extend_content";

const std::string ATTR_NAME_UNKNOWN_SHAPE = "_unknown_shape";

const std::string ATTR_NAME_IS_UNKNOWN_GRAPH = "_fe_is_unknown_graph";

const std::string ATTR_NAME_IS_UNKNOWN_SHAPE_OP = "_fe_is_unknown_shape_op";

const std::string ATTR_NAME_OWNER_GRAPH_IS_UNKNOWN = "OwnerGraphIsUnknown";

const std::string ATTR_NAME_TBE_KERNEL_SIZE = "_tbeKernelSize";

const std::string  ATTR_NAME_ALIAS_ENGINE_NAME = "_alias_engine_name";
/* This is not same with the attr engine type.
 * It reveals the op is cube operator or vector operator(implementation related).
 * Firstly, it will be set by op information library and secondly
 * it will be set after pre-compile (then using this attr to set compilation params)
 * Finnaly, after compilation, it will be set by the compilation result. */
const std::string ATTR_NAME_CUBE_VECTOR_CORE_TYPE = "_cube_vector_core_type";

const std::string kSgtCubeVectorCoreType = "_sgt_cube_vector_core_type";

const std::string kTaskRadio = "_task_ratio";

const std::string kModeInArgsFirstField = "_mode_in_args_first_field";

const std::string kForceFp32ToFp16 = "_force_fp32_to_fp16";

const std::string ATTR_NAME_KERNEL_LIST_FIRST_NAME = "_kernel_list_first_name";

const std::string OP_SLICE_INFO = "_op_slice_info";

const std::string FUSION_OP_SLICE_INFO = "_fusion_op_slice_info";

const std::string NEED_RE_PRECOMPILE = "need_re_precompile";

const std::string CORE_TYPE_VALUE = "core_type_value";

// set the attr by L2 Fusion
const std::string ATTR_NAME_LX_FUSION_PASS = "lx_fusion_pass";

const std::string ATTR_NAME_SUPPORT_DYNAMIC_SHAPE = "support_dynamicshape";

const std::string ATTR_NAME_IS_OP_DYNAMIC_IMPL = "_is_op_dynamic_impl";

const std::string kThreadScopeId = "_thread_scope_id";

const std::string kThreadMode = "_thread_mode";

const std::string kTypeFFTSPlus = "_ffts_plus";

const std::string kNoMemReuse = "_no_mem_reuse";

const std::string kNoStreamSplit = "_no_stream_split";

const std::string kAttrEngineType = "_engine_type";

const std::string kLxNoReuseMemFlag = "no_reuse_mem_flag";

const std::string kKBHit = "KBHit";   // knowledge bank

const std::string kReusedParam = "_param_reused";

const std::string kRelationReusedParam = "_param_reused_relation";

const std::string kCmoPrefetch = "Prefetch";

const std::string kCmoInvalid = "Invalid";

const std::string kCmoBarrier = "Barrier";

const std::string kCmoWriteback = "Writeback";

const std::string kOpExtattrNameCmo = "cmo_";

const std::string kAttrExtraParams = "_extra_params";

const std::string kOpShapeOrRangeUnsupport = "_shape_or_range_not_support";

const std::string kGlobalworkspaceBytes = "globalworkspace_spec_workspace_bytes";

const std::string kFusionOpBuildOptions = "fusion_op_build_options";

const std::string ATTR_NAME_CONCAT_C_OPTIMIZE = "concat_c_optimize";

const std::string kGlobalworkspaceSize = "_globalworkspace_size";

const std::string kGlobalworkspaceType = "_globalworkspace_type";

const std::string kAtomicCleanAloneMode = "0";

const std::string kAtomicCleanPolicy = "atomic_clean_policy";
} // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_ATTR_DEFINE_H_
