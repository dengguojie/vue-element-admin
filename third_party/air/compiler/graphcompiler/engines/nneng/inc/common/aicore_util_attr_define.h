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
static const std::string NON_TAIL_WORKSPACE_SIZE = "_non_tail_workspace_size";

static const std::string ATTR_NAME_THREAD_CUBE_VECTOR_CORE_TYPE = "_thread_cube_vector_core_type";

static const std::string ATTR_NAME_THREAD_TBE_KERNEL_SIZE = "_thread_tbeKernelSize";

static const std::string ATTR_NAME_THREAD_TVM_CACHE_READ_MODE = "thread_tvm_cache_read_mode";

static const std::string SCOPE_ID_ATTR = "fusion_scope";

static const std::string L1_SCOPE_ID_ATTR = "_l1_fusion_scope";

static const std::string FE_IMPLY_TYPE = "_fe_imply_type";

static const std::string PARENT_OP_TYPE = "parentOpType";

static const std::string ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR = "task_l2_fusion_info_extend_content";

static const std::string ATTR_DATA_DUMP_REF = "_datadump_ref";

static const std::string ATTR_NAME_L2_FUSION_EXTEND_PTR = "l2_fusion_extend_content";

static const std::string ATTR_NAME_UNKNOWN_SHAPE = "_unknown_shape";

static const std::string ATTR_NAME_IS_UNKNOWN_GRAPH = "_fe_is_unknown_graph";

static const std::string ATTR_NAME_IS_UNKNOWN_SHAPE_OP = "_fe_is_unknown_shape_op";

static const std::string ATTR_NAME_TBE_KERNEL_SIZE = "_tbeKernelSize";

static const std::string ATTR_NAME_CUBE_VECTOR_CORE_TYPE = "_cube_vector_core_type";

static const std::string ATTR_NAME_KERNEL_LIST_FIRST_NAME = "_kernel_list_first_name";

static const std::string OP_SLICE_INFO = "_op_slice_info";

static const std::string FUSION_OP_SLICE_INFO = "_fusion_op_slice_info";

static const std::string NEED_RE_PRECOMPILE = "need_re_precompile";

// set the attr by L2 Fusion
static const std::string ATTR_NAME_LX_FUSION_PASS = "lx_fusion_pass";
} // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_ATTR_DEFINE_H_
