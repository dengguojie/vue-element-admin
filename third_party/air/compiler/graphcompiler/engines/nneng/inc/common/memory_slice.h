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
#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_FFTS_PLUS_TASK_BUILDER_DATA_MEMORY_SLICE_H
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_FFTS_PLUS_TASK_BUILDER_DATA_MEMORY_SLICE_H
#include "common/sgt_slice_type.h"
#include "fe_error_code.h"
#include "fe_log.h"
#include "graph/compute_graph.h"
#include "string_utils.h"
#include "aicore_util_constants.h"

namespace fe {
struct DataContextParam {
  int64_t num_inner;
  int64_t len_inner;
  int64_t stride_inner;
  int64_t num_outter;
  int64_t stride_outter;
  int64_t base_addr_offset;
  uint32_t index;
};

const int kBlockDim = 3;
struct Block {
  int64_t dim[kBlockDim];
  int64_t dim_stride[kBlockDim];
  int64_t offset;
  int64_t count;
  int64_t stride;
};

inline Status CheckInt64MultiOverflow(int64_t a, int64_t b) {
  if (a > 0) {
	if (b > 0) {
	  if (a > (static_cast<int64_t>(INT64_MAX) / b)) {
		return FAILED;
	  }
	}
	else {
	  if (b < (static_cast<int64_t>(INT64_MIN) / a)) {
		return FAILED;
	  }
	}
  }
  else {
	if (b > 0) {
	  if (a < (static_cast<int64_t>(INT64_MIN) / b)) {
		return FAILED;
	  }
	}
	else {
	  if ((a != 0) && (b < (static_cast<int64_t>(INT64_MAX) / a))) {
		return FAILED;
	  }
	}
  }
  return SUCCESS;
}

#define FE_INT64_MULCHECK(a, b)                     \
  if (CheckInt64MultiOverflow((a), (b)) != SUCCESS) { \
    return FAILED;                                  \
  }

#define FE_INT64_ADD_CHECK(a, b)                                              \
  if (CheckInt64MultiOverflow((a), (b)) != SUCCESS) {                          \
    FE_LOGE("Int64 %ld and %ld addition can result in overflow!", (a), (b)); \
    return FAILED;                                                   \
  }
class MemorySlice {
 public:
  MemorySlice();
  ~MemorySlice();

  static Status GenerateManualDataCtxParam(const ge::NodePtr &node, int index, bool is_input, int64_t burst_len,
                                           std::vector<DataContextParam> &data_ctx);

  static Status GenerateAutoDataCtxParam(const ge::NodePtr &node, int index, bool is_input, int64_t burst_len,
                                         std::vector<DataContextParam> &param_nontail_tail);

  static Status GenerateDataCtxParam(ge::GeTensorDescPtr tensor, const std::vector<ffts::DimRange> &slice,
                                     int64_t burst_len, std::vector<DataContextParam> &data_ctx, uint32_t index);
  MemorySlice(const MemorySlice &builder) = delete;
  MemorySlice &operator=(const MemorySlice &builder) = delete;
};
Status GetPeerNodeAndOutIdx(const ge::NodePtr &ori_node, int ori_index, ge::NodePtr &peer_node, int &peer_index);
}  // namespace fe
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_FFTS_PLUS_TASK_BUILDER_DATA_MEMORY_SLICE_H
