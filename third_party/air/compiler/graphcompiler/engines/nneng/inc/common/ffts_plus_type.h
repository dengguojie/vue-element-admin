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

#include "common/sgt_slice_type.h"

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FFTS_PLUS_TYPE_H
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FFTS_PLUS_TYPE_H
#include <cstdint>

namespace fe {

const uint32_t kManualMode = 0;
const uint32_t kAutoMode = 1;

const uint32_t kDefaultWindowSize = 4;
const uint32_t kMaxPersistNum = 8;
const size_t kMaxCacheOperationSize = 64;
extern const std::unordered_set<std::string> CONTROL_OP_V2_TYPE;

inline bool OpIsAutoThread(ffts::ThreadSliceMapPtr slice_info_ptr)
{
  if ((slice_info_ptr != nullptr) && (slice_info_ptr->thread_mode == 1)
      && (!slice_info_ptr->input_tensor_slice.empty() || !slice_info_ptr->input_tensor_slice.empty())) {
    return true;
  }
  return false;
}

void inline SetBitOne(const uint32_t pos, uint32_t &bm) {
  bm |= (0x1 << pos);
}

void inline SetBitOne(const uint32_t pos, uint64_t &bm) {
  bm |= (0x1 << pos);
}

struct TickCacheMap {
  std::vector<int32_t> src_out_of_graph_input_index;
  std::map<int32_t, uint8_t> input_cache_table;
  std::map<int32_t, uint8_t> output_cache_table;
};

enum class CACHE_OPERATION {
  PREFETCH = 0,
  INVALIDATE = 1,
  WRITE_BACK = 2,
  CACHE_OPERATION_BOTTOM = 3
};


}

#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FFTS_PLUS_TYPE_H
