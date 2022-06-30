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

#ifndef FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_
#define FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_

#include <atomic>
#include "graph/op_desc.h"

namespace fe {
class ScopeAllocator {
 public:
  static ScopeAllocator& Instance();

  ScopeAllocator(const ScopeAllocator& in) = delete;
  ScopeAllocator& operator = (const ScopeAllocator& in) = delete;

  int64_t AllocateScopeId();
  int64_t GetCurrentScopeId() const;
  int64_t AllocateNegScopeId();
  int64_t GetCurrentNegScopeId() const;

  static bool HasScopeAttr(ge::ConstOpDescPtr op_desc);
  static bool GetScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &scope_id);
  static bool SetScopeAttr(const ge::OpDescPtr op_desc, const int64_t &scope_id);
  static bool HasL1ScopeAttr(ge::ConstOpDescPtr op_desc);
  static bool GetL1ScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &scope_id);
  static bool SetL1ScopeAttr(const ge::OpDescPtr op_desc, const int64_t &scope_id);

 private:
  ScopeAllocator();
  virtual ~ScopeAllocator();

  std::atomic<int64_t> scope_id_;
  std::atomic<int64_t> neg_scope_id_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_
