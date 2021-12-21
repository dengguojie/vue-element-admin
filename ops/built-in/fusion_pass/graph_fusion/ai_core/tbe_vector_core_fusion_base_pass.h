/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

/*!
 * \file tbe_vector_core_fusion_base_pass.h
 * \brief base pass for vector core fusion
 */
#ifndef CANN_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_VECTOR_CORE_FUSION_BASE_PASS_H_
#define CANN_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_VECTOR_CORE_FUSION_BASE_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

#include <vector>

namespace fe {
class TbeEnableVectorCoreFusionBasePass: public fe::PatternFusionBasePass {
 protected:
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& fusionNodes) override;

  virtual bool NeedEnableVectorCore(const Mapping& mapping);
  virtual bool InitCoreCount();

  uint32_t GetAiCoreCount() const {
    return ai_core_count_;
  }

  uint32_t GetVectorCoreCount() const {
    return vector_core_count_;
  }

  uint32_t GetAiCoreLoops(uint32_t count) const {
    return GetCoreLoops(count, ai_core_count_);
  }

  uint32_t GetAllCoreLoops(uint32_t count) const {
    return GetCoreLoops(count, ai_core_count_ + vector_core_count_);
  }

 private:
  static uint32_t GetCoreLoops(uint32_t count, uint32_t core_count) {
    if (core_count == 0) {
      return core_count;
    }

    return (count + core_count - 1) / core_count;
  }

 private:
  uint32_t ai_core_count_ = 1;
  uint32_t vector_core_count_ = 0;
};

} // end namespace fe

#endif // CANN_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_VECTOR_CORE_FUSION_BASE_PASS_H_
