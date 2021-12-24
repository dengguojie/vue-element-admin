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
 * \file tbe_vector_core_fusion_base_pass.cc
 * \brief base pass for vector core fusion
 */
#include "tbe_vector_core_fusion_base_pass.h"

#include "common/util/platform_info.h"
#include "op_log.h"

namespace fe {
Status TbeEnableVectorCoreFusionBasePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                 std::vector<ge::NodePtr>& fusionNodes) {
  return SUCCESS;
}

bool TbeEnableVectorCoreFusionBasePass::NeedEnableVectorCore(const Mapping& mapping) {
  return vector_core_count_ > 0 && ai_core_count_ > 0;
}

bool TbeEnableVectorCoreFusionBasePass::InitCoreCount() {
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) !=
      SUCCESS) {
    OP_LOGI("Fail to get platform info.");
    return false;
  }

  ai_core_count_ = platform_info.soc_info.ai_core_cnt;
  vector_core_count_ = platform_info.soc_info.vector_core_cnt;
  OP_LOGI("VectorCoreFusion", "platform info: ai_core_cnt=%u, vector_core_cnt=%u.", ai_core_count_, vector_core_count_);
  return vector_core_count_ > 0 && ai_core_count_ > 0;
}

}  // end namespace fe
