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
 * \file batch_multi_class_nms_enable_vector_core_fusion_pass.h
 * \brief batch_multi_class_nms enable vector core fusion pass
 */
#ifndef CANN_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_BATCH_MULTI_CLASS_NMS_ENABLE_AI_CORE_FUSION_PASS_H_
#define CANN_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_BATCH_MULTI_CLASS_NMS_ENABLE_AI_CORE_FUSION_PASS_H_

#include "tbe_vector_core_fusion_base_pass.h"

namespace fe {
class BatchMultiClassNonMaxSuppressionEnableVectorCoreFusionPass : public TbeEnableVectorCoreFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& fusionNodes) override;
  bool NeedEnableVectorCore(const Mapping& mapping) override;
};

} // namespace fe

#endif // CANN_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_BATCH_MULTI_CLASS_NMS_ENABLE_AI_CORE_FUSION_PASS_H_
