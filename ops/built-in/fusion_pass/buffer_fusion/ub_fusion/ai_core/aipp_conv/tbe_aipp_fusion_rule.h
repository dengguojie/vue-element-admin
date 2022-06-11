/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file tbe_aipp_fusion_rule.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_FUSION_RULE_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_FUSION_RULE_H_
#include <string>
#include "graph/node.h"
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

class TbeAippFusionRule {
 public:
  static bool CheckAippConvStridehValidation(const ge::NodePtr conv_node);
  static bool CheckConvload2dNodeValidation(const ge::NodePtr conv_node);
  static bool CheckAippConvEltwiseFusionValidation(const ge::NodePtr conv_node, const string& input_format);
  static bool CheckElemwiseValidation(ge::NodePtr elemwise_node);
  static void SetSplitInfo(std::vector<ge::NodePtr> &conv_nodes, std::vector<ge::NodePtr> &fusion_nodes,
                           const bool &is_deal_c_axis, const OpL1FusionType& aipp_L1_fusion_type);
private:
  static int64_t CalcMinAIPPTbeL1Space(const ge::NodePtr& conv_node);
};

class BroadcastPatternRule {
 public:
  static bool CheckTensorData(const ge::GeTensorDesc tensor_0, const ge::GeTensorDesc tensor_1);
  static bool CheckBroadcastFusionScenario(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes);
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_AIPP_CONV_TBE_AIPP_FUSION_RULE_H_

