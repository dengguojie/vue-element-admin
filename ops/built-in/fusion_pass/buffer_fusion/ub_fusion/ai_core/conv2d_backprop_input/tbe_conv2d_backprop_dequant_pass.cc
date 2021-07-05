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

#include "tbe_conv2d_backprop_dequant_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;
static const char kPatternConv[] = "convolution";
static const char kPatternDequant[] = "dequant";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternLeakyRelu[] = "leakyrelu";
static const char kPatternPRelu[] = "PRelu";
static const string kFusedOpType = "FusedOp";

/*
 * @brief:  define convolution and single input op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    Convolution-->AcendDeQuant
 *
 * fusion node: AcendDeQuant,  Convolution
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2DBackpropDequantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name = "TbeConv2DBackpropDequantFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternLeakyRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternLeakyRelu})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});

  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeConv2DBackpropDequantFusionPass::GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do TbeConv2DBackpropDequantFusionPass!");
  fusion_nodes = GetMatchedNodes(mapping);
  // the output_data can't be fused
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);

    if (opdesc != item.first->types.end()) {
      for (auto &node : item.second) {
        auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        if (node_ptr != fusion_nodes.end()) {
          fusion_nodes.erase(node_ptr);
        }
      }
    }
    const BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr && op_desc->desc_name == kPatternLeakyRelu) {
      ge::NodePtr node = item.second[0];
      if (node == nullptr) {
        OP_LOGE(kFusedOpType.c_str(), "node is nullptr");
        return PARAM_INVALID;
      }
      if (node->GetType() == kPatternPRelu) {
        fusion_nodes.clear();
        OP_LOGD(kFusedOpType.c_str(), "Eltwise is op %s, type %s, skip fusion.", node->GetName().c_str(), node->GetType().c_str());
        break;
      }
    }
  }

  OP_LOGD(kFusedOpType.c_str(), "End to do TbeConv2DBackpropDequantFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2DBackpropDequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2DBackpropDequantFusionPass);
}  // namespace fe
