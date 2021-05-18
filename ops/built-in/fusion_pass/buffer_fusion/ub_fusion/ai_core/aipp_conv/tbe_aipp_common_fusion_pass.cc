/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file tbe_aipp_conv_fusion_pass.cpp
 * \brief tbe aipp convolution ops fusion pattern
 */
#include "tbe_aipp_common_fusion_pass.h"
#include <math.h>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "tbe_aipp_fusion_rule.h"

namespace fe {
static const char kPatternAipp[] = "aipp";
static const char kPatternConv[] = "convolution";
static const char kPatternElemwise[] = "elemwise";
static const char kPatternQuant[] = "quant";
static const char kPatternStridedWrite[] = "stridedwrite";

vector<BufferFusionPattern*> TbeAippCommonFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pattern_name1 = "TbeAippCommonFusionPattern1";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(kPatternAipp, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternElemwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_MAX)
          .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternStridedWrite, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({kPatternAipp})
          .SetOutputs(kPatternAipp, {kPatternConv})
          .SetOutputs(kPatternConv, {kPatternElemwise}, TBE_OUTPUT_BRANCH_SINGLE, true)
          .SetOutputs(kPatternElemwise, {kPatternQuant}, TBE_OUTPUT_BRANCH_SINGLE, true, true)
          .SetOutputs(kPatternQuant, {kPatternStridedWrite});

  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name1.c_str());

  return patterns;
}

Status TbeAippCommonFusionPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do TbeAippCommonFusionPass.");
  std::vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  std::vector<ge::NodePtr> aipp_nodes = GetMatchedNodesByDescName(kPatternAipp, mapping);
  std::vector<ge::NodePtr> elemwise_nodes = GetMatchedNodesByDescName(kPatternElemwise, mapping);
  std::vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(kPatternQuant, mapping);
  std::vector<ge::NodePtr> strided_write_nodes = GetMatchedNodesByDescName(kPatternStridedWrite, mapping);

  std::string input_format = "";

  for (auto aipp_node : aipp_nodes) {
    string aipp_config_str = "";
    FUSION_PASS_CHECK(!ge::AttrUtils::GetStr(aipp_node->GetOpDesc(), "aipp_config_path", aipp_config_str),
                      OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s aipp_config_path attr not success.",
                              aipp_node->GetName().c_str()),
                      return FAILED);

    nlohmann::json aipp_config_json = nlohmann::json::parse(aipp_config_str);
    FUSION_PASS_CHECK(!aipp_config_json.is_object(),
                      OP_LOGD(fused_op_type_.c_str(), "The aipp_config_str is not an object, aipp_config_str is %s.",
                              aipp_config_str.c_str()),
                      return FAILED);
    input_format = aipp_config_json["input_format"];
    OP_LOGD(fused_op_type_.c_str(), "aipp input_format is %s!", input_format.c_str());
  }

  for (auto conv_node : conv_nodes) {
    if (!TbeAippFusionRule::CheckConvload2dNodeValidation(conv_node)) {
      OP_LOGD(fused_op_type_.c_str(), "Node[%s] not satisfied with fusion condition.", conv_node->GetName().c_str());
      return SUCCESS;
    }
    if (!TbeAippFusionRule::CheckAippConvEltwiseFusionValidation(conv_node, input_format)) {
      OP_LOGD(fused_op_type_.c_str(),
              "The AIPP YUV exceed the L1 buffer, "
              "Node[%s] not satisfied with fusion condition.",
              conv_node->GetName().c_str());
      return SUCCESS;
    }
    if (!TbeAippFusionRule::CheckAippConvStridehValidation(conv_node)) {
      OP_LOGD(fused_op_type_.c_str(),
              "The case is the strideh optim. "
              "Node[%s] not satisfied with fusion condition.",
              conv_node->GetName().c_str());
      return SUCCESS;
    }
  }

  fusion_nodes = GetMatchedNodes(mapping);
  FilterElemwiseNodes(elemwise_nodes, quant_nodes, strided_write_nodes, fusion_nodes);

  if (quant_nodes.empty()) {
    TbeAippFusionRule::SetSplitInfo(conv_nodes, fusion_nodes, false);
  } else {
    TbeAippFusionRule::SetSplitInfo(conv_nodes, fusion_nodes, true);
  }

  OP_LOGD(fused_op_type_.c_str(), "End to do TbeAippCommonFusionPass.");
  return SUCCESS;
}

void TbeAippCommonFusionPass::FilterElemwiseNodes(std::vector<ge::NodePtr> &elemwise_nodes,
                                                  std::vector<ge::NodePtr> &quant_nodes,
                                                  std::vector<ge::NodePtr> &strided_write_nodes,
                                                  std::vector<ge::NodePtr> &fusion_nodes) {
  if (elemwise_nodes.empty()) {
    return;
  }

  vector<ge::NodePtr> non_relu_nodes;
  for (ge::NodePtr &elemwise_node : elemwise_nodes) {
    if (!TbeAippFusionRule::CheckElemwiseValidation(elemwise_node)) {
      OP_LOGD(elemwise_node->GetName().c_str(), "Op type[%s] must be relu, relu6 or leakyrelu.",
              elemwise_node->GetType().c_str());
      non_relu_nodes.push_back(elemwise_node);
    }
  }
  if (non_relu_nodes.empty()) {
    return;
  }

  std::vector<ge::NodePtr> remove_nodes;
  for (ge::NodePtr &non_relu_node : non_relu_nodes) {
    AddRemovingReluNodes(non_relu_node, elemwise_nodes, remove_nodes);
  }

  // quant and strided_write nodes is behind elemwise nodes
  remove_nodes.insert(remove_nodes.end(), quant_nodes.begin(), quant_nodes.end());
  remove_nodes.insert(remove_nodes.end(), strided_write_nodes.begin(), strided_write_nodes.end());

  for (ge::NodePtr &remove_node : remove_nodes) {
    OP_LOGD(remove_node->GetName().c_str(), "This node needs to be removed.");
    auto iter = std::find(fusion_nodes.begin(), fusion_nodes.end(), remove_node);
    if (iter != fusion_nodes.end()) {
      fusion_nodes.erase(iter);
    }
  }
}

void TbeAippCommonFusionPass::AddRemovingReluNodes(ge::NodePtr remove_node,
                                                   const std::vector<ge::NodePtr> &elemwise_nodes,
                                                   std::vector<ge::NodePtr> &remove_nodes) {
  if (remove_node == nullptr) {
    return;
  }
  if (std::find(elemwise_nodes.begin(), elemwise_nodes.end(), remove_node) == elemwise_nodes.end()) {
    return;
  }
  if (std::find(remove_nodes.begin(), remove_nodes.end(), remove_node) != remove_nodes.end()) {
    return;
  }
  remove_nodes.push_back(remove_node);
  auto out_data_nodes = remove_node->GetOutDataNodes();
  if (out_data_nodes.empty()) {
    return;
  }
  AddRemovingReluNodes(out_data_nodes.at(0), elemwise_nodes, remove_nodes);
}

REGISTER_BUFFER_FUSION_PASS("TbeAippCommonFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeAippCommonFusionPass);
}  // namespace fe
