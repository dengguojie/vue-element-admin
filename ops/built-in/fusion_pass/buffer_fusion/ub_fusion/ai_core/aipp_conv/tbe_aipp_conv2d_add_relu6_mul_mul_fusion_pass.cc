/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file tbe_aipp_conv2d_add_relu6_mul_mul_fusion_pass.cc
 * \brief tbe aipp + conv2d + add + relu6 + mul + mul ops fusion pattern
 */
#include "tbe_aipp_conv2d_add_relu6_mul_mul_fusion_pass.h"
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
static const char kPatternAdd[] = "add";
static const char kPatternRelu6[] = "relu6";
static const char kPatternMul1[] = "mul1";
static const char kPatternMul2[] = "mul2";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherInput2[] = "otherInput2";

vector<BufferFusionPattern*> AippConv2dAddRelu6MulMulFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pattern_name1 = "TbeAippConv2dAddRelu6MulMulFusionPattern1";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name1.c_str());

  /* define pattern  aipp -->  conv2d   -->     add  -->  relu6  -->   mul1   -->    mul2
  *                              |  otherinput1--/           otherInput2/             /
  *                              |---------------------------------------------------/
  */
  pattern1->AddOpDesc(kPatternAipp, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({kPatternAipp})
          .SetOutputs(kPatternAipp, {kPatternConv})
          .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI, true)
          .SetOutputs(kPatternOtherInput1, {kPatternAdd})
          .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
          .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
          .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE)
          .SetOutputs(kPatternOtherInput2, {kPatternMul2});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name1.c_str());

  return patterns;
}

Status AippConv2dAddRelu6MulMulFusionPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do TbeAippConv2dAddRelu6MulMulFusionPass.");
  std::vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  std::vector<ge::NodePtr> aipp_nodes = GetMatchedNodesByDescName(kPatternAipp, mapping);

  auto kPatternAdd_nodes = GetMatchedNodesByDescName(kPatternAdd, mapping);
  auto kPatternRelu6_nodes = GetMatchedNodesByDescName(kPatternRelu6, mapping);
  auto kPatternMul1_nodes = GetMatchedNodesByDescName(kPatternMul1, mapping);
  auto kPatternMul2_nodes = GetMatchedNodesByDescName(kPatternMul2, mapping);

  std::string input_format = "";

  for (auto aipp_node : aipp_nodes) {
    string aipp_config_str = "";
    FUSION_PASS_CHECK(aipp_node->GetOpDesc() == nullptr, OP_LOGD(fused_op_type_.c_str(), "get desc failed."),
                      return FAILED);
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
  
  FUSION_PASS_CHECK(kPatternAdd_nodes.empty() || kPatternAdd_nodes[0]->GetType() != "Add",
                    OP_LOGD(fused_op_type_.c_str(), "Add not support ub fusion"),
                    return SUCCESS);
  FUSION_PASS_CHECK(kPatternRelu6_nodes.empty() || kPatternRelu6_nodes[0]->GetType() != "Relu6",
                    OP_LOGD(fused_op_type_.c_str(), "Relu6 not support ub fusion"),
                    return SUCCESS);
  FUSION_PASS_CHECK(kPatternMul1_nodes.empty() || kPatternMul1_nodes[0]->GetType() != "Mul",
                    OP_LOGD(fused_op_type_.c_str(), "Mul1 not support ub fusion"),
                    return SUCCESS);
  FUSION_PASS_CHECK(kPatternMul2_nodes.empty() || kPatternMul2_nodes[0]->GetType() != "Mul",
                    OP_LOGD(fused_op_type_.c_str(), "Mul2 not support ub fusion"),
                    return SUCCESS);

  TbeAippFusionRule::SetSplitInfo(conv_nodes, fusion_nodes, false, L1FUSION_BASIC);

  OP_LOGD(fused_op_type_.c_str(), "End to do TbeAippConv2dAddRelu6MulMulFusionPass.");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeAippConv2dAddRelu6MulMulFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, AippConv2dAddRelu6MulMulFusionPass);
}  // namespace fe
