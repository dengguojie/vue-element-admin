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

#include "tbe_depthwiseconv_dequant_fusion_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;
static const char kPatternConv[] = "DepthwiseConvolution";
static const char kPatternDequant[] = "dequant";
static const char kPatternRelu[] = "relu";
static const char kPatternRelu6[] = "relu6";
static const char kPatternQuant[] = "quant";
static const char kPatternRequant[] = "requant";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput2[] = "otherInput2";
static const string kPatternMul = "mul";
static const string kPatternSigmoid = "sigmoid";
static const string kPatternPower1 = "power1";
static const string kPatternPower2 = "power2";
static const string kPatternElemwise = "eltwise";
static const bool kIgnoreInputNum = true;
static const bool kIgnoreOutputNum = true;
static const int kTbeOutputBranchSingle = 1;
static const string kFusedOpType = "FusedOp";
/*
 * @brief:  define convolution and single input op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 * DepthwiseConvolution-->AcendDeQuant
 *
 * fusion node: AcendDeQuant,  DepthwiseConvolution
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> DepthwiseConvDequantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name = "TbeDepthwiseConvDequantReluFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternRelu})
      .SetOutputs(kPatternRelu, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());


  string pass_name2 = "TbeDepthwiseConvDequantMulQuantFusion";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern2->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternMul})
      .SetOutputs(kPatternOtherInput2, {kPatternMul})
      .SetOutputs(kPatternMul, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern2);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  string pass_name3 = "TbeDepthwiseConvDequantMulFusion";
  BufferFusionPattern *pattern3 = new (std::nothrow) BufferFusionPattern(pass_name3);
  FUSION_PASS_CHECK(pattern3 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name3.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern3->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternMul})
      .SetOutputs(kPatternOtherInput2, {kPatternMul})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern3);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name3.c_str());

  string pass_name4 = "TbeDepthwiseConvDequantSigmoidMulFusion";
  BufferFusionPattern *pattern4 = new (std::nothrow) BufferFusionPattern(pass_name4);
  FUSION_PASS_CHECK(pattern4 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name4.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern4->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternSigmoid, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternSigmoid, kPatternMul}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternSigmoid, {kPatternMul})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern4);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name4.c_str());

  string pass_name1 = "TbeDepthwiseConvDequantFusion";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern1->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern1);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name5 = "TbeDepthwiseConvRequantFusion";
  BufferFusionPattern *pattern5 = new (std::nothrow) BufferFusionPattern(pass_name5);
  FUSION_PASS_CHECK(pattern5 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name5.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern5->AddOpDesc(kPatternRequant, {OP_PATTERN_REQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternRequant})
      .SetOutputs(kPatternOtherInput, {kPatternRequant});
  patterns.push_back(pattern5);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name5.c_str());

  string pass_name6 = "TbeDepthwiseConvDequantPowerFusion";
  BufferFusionPattern *pattern6 = new (std::nothrow) BufferFusionPattern(pass_name6);
  FUSION_PASS_CHECK(pattern6 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name6.c_str());
  // define pattern rules Convolution-->AcendDeQuant
  pattern6->AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternPower1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternPower2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternElemwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternPower1, kPatternElemwise}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternPower1, {kPatternRelu6})
      .SetOutputs(kPatternRelu6, {kPatternPower2})
      .SetOutputs(kPatternPower2, {kPatternElemwise})
      .SetOutputs(kPatternElemwise, {kPatternQuant}, kTbeOutputBranchSingle, kIgnoreInputNum, kIgnoreOutputNum)
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern6);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name6.c_str());


  return patterns;
}

void SearchMatchNode(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
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
    }
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status DepthwiseConvDequantFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                         vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do TbeDepthwiseConvDequantFusion!");
  vector<ge::NodePtr> elem2_node = GetMatchedNodesByDescName(kPatternMul, mapping);
  vector<ge::NodePtr> elem3_node = GetMatchedNodesByDescName(kPatternSigmoid, mapping);
  vector<ge::NodePtr> relu_node = GetMatchedNodesByDescName(kPatternRelu, mapping);
  vector<ge::NodePtr> power_node = GetMatchedNodesByDescName(kPatternPower1, mapping);
  vector<ge::NodePtr> eletwise_node = GetMatchedNodesByDescName(kPatternElemwise, mapping);
  if (!relu_node.empty()) {
    if ((relu_node[0]->GetType() != "LeakyRelu") && (relu_node[0]->GetType() != "Relu6") &&
        (relu_node[0]->GetType() != "Relu6D")) {
      OP_LOGD(kFusedOpType.c_str(), 
          "The optype of node[%s] should be LeakyRelu or Relu6 or Relu6D,"
          "but actually is [%s], no need to do fusion.",
          relu_node[0]->GetName().c_str(), relu_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!elem2_node.empty()) {
    if (elem2_node[0]->GetType() != "Mul") {
      OP_LOGD(kFusedOpType.c_str(), 
          "The optype of node[%s] should be Mul,"
          "but actually is [%s], no need to do fusion.",
          elem2_node[0]->GetName().c_str(), elem2_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!elem3_node.empty()) {
    if (elem3_node[0]->GetType() != "Sigmoid") {
      OP_LOGD(kFusedOpType.c_str(), 
          "The optype of node[%s] should be Sigmoid,"
          "but actually is [%s], no need to do fusion.",
          elem3_node[0]->GetName().c_str(), elem3_node[0]->GetType().c_str());
      return SUCCESS;
    }
    OP_LOGD(kFusedOpType.c_str(), "DepthwiseConvElemwise + dequant + sigmoid + mul + quant ub fusion!");
  }
  if (!power_node.empty()) {
    if (power_node[0]->GetType() != "Power") {
      OP_LOGD(kFusedOpType.c_str(), 
          "The optype of node[%s] should be Sigmoid,"
          "but actually is [%s], no need to do fusion.",
          power_node[0]->GetName().c_str(), power_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!eletwise_node.empty()) {
    if (eletwise_node[0]->GetType() != "Eltwise") {
      OP_LOGD(kFusedOpType.c_str(), 
          "The optype of node[%s] should be Sigmoid,"
          "but actually is [%s], no need to do fusion.",
          eletwise_node[0]->GetName().c_str(), eletwise_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  fusion_nodes = GetMatchedNodes(mapping);
  // the output_data can't be fused
  SearchMatchNode(mapping, fusion_nodes);

  OP_LOGD(kFusedOpType.c_str(), "End to do TbeDepthwiseConvDequantFusion!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDepthwiseConvDequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            DepthwiseConvDequantFusionPass);
}  // namespace fe
