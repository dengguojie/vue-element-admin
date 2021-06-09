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

#include "tbe_depthwiseconv_elemwise_fusion_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;

static const string kPatternDepthwiseConv = "DepthwiseConvolution";
static const string kPatternElemwise = "eltwise";
static const string kPatternElemwise1 = "eltwise1";
static const string kPatternElemwise2 = "eltwise2";
static const string kPatternSigmoid = "sigmoid";
static const string kPatternMul = "mul";
static const string kPatternQuant = "quant";
static const string kPatternOtherInput = "otherInput";
static const string kPatternOtherInput1 = "otherInput1";
static const string kPatternOtherInput2 = "otherInput2";
static const string kFusedOpType = "FusedOp";
/*
 * @brief:  define depthwise convolution and relu op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    DepthwiseConvolution-->Elemwise
 *    Elemwise-->DepthwiseConvolution
 *
 * fusion node: depthwiseconv, relu
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> DepthwiseConvElemwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name1 = "TbeDepthwiseConvElemwiseFusionPass";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // define pattern DepthwiseConvolution-->Elemwise
  pattern1
      ->AddOpDesc(kPatternDepthwiseConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternElemwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternDepthwiseConv})
      .SetOutputs(kPatternDepthwiseConv, {kPatternElemwise})
      .SetOutputs(kPatternElemwise, {kPatternQuant});
  patterns.push_back(pattern1);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeElemwiseDepthwiseConvInputTwoFusionPass";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  /* define pattern Elemwise-->DepthwiseConvolution
   *                          /
   *                       input
   */
  pattern2
      ->AddOpDesc(kPatternDepthwiseConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternElemwise1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternElemwise1})
      .SetOutputs(kPatternElemwise1, {kPatternDepthwiseConv})
      .SetOutputs(kPatternOtherInput, {kPatternDepthwiseConv});
  patterns.push_back(pattern2);

  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  string pass_name3 = "TbeElemwiseDepthwiseConvInputThreeFusionPass";
  BufferFusionPattern *pattern3 = new (std::nothrow) BufferFusionPattern(pass_name3);
  FUSION_PASS_CHECK(pattern3 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name3.c_str());
  /* define pattern Elemwise-->DepthwiseConvolution
   *                          /     |
   *                       input   bais
   */
  pattern3
      ->AddOpDesc(kPatternDepthwiseConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternElemwise1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternElemwise1})
      .SetOutputs(kPatternElemwise1, {kPatternDepthwiseConv})
      .SetOutputs(kPatternOtherInput, {kPatternDepthwiseConv})
      .SetOutputs(kPatternOtherInput1, {kPatternDepthwiseConv});

  patterns.push_back(pattern3);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name3.c_str());

  string pass_name4 = "TbeDepthwiseConvElewiseInputFusionPass";
  BufferFusionPattern *pattern4 = new (std::nothrow) BufferFusionPattern(pass_name4);
  FUSION_PASS_CHECK(pattern4 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name4.c_str());
  /* define pattern DepthwiseConvolution--->Mul
  *                                        /
  *                                      input
  */
  pattern4
      ->AddOpDesc(kPatternDepthwiseConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternElemwise2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternDepthwiseConv})
      .SetOutputs(kPatternDepthwiseConv, {kPatternElemwise2})
      .SetOutputs(kPatternOtherInput2, {kPatternElemwise2});
  patterns.push_back(pattern4);

  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name4.c_str());

  string pass_name5 = "TbeDepthwiseConvSigmoidMulFusionPass";
  BufferFusionPattern *pattern5 = new (std::nothrow) BufferFusionPattern(pass_name5);
  FUSION_PASS_CHECK(pattern5 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name5.c_str());
  /* define pattern DepthwiseConvolution--->Sigmoid--->Mul
  *                          |_________________________|
  */
  pattern5
      ->AddOpDesc(kPatternDepthwiseConv, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternSigmoid, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternDepthwiseConv})
      .SetOutputs(kPatternDepthwiseConv, {kPatternSigmoid, kPatternMul}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternSigmoid, {kPatternMul});
  patterns.push_back(pattern5);

  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name5.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status DepthwiseConvElemwiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                          vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin of DepthwiseConvElemwise ub fusion pass!!");
  vector<ge::NodePtr> elem3_node = GetMatchedNodesByDescName(kPatternSigmoid, mapping);
  vector<ge::NodePtr> elem2_node = GetMatchedNodesByDescName(kPatternElemwise2, mapping);
  vector<ge::NodePtr> elem1_node = GetMatchedNodesByDescName(kPatternElemwise1, mapping);
  vector<ge::NodePtr> elem_node = GetMatchedNodesByDescName(kPatternElemwise, mapping);
  if (!elem_node.empty()) {
    if (elem_node[0]->GetType() != "LeakyRelu" && elem_node[0]->GetType() != "Relu6") {
      OP_LOGD(kFusedOpType.c_str(),
              "The optype of node[%s] should be LeakyRelu or Relu6,"
          "but actually is [%s], no need to do fusion.",
          elem_node[0]->GetName().c_str(), elem_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!elem1_node.empty()) {
    if (elem1_node[0]->GetType() != "LeakyRelu") {
      OP_LOGD(kFusedOpType.c_str(),
              "The optype of node[%s] should be LeakyRelu,"
          "but actually is [%s], no need to do fusion.",
          elem1_node[0]->GetName().c_str(), elem1_node[0]->GetType().c_str());
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
    OP_LOGD(kFusedOpType.c_str(), "DepthwiseConvElemwise + mul ub fusion!");
  }
  if (!elem3_node.empty()) {
    if (elem3_node[0]->GetType() != "Sigmoid") {
      OP_LOGD(kFusedOpType.c_str(),
              "The optype of node[%s] should be Mul,"
          "but actually is [%s], no need to do fusion.",
          elem3_node[0]->GetName().c_str(), elem3_node[0]->GetType().c_str());
      return SUCCESS;
    }
    OP_LOGD(kFusedOpType.c_str(), "DepthwiseConvSigmoidMul + mul ub fusion!");
  }
  fusion_nodes = GetMatchedNodes(mapping);
  OP_LOGD(kFusedOpType.c_str(), "End of DepthwiseConvElemwise ub fusion pass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDepthwiseConvElemwiseFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            DepthwiseConvElemwiseFusionPass);
}  // namespace fe
