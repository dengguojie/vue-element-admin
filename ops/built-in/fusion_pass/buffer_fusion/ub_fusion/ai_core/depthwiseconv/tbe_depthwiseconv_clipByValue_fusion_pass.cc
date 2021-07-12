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

#include "tbe_depthwiseconv_clipByValue_fusion_pass.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include <string>
#include <vector>


namespace fe {
using std::vector;

static const string PATTERN_DEPTHWISECONV = "DepthwiseConvolution";
static const string PATTERN_ELEMWISE = "eltwise";
static const string PATTERN_OTHER_INPUT1 = "otherInput1";
static const string PATTERN_OTHER_INPUT2 = "otherInput2";
/*
 * @brief: define depthwise convolution and relu op fusion pattern
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
vector<BufferFusionPattern *> DepthwiseconvClipByValueFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string passName = "DepthwiseconvClipByValueFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
  FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
  FE_LOGD("Start to define %s pass pattern.", passName.c_str());
  /* define pattern     DepthwiseConvolution -->  Elemwise
   *                                              /     |
   *                                          input1  input2
   */
  pattern
      ->AddOpDesc(PATTERN_DEPTHWISECONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_DEPTHWISECONV, {PATTERN_ELEMWISE})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEMWISE})
      .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELEMWISE});

  patterns.push_back(pattern);
  FE_LOGD("End to define %s pass pattern.", passName.c_str());


  string passName1 = "ClipByValueDepthwiseconvFusionPass";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
  FE_CHECK((pattern1 == nullptr), FE_LOGE("new an object failed."), return patterns);
  FE_LOGD("Start to define %s pass pattern.", passName1.c_str());
  /* define pattern   Elemwise  -->  DepthwiseConvolution
   *                                       /
   *                                     input1
   */
  pattern1
      ->AddOpDesc(PATTERN_DEPTHWISECONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_ELEMWISE})
      .SetOutputs(PATTERN_ELEMWISE, {PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_DEPTHWISECONV});

  patterns.push_back(pattern1);
  FE_LOGD("End to define %s pass pattern.", passName1.c_str());

  string passName2 = "ClipByValueDepthwiseconvFusionTwoInputPass";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(passName2);
  FE_CHECK((pattern2 == nullptr), FE_LOGE("new an object failed."), return patterns);
  FE_LOGD("Start to define %s pass pattern.", passName2.c_str());
  /* define pattern   Elemwise  -->  DepthwiseConvolution
  *                                       /     |
  *                                     input1  bias
  */
  pattern2
      ->AddOpDesc(PATTERN_DEPTHWISECONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_ELEMWISE})
      .SetOutputs(PATTERN_ELEMWISE, {PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_DEPTHWISECONV});
  patterns.push_back(pattern2);
  FE_LOGD("End to define %s pass pattern.", passName2.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status DepthwiseconvClipByValueFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                         vector<ge::NodePtr> &fusionNodes) {
  FE_LOGD("Begin of DepthwiseConvClipByValue ub fusion pass!!");
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEMWISE, mapping);
  vector<ge::NodePtr> depthwiseconv_node = GetMatchedNodesByDescName(PATTERN_DEPTHWISECONV, mapping);
  if (!depthwiseconv_node.empty()) {
    if (depthwiseconv_node[0]->GetType() != "DepthwiseConv2D") {
      FE_LOGI(
        "The optype of node[%s] should be DepthwiseConvolution,"
        "but actually is [%s], no need to do fusion.",
        depthwiseconv_node[0]->GetName().c_str(), depthwiseconv_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!elemNode.empty()) {
    if (elemNode[0]->GetType() != "ClipByValue") {
      FE_LOGI(
        "The optype of node[%s] should be ClipByValue,"
        "but actually is [%s], no need to do fusion.",
        elemNode[0]->GetName().c_str(), elemNode[0]->GetType().c_str());
      return SUCCESS;
    }
  }

  fusionNodes = GetMatchedNodes(mapping);
  FE_LOGD("End of DepthwiseConvElemwise ub fusion pass!");
  return SUCCESS;
  }
  REGISTER_BUFFER_FUSION_PASS("DepthwiseconvClipByValueFusionPass",
                              BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                              DepthwiseconvClipByValueFusionPass);
}  // namespace fe