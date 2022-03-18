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

#include "tbe_conv2d_backprop_elemwise_fusion_pass.h"

#include <string>
#include <vector>

#include "anchor_util.h"
#include "graph/types.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
using std::vector;
static const char kPatternDx[] = "conv2dBackpropInput";
static const char kPatternEltwise[] = "eltwise";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternEltwise1[] = "eltwise1";
static const char kPatternOtherInput1[] = "otherInput1";
static const string kFusedOpType = "FusedOp";

/*
 * @brief:  define dx and two inputs elemwise op fusion pattern
 *
 *   Pattern:
 *   conv2d_backprop_input-->elemwise1-->elemwise2-->OpTypeAny
 *              other_input1--/          /
 *                        other_input2--/
 * fusion node: conv2d_backprop_input, elemwise1, elemwise2
 *
 *   Pattern1:
 *   conv2d_backprop_input-->elemwise-->OpTypeAny
 *                other_input--/
 * fusion node: conv2d_backprop_input, elemwise
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> Conv2DBackpropElemwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name = "TbeConv2dBackpropInputTwoElemwiseFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());

  // define pattern rules
  pattern->AddOpDesc(kPatternDx, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_MAX)
      .AddOpDesc(kPatternEltwise1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternDx})
      .SetOutputs(kPatternDx, {kPatternEltwise})
      .SetOutputs(kPatternOtherInput, {kPatternEltwise})
      .SetOutputs(kPatternEltwise, {kPatternEltwise1}, 1, true)
      .SetOutputs(kPatternOtherInput1, {kPatternEltwise1});

  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name1 = "TbeConv2dBackpropInputElemwiseFusion";
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);

  // define pattern1 rules
  pattern1->AddOpDesc(kPatternDx, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternDx})
      .SetOutputs(kPatternDx, {kPatternEltwise})
      .SetOutputs(kPatternOtherInput, {kPatternEltwise});

  patterns.push_back(pattern1);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status Conv2DBackpropElemwiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                        vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do Conv2dBackpropInputElemwiseFusion!");
  vector<ge::NodePtr> dx_nodes = GetMatchedNodesByDescName(kPatternDx, mapping);
  vector<ge::NodePtr> elem1_node = GetMatchedNodesByDescName(kPatternEltwise1, mapping);
  vector<ge::NodePtr> elem_node = GetMatchedNodesByDescName(kPatternEltwise, mapping);
  FUSION_PASS_CHECK(dx_nodes.empty(), OP_LOGD(kFusedOpType.c_str(), "dx node is no matched"), return SUCCESS);

  FUSION_PASS_CHECK(dx_nodes[0]->GetOpDesc()->MutableOutputDesc(0)->GetDataType() == DT_FLOAT,
                    OP_LOGD(kFusedOpType.c_str(), "fusion with fp32, may be slower"), return SUCCESS);
  if (elem_node.empty()) {
    OP_LOGD(kFusedOpType.c_str(), "Elemwise node not matched.");
    return SUCCESS;
  }
  if (!elem1_node.empty()) {
    auto input0desc = GetCurrNodeInputDesc(elem_node[0], 0);
    auto elem_node_input_desc_size = elem_node[0]->GetInDataNodes().size();
    if (elem_node_input_desc_size != 2 && elem_node_input_desc_size != 3) {
      OP_LOGD(kFusedOpType.c_str(), "The optype of node[%s] should have  2 or 3 inputs, but actually is [%d].",
              elem_node[0]->GetType().c_str(), elem_node_input_desc_size);
      return SUCCESS;
    }
    auto input0_dims = input0desc->GetShape().GetDims();
    auto input0_dtype = input0desc->GetDataType();
    bool check_dtype = true;
    bool check_dim = true;
    for (size_t idx = 1; idx < elem_node_input_desc_size; idx++) {
      auto curr_input_desc = GetCurrNodeInputDesc(elem_node[0], idx);
      FUSION_PASS_CHECK((curr_input_desc == nullptr), OP_LOGE(kFusedOpType.c_str(), "inputDesc%lu is null", idx),
                        return SUCCESS);
      check_dtype = check_dtype && (input0_dtype == curr_input_desc->GetDataType());
      check_dim = check_dim && (input0_dims == curr_input_desc->GetShape().GetDims());
    }
    bool check_elemwise = elem1_node[0]->GetType() == "ReluGradV2" &&
                          (elem_node[0]->GetType() == "AddN" || elem_node[0]->GetType() == "Add") && check_dtype &&
                          check_dim;
    if (!check_elemwise) {
      OP_LOGD(kFusedOpType.c_str(),
              "Node [%s] op type [%s] and node [%s] op type [%s], check dim result = [%d], check dtype result = [%d], "
              "can not be fused",
              elem_node[0]->GetName().c_str(), elem_node[0]->GetType().c_str(), elem1_node[0]->GetName().c_str(),
              elem1_node[0]->GetType().c_str(), check_dim, check_dtype);
      return SUCCESS;
    }
  } else {
    if (elem_node[0]->GetType() != "ReluGradV2" && elem_node[0]->GetType() != "Add") {
      OP_LOGD(kFusedOpType.c_str(),
              "The optype of node[%s] should be ReluGradV2/Add,"
              "but actually is [%s], no need to do fusion.",
              elem_node[0]->GetName().c_str(), elem_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  fusion_nodes = GetMatchedNodes(mapping);
  OP_LOGD(kFusedOpType.c_str(), "End to do Conv2dBackpropInputElemwiseFusion!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2DBackpropElemwiseFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            Conv2DBackpropElemwiseFusionPass);
}  // namespace fe
