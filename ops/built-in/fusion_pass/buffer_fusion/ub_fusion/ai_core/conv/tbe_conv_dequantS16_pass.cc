/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv2d_dequantS16_pass.cpp
 * \brief  tbe conv2d + ascend_dequants16 ops fusion pattern
 */
#include "tbe_conv_dequantS16_pass.h"
#include <string>
#include <vector>
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const string kPatternConv = "conv2d";
static const string kPatternDequantS16 = "dequants16";
static const string kPatternOtherInput = "otherInput";
static const string kPatternOtherInput1 = "otherInput1";
static const string kPatternRequantS16 = "requants16";
static const string kPatternOtherInput2 = "otherInput2";
static const string kPatternOtherInput3 = "otherInput3";
static const string kOpTypeReadSelect = "ReadSelect";
/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequants16 --> requants16
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> ConvDequantS16FusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeConvDequantS16RequantS16Fusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // conv2d --> dequants16 --> requants16
  pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequantS16, {OP_PATTERN_DEQUANTS16}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRequantS16, {OP_PATTERN_REQUANTS16}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequantS16})
      .SetOutputs(kPatternOtherInput, {kPatternDequantS16})
      .SetOutputs(kPatternOtherInput1, {kPatternDequantS16})
      .SetOutputs(kPatternDequantS16, {kPatternRequantS16})
      .SetOutputs(kPatternOtherInput2, {kPatternRequantS16})
      .SetOutputs(kPatternOtherInput3, {kPatternRequantS16});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name1 = "TbeConvDequantS16Fusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // conv2d --> dequantS16
  pattern1->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequantS16, {OP_PATTERN_DEQUANTS16}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequantS16})
      .SetOutputs(kPatternOtherInput, {kPatternDequantS16})
      .SetOutputs(kPatternOtherInput1, {kPatternDequantS16});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  return patterns;
}

/*
 *          conv2d
 *            |
 *       dequants16        other_input(s16)
 *           \      const     /
 *            \       |     /    -->
 *                requants16            ->memory_reuse
 *                  /   \        -->
 *                s8    s16
 */
Status ConvDequantS16FusionPass::GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do TbeConvDequantS16FusionPass!");
  fusion_nodes = GetMatchedNodes(mapping);
  auto req_matched = GetMatchedNodesByDescName(kPatternRequantS16, mapping);
  if (req_matched.size() == 1) {
    uint32_t in_pre = 0;
    std::string deq_name;
    auto conv_matched = GetMatchedNodesByDescName(kPatternConv, mapping);
    in_pre += conv_matched.at(0)->GetInDataNodes().size() - 1;
    auto deq_matched = GetMatchedNodesByDescName(kPatternDequantS16, mapping);
    in_pre += deq_matched.at(0)->GetInDataNodes().size() - 1;
    deq_name = deq_matched.at(0)->GetName();
    // pre request check
    auto req_s16_node = req_matched.at(0);
    auto all_in_node = req_s16_node->GetInDataNodes();
    OpDescPtr req_s16_desc = req_s16_node->GetOpDesc();

    for (auto node_ptr : req_s16_node->GetInAllNodes()) {
      if (node_ptr->GetType() == kOpTypeReadSelect) {
        fusion_nodes.push_back(node_ptr);
      }
    }

    uint32_t in_pos = 0;
    OP_LOGD(fused_op_type_.c_str(), "dequants16 node name: %s", deq_name.c_str());
    if (all_in_node.at(0)->GetName() == deq_name) {
      in_pos = 2;
    }
    in_pre += in_pos == 0 ? 1 : 2;
    OP_LOGD(fused_op_type_.c_str(), "get reuse input over, fuse index is:: %d, single index is %d",
            int(in_pre), int(in_pos));
    auto input_out = req_s16_node->GetInDataAnchor(in_pos)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(input_out == nullptr,
                      OP_LOGD(fused_op_type_.c_str(), "node %s input is null", req_s16_node->GetName().c_str()),
                      return SUCCESS);
    size_t peer_inputs = input_out->GetPeerInDataAnchors().size();
    if (peer_inputs > 1) {
      OP_LOGD(fused_op_type_.c_str(), "memory reuse only support requants16 input single-refer scene");
      return SUCCESS;
    }
    auto all_out_desc = req_s16_desc->GetAllOutputsDesc();
    if (all_out_desc.size() < 2) {
      OP_LOGD(fused_op_type_.c_str(), "memory reuse only support requants16 double-out scene");
      return SUCCESS;
    }
    // bind output reuse tensor desc with input
    int out_pos = 0;
    for (auto out_desc : all_out_desc) {
      if (out_desc.GetDataType() == DT_INT16) {
        TensorUtils::SetReuseInput(out_desc, true);
        TensorUtils::SetReuseInputIndex(out_desc, in_pre);
        req_s16_desc->UpdateOutputDesc(out_pos, out_desc);
        OP_LOGD(fused_op_type_.c_str(),
                "set reuse tags over, output position is %d, index is: %d", out_pos, int(in_pre));
        break;
      }
      out_pos++;
    }
  }
  OP_LOGD(fused_op_type_.c_str(), "End to do TbeConvDequantS16FusionPass!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvDequantS16FusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvDequantS16FusionPass);
}  // namespace fe
