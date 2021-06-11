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
 * \file conv2d_dequant_vadd_relu_pass.cpp
 * \brief  tbe conv2d + ascend_dequant + vadd + relu ops fusion pattern
 */
#include "tbe_conv_dequant_vadd_relu_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char kPatternConvolution[] = "convolution";
static const char kPatternDequant[] = "dequant";
static const char kPatternVadd[] = "vadd";
static const char kPatternEltwise[] = "eltwise";
static const char kPatternRelu[] = "relu";
static const char kOpTypePrelu[] = "PRelu";
static const char kPatternLeakyRelu[] = "leakyrelu";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherOutput[] = "otherOutput";
static const string fused_op_type_ = "FusedOp";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequant --> vadd --> relu
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> ConvDequantVaddReluFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pass_name = "TbeConvDequantVaddReluFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // conv2d --> dequant --> vadd --> relu
  pattern->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutput, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternRelu})
      .SetOutputs(kPatternRelu, {kPatternOtherOutput})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name1 = "TbeConvDequantLeakyreluVaddFusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // conv2d --> dequant --> leakyrelu --> vadd
  pattern1->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternLeakyRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternLeakyRelu})
      .SetOutputs(kPatternLeakyRelu, {kPatternVadd})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeConvDequantVaddFusion";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // conv2d --> dequant --> vadd
  pattern2->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternVadd})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ConvDequantVaddReluFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                     vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do TbeConvDequantVaddRelu!");
  bool use_common_rules_flag = true;
  fusion_nodes = GetMatchedNodes(mapping);
  // the output data can not fusion
  for (auto &item : mapping) {
    const BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr && (op_desc->desc_name == kPatternVadd || op_desc->desc_name == kPatternEltwise)) {
      ge::NodePtr node = item.second[0];
      if (node == nullptr) {
        return FAILED;
      }
      if (node->GetType() == kOpTypePrelu) {
        fusion_nodes.clear();
        OP_LOGD(fused_op_type_.c_str(),
                "Elemwise is op %s, type %s, skip fusion.", node->GetName().c_str(), node->GetType().c_str());
        break;
      }
      for (auto in_data_anchor : node->GetAllInDataAnchors()) {
        ge::OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        if (peer_out_anchor == nullptr) {
          continue;
        }
        ge::NodePtr src_node = peer_out_anchor->GetOwnerNode();
        if (src_node == nullptr) {
          return FAILED;
        }
        if (src_node->GetType() == "ReadSelect") {
          use_common_rules_flag = false;
          fusion_nodes.push_back(src_node);
          break;
        }
      }
    }
  }
  if (use_common_rules_flag) {
    fusion_nodes.clear();
  }
  OP_LOGD(fused_op_type_.c_str(), "End to do TbeConvDequantVaddRelu!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvDequantVaddReluFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvDequantVaddReluFusionPass);
}  // namespace fe
