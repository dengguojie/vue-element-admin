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
 * \file conv2d_dequant_vadd_relu_quant_pass.cpp
 * \brief  tbe conv2d + ascend_dequant + vadd + relu + quant ops fusion pattern
 */
#include "tbe_conv_dequant_vadd_relu_quant_pass.h"
#include <string>
#include <vector>
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

using namespace ge;

namespace fe {

static const char kPatternConvolution[] = "convolution";
static const char kPatternDequant[] = "dequant";
static const char kPatternVadd[] = "vadd";
static const char kPatternEltwise[] = "eltwise";
static const char kPatternRelu[] = "relu";
static const char kPatternLeakyRelu[] = "leakyrelu";
static const char kPatternQuant[] = "quant";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherInput2[] = "otherInput2";
static const char kPatternOtherOutput[] = "otherOutput";
static const int NCHW_INDEX_C = 1;
static const int HWCN_INDEX_C = 2;
static const int NHWC_INDEX_C = 3;
static const string fused_op_type_ = "FusedOp";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequant --> vadd --> relu --> quant
 *    conv2d --> dequant --> vadd --> relu(multi output) --> quant
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> ConvDequantVaddReluQuantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeConvDequantVaddQuantFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // conv2d --> dequant --> vadd --> quant
  pattern->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternEltwise})
      .SetOutputs(kPatternEltwise, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternEltwise});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name1 = "TbeConvDequantVaddReluQuantFusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // conv2d --> dequant --> vadd --> relu --> quant
  pattern1->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternRelu})
      .SetOutputs(kPatternRelu, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeConvDequantVaddMultiOutReluQuantFusion";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // conv2d --> dequant --> vadd -->relu --> quant
  pattern2->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutput, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternVadd})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternRelu})
      .SetOutputs(kPatternRelu, {kPatternQuant, kPatternOtherOutput}, TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  string pass_name3 = "TbeConvDequantLeakyReluVaddQuantFusion";
  BufferFusionPattern* pattern3 = new (std::nothrow) BufferFusionPattern(pass_name3, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern3 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name3.c_str());
  // conv2d --> dequant --> leakyrelu -->vadd --> quant
  pattern3->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternLeakyRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternLeakyRelu})
      .SetOutputs(kPatternLeakyRelu, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern3);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name3.c_str());

  string pass_name4 = "TbeConvDequantLeakyReluVaddMultiOutQuantFusion";
  BufferFusionPattern* pattern4 = new (std::nothrow) BufferFusionPattern(pass_name4, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern4 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name4.c_str());
  // conv2d --> dequant --> leakyrelu -->vadd --> quant
  pattern4->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternLeakyRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutput, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternLeakyRelu})
      .SetOutputs(kPatternLeakyRelu, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternQuant, kPatternOtherOutput}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern4);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name4.c_str());

  string pass_name5 = "TbeConvDequantLeakyreluMultiOutVaddQuantFusion";
  BufferFusionPattern* pattern5 = new (std::nothrow) BufferFusionPattern(pass_name5, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern5 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name5.c_str());
  // conv2d --> dequant --> vadd -->relu --> quant
  pattern5->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternLeakyRelu, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutput, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternLeakyRelu, kPatternOtherOutput}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternLeakyRelu, {kPatternQuant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd});
  patterns.push_back(pattern5);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name5.c_str());

  string pass_name6 = "TbeConvDequantMultiOutVaddQuantFusion";
  BufferFusionPattern* pattern6 = new (std::nothrow) BufferFusionPattern(pass_name6, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern6 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name6.c_str());
  // conv2d --> dequant --> vadd --> quant
  pattern6->AddOpDesc(kPatternConvolution, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternVadd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutput, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConvolution})
      .SetOutputs(kPatternConvolution, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternVadd})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternOtherInput1, {kPatternVadd})
      .SetOutputs(kPatternVadd, {kPatternQuant, kPatternOtherOutput}, TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern6);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name6.c_str());

  return patterns;
}

static Status AddReadSelectFromGraph(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes,
                                     bool &use_common_rules_flag) {
  for (auto &item : mapping) {
    const BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr && op_desc->desc_name == kPatternVadd) {
      ge::NodePtr node = item.second[0];
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
  return SUCCESS;
}

static void EraseNodeFromMapping(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes,
                                 const string &matched_pattern) {
  for (auto &item : mapping) {
    if (item.first == nullptr) {
      continue;
    }
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), matched_pattern);
    if (opdesc != item.first->types.end()) {
      for (auto &node : item.second) {
        auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        fusion_nodes.erase(node_ptr);
      }
    }
  }
}

static Status GetDimSizes(const ge::GeTensorDesc &conv_input_tensor, int64_t &c) {
  ge::GeShape shape = conv_input_tensor.GetOriginShape();
  ge::Format format = conv_input_tensor.GetOriginFormat();

  switch (format) {
    case ge::FORMAT_NCHW:
      c = shape.GetDim(NCHW_INDEX_C);
      OP_LOGD(fused_op_type_.c_str(), "parse node's dim c is %ld", c);
      break;
    case ge::FORMAT_HWCN:
      c = shape.GetDim(HWCN_INDEX_C);
      OP_LOGD(fused_op_type_.c_str(), "parse node's dim c is %ld", c);
      break;
    case ge::FORMAT_NHWC:
      c = shape.GetDim(NHWC_INDEX_C);
      OP_LOGD(fused_op_type_.c_str(), "parse node's dim c is %ld", c);
      break;
    default:
      OP_LOGD(fused_op_type_.c_str(), "just support format NCHW, HWCN, NHWC now, but actually is %d.", format);
      return FAILED;
  }
  return SUCCESS;
}

/*
 * @brief: check memery reuse input output tensor shape is equal or not
 */
static bool IsShapeEqual(const NodePtr a_node, uint32_t id_in, uint32_t id_out) {
  if (a_node == nullptr) {
    OP_LOGD(fused_op_type_.c_str(), "input node is nullptr");
    return false;
  }
  OpDescPtr a_desc = a_node->GetOpDesc();
  if (a_desc == nullptr) {
    OP_LOGD(fused_op_type_.c_str(), "get node desc failed");
    return false;
  }
  if (id_in >= a_desc->GetInputsSize()) {
    OP_LOGD(fused_op_type_.c_str(), "index is illegal: input id_in(%u) is out of range(%u)",
            id_in, a_desc->GetInputsSize());
    return false;
  }
  if (id_out >= a_desc->GetOutputsSize()) {
    OP_LOGD(fused_op_type_.c_str(), "index is illegal: input id_out(%u) is out of range(%u)",
            id_out, a_desc->GetOutputsSize());
    return false;
  }
  int64_t in_size = 0;
  int64_t out_size = 0;
  ge::graphStatus graph_status = ge::TensorUtils::GetTensorMemorySizeInBytes(a_desc->GetInputDesc(id_in), in_size);
  if (graph_status != ge::GRAPH_SUCCESS) {
    OP_LOGD(fused_op_type_.c_str(), "get input tensor memory size in bytes failed");
    return false;
  }
  graph_status = ge::TensorUtils::GetTensorMemorySizeInBytes(a_desc->GetOutputDesc(id_out), out_size);
  if (graph_status != ge::GRAPH_SUCCESS) {
    OP_LOGD(fused_op_type_.c_str(), "get output tensor memory size in bytes failed");
    return false;
  }
  OP_LOGD(fused_op_type_.c_str(), "node[%s]'s reuse input %u size is %lld, output %u size is %lld",
          a_node->GetName().c_str(), id_in, in_size, id_out, out_size);
  return in_size == out_size;
}

void ConvDequantVaddReluQuantFusionPass::SetMemoryReuse(const BufferFusionMapping &mapping) {
  size_t in_pre = 0;
  std::string deq_name;
  uint32_t in_pos = 0;
  vector<ge::NodePtr> conv_node = GetMatchedNodesByDescName(kPatternConvolution, mapping);
  vector<ge::NodePtr> dequant_node = GetMatchedNodesByDescName(kPatternDequant, mapping);
  vector<ge::NodePtr> vadd_node = GetMatchedNodesByDescName(kPatternVadd, mapping);
  vector<ge::NodePtr> relu_node = GetMatchedNodesByDescName(kPatternRelu, mapping);
  bool invalid = conv_node.empty() || dequant_node.empty() || vadd_node.empty() || relu_node.empty();
  if (invalid) {
    return;
  }
  auto conv_inputs = conv_node[0]->GetInDataNodes();
  in_pre += conv_inputs.size() - 1;
  auto dequant_inputs = dequant_node[0]->GetInDataNodes();
  in_pre += dequant_inputs.size() - 1;
  deq_name = dequant_node[0]->GetName();
  auto all_in_node = vadd_node[0]->GetInDataNodes();
  invalid = all_in_node.empty() || all_in_node.at(0) == nullptr;
  if (invalid) {
    OP_LOGD(fused_op_type_.c_str(), "get node failed");
    return;
  }
  in_pre += 1;
  OP_LOGD(fused_op_type_.c_str(), "dequant node name: %s", deq_name.c_str());
  in_pos = all_in_node.at(0)->GetName() == deq_name ? 1 : 0;
  if (!IsShapeEqual(vadd_node[0], in_pos, 0)) {
    OP_LOGD(fused_op_type_.c_str(), "[Node:%s type:%s] input memory size is not equal with output",
            vadd_node[0]->GetName().c_str(), vadd_node[0]->GetType().c_str());
    return;
  }
  invalid = vadd_node[0]->GetInDataAnchor(in_pos) == nullptr ||
            vadd_node[0]->GetInDataAnchor(in_pos)->GetPeerOutAnchor() == nullptr;
  if (invalid) {
    OP_LOGD(fused_op_type_.c_str(), "get anchor failed");
    return;
  }
  auto input_out = vadd_node[0]->GetInDataAnchor(in_pos)->GetPeerOutAnchor();
  size_t peer_inputs = input_out->GetPeerInDataAnchors().size();
  if (peer_inputs > 1) {
    OP_LOGD(fused_op_type_.c_str(),
            "[Node:%s type:%s] has %zu output, but supports only single-output and single-refer in memory reuse.",
            vadd_node[0]->GetName().c_str(), vadd_node[0]->GetType().c_str(), peer_inputs);
    return;
  }
  // pre request check
  if (relu_node[0]->GetOutDataAnchor(0) == nullptr) {
    OP_LOGD(fused_op_type_.c_str(), "get anchor failed");
    return;
  }
  if (relu_node[0]->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() <= 1) {
    OP_LOGD(fused_op_type_.c_str(), "[Node:%s type:%s] has %zu output, but supports only single-output and multi-refer in memory reuse.",
            relu_node[0]->GetName().c_str(), relu_node[0]->GetType().c_str(), relu_node[0]->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    return;
  }
  OP_LOGD(fused_op_type_.c_str(), "get reuse input over, fuse index is: %zu, single index is %u", in_pre, in_pos);
  OpDescPtr relu_desc = relu_node[0]->GetOpDesc();
  if (relu_desc == nullptr) {
    OP_LOGD(fused_op_type_.c_str(), "get desc failed");
    return;
  }
  auto out_desc = relu_desc->MutableOutputDesc(0);
  if (out_desc == nullptr) {
    OP_LOGD(fused_op_type_.c_str(), "out_desc 0 is null");
    return;
  }
  // reuse rollback if compile failed
  std::vector<string> roll_back_attrs = {"reuse_input"};
  if (!ge::AttrUtils::SetListStr(relu_desc, "_rollback_if_failed", roll_back_attrs)) {
    OP_LOGD(fused_op_type_.c_str(), "set reuse rollback attr failed");
    return;
  }
  // bind output reuse tensor desc with input
  TensorUtils::SetReuseInput(*out_desc.get(), true);
  TensorUtils::SetReuseInputIndex(*out_desc.get(), in_pre);
  OP_LOGD(fused_op_type_.c_str(), "set reuse tags over, output position is %d, index is: %zu", 0, in_pre);
}

static void GetQuantAndConvNodes(vector<ge::NodePtr> &matched_elem_node, vector<ge::NodePtr> &dequant_nodes,
                                 vector<ge::NodePtr> &conv_nodes) {
  for (unsigned int i = 0; i < matched_elem_node[0]->GetAllInDataAnchors().size(); i++) {
    auto peerAnchor = matched_elem_node[0]->GetAllInDataAnchors().at(i)->GetPeerOutAnchor();
    if (peerAnchor == nullptr) {
      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s first peer in anchor is null", matched_elem_node[0]->GetName().c_str());
      continue;
    }
    ge::NodePtr dequant_node = peerAnchor->GetOwnerNode();
    if (dequant_node->GetType() != "AscendDequant") {
      OP_LOGD(fused_op_type_.c_str(), "Node[%s] type[%s] not matched to AscendDequant.",
              dequant_node->GetName().c_str(), dequant_node->GetType().c_str());
      continue;
    }
    if (dequant_node->GetOutAllNodes().size() > 1) {
      OP_LOGD(fused_op_type_.c_str(), "Node[%s] type[%s] has %zu output nodes, no need to do fusion.",
              dequant_node->GetName().c_str(), dequant_node->GetType().c_str(), dequant_node->GetOutAllNodes().size());
      continue;
    }
    dequant_nodes.push_back(dequant_node);
    auto firstPeerAnchor = dequant_node->GetAllInDataAnchors().at(0)->GetPeerOutAnchor();
    if (firstPeerAnchor == nullptr) {
      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s first peer in anchor is null", dequant_node->GetName().c_str());
      continue;
    }
    ge::NodePtr conv_node = firstPeerAnchor->GetOwnerNode();
    if (conv_node->GetType() != "Conv2D") {
      OP_LOGD(fused_op_type_.c_str(), "Node[%s] type[%s] not match to Conv2D.", conv_node->GetName().c_str(),
              conv_node->GetType().c_str());
      continue;
    }
    conv_nodes.push_back(conv_node);
  }
}

static void UpdateFusionNodes(vector<ge::NodePtr> &conv_nodes, vector<ge::NodePtr> &dequant_nodes,
                              vector<ge::NodePtr> &matched_conv_node, const BufferFusionMapping &mapping,
                              vector<ge::NodePtr> &fusion_nodes, bool &use_common_rules_flag) {
  if (conv_nodes.size() > 1 && (dequant_nodes.size() == conv_nodes.size())) {
    int64_t matched_conv_node_c_in = -1;
    ge::GeTensorDesc matched_conv_node_c_in_anchor = matched_conv_node[0]->GetOpDesc()->GetAllInputsDesc().at(0);
    if (GetDimSizes(matched_conv_node_c_in_anchor, matched_conv_node_c_in) != SUCCESS) {
      OP_LOGD(fused_op_type_.c_str(),
              "node[%s]'s format just support format NCHW, HWCN, NHWC now.", matched_conv_node[0]->GetName().c_str());
    } else {
      for (ge::NodePtr each_conv_node : conv_nodes) {
        int64_t conv_node_c_in = -1;
        ge::GeTensorDesc conv_node_c_in_anchor = each_conv_node->GetOpDesc()->GetAllInputsDesc().at(0);
        if (GetDimSizes(conv_node_c_in_anchor, conv_node_c_in) != SUCCESS) {
          OP_LOGD(fused_op_type_.c_str(),
                  "node[%s]'s format just support format NCHW, HWCN, NHWC now.", each_conv_node->GetName().c_str());
          continue;
        }
        if (conv_node_c_in > matched_conv_node_c_in) {
          use_common_rules_flag = false;
          EraseNodeFromMapping(mapping, fusion_nodes, OP_PATTERN_CONV);
          fusion_nodes.push_back(each_conv_node);
          EraseNodeFromMapping(mapping, fusion_nodes, OP_PATTERN_DEQUANT);
          auto firstPeerAnchor = each_conv_node->GetAllOutAnchors().at(0)->GetFirstPeerAnchor();
          if (firstPeerAnchor == nullptr) {
            OP_LOGD(fused_op_type_.c_str(), "node[%s]'s first peer out anchor is null", each_conv_node->GetName().c_str());
            continue;
          }
          ge::NodePtr each_dequant_node = firstPeerAnchor->GetOwnerNode();
          fusion_nodes.push_back(each_dequant_node);
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
Status ConvDequantVaddReluQuantFusionPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do ConvDequantVaddReluQuant!");
  bool use_common_rules_flag = true;
  vector<ge::NodePtr> eltwise_node = GetMatchedNodesByDescName(kPatternEltwise, mapping);
  if (!eltwise_node.empty()) {
    if ((eltwise_node[0]->GetType() != "Eltwise") && (eltwise_node[0]->GetType() != "Add")) {
      OP_LOGD(fused_op_type_.c_str(), "The optype of node[%s] should be Eltwise or Add, but actually is [%s], no need to do fusion.",
              eltwise_node[0]->GetName().c_str(), eltwise_node[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  auto relu_node = GetMatchedNodesByDescName(kPatternRelu, mapping);
  if (relu_node.size() == 1) {
    use_common_rules_flag = false;
    SetMemoryReuse(mapping);
  }
  fusion_nodes = GetMatchedNodes(mapping);
  // the output_data can't be fused
  EraseNodeFromMapping(mapping, fusion_nodes, TBE_PATTERN_OUTPUT_NODE);
  if (SUCCESS != AddReadSelectFromGraph(mapping, fusion_nodes, use_common_rules_flag)) {
    OP_LOGE(fused_op_type_.c_str(), "Find ReadSelect node from graph failed.");
    return FAILED;
  }
  vector<ge::NodePtr> matched_quant_node = GetMatchedNodesByDescName(kPatternQuant, mapping);
  vector<ge::NodePtr> matched_elem_node = GetMatchedNodesByDescName(kPatternVadd, mapping);
  vector<ge::NodePtr> matched_conv_node = GetMatchedNodesByDescName(kPatternConvolution, mapping);
  vector<ge::NodePtr> dequant_nodes;
  vector<ge::NodePtr> conv_nodes;
  bool check_node_valid = !matched_quant_node.empty() && !matched_elem_node.empty() && !matched_conv_node.empty();
  if (check_node_valid) {
    GetQuantAndConvNodes(matched_elem_node, dequant_nodes, conv_nodes);
  }
  UpdateFusionNodes(conv_nodes, dequant_nodes, matched_conv_node, mapping, fusion_nodes, use_common_rules_flag);

  if (use_common_rules_flag) {
    fusion_nodes.clear();
  }
  OP_LOGD(fused_op_type_.c_str(), "End to do TbeConvDequantVaddReluQuant!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvDequantVaddReluQuantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvDequantVaddReluQuantFusionPass);
}  // namespace fe
