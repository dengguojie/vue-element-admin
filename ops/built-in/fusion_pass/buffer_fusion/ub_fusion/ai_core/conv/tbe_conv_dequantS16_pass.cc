/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include <memory>
#include <string>
#include <vector>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {

static const string kPatternConv = "conv2d";
static const string kPatternDequantS16 = "dequants16";
static const string kPatternOtherInput = "otherInput";
static const string kPatternOtherInput1 = "otherInput1";
static const string kPatternRequantS16 = "requants16";
static const string kPatternOtherInput2 = "otherInput2";
static const string kPatternOtherInput3 = "otherInput3";
static const string kOpTypeReadSelect = "ReadSelect";
static const string fused_op_type_ = "FusedOp";

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
  pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternDequantS16, {OP_PATTERN_DEQUANTS16}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternRequantS16, {OP_PATTERN_REQUANTS16}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
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
  pattern1->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternDequantS16, {OP_PATTERN_DEQUANTS16}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequantS16})
      .SetOutputs(kPatternOtherInput, {kPatternDequantS16})
      .SetOutputs(kPatternOtherInput1, {kPatternDequantS16});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  return patterns;
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
  if (!req_matched.empty()) {
    size_t in_pre = 0;
    std::string deq_name;
    auto conv_matched = GetMatchedNodesByDescName(kPatternConv, mapping);
    auto deq_matched = GetMatchedNodesByDescName(kPatternDequantS16, mapping);
    FUSION_PASS_CHECK((conv_matched.empty() || deq_matched.empty()),
                      OP_LOGD(fused_op_type_.c_str(), "get node failed."),
                      return SUCCESS);
    in_pre += conv_matched[0]->GetInDataNodes().size() - 1;
    in_pre += deq_matched[0]->GetInDataNodes().size() - 1;
    deq_name = deq_matched[0]->GetName();
    // pre request check
    auto &req_s16_node = req_matched[0];
    auto all_in_node = req_s16_node->GetInDataNodes();
    FUSION_PASS_CHECK(all_in_node.empty(),
                      OP_LOGD(fused_op_type_.c_str(), "get node failed."),
                      return SUCCESS);
    OpDescPtr req_s16_desc = req_s16_node->GetOpDesc();
    FUSION_PASS_CHECK(req_s16_desc == nullptr,
                      OP_LOGD(fused_op_type_.c_str(), "get desc failed."),
                      return SUCCESS);

    for (auto node_ptr : req_s16_node->GetInAllNodes()) {
      if (node_ptr != nullptr && node_ptr->GetType() == kOpTypeReadSelect) {
        fusion_nodes.push_back(node_ptr);
      }
    }

    uint32_t in_pos = 0;
    OP_LOGD(fused_op_type_.c_str(), "dequants16 node name: %s", deq_name.c_str());
    in_pos = all_in_node.at(0)->GetName() == deq_name ? 2 : 0;
    in_pre += in_pos == 0 ? 1 : 2;
    OP_LOGD(fused_op_type_.c_str(), "get reuse input over, fuse index is: %zu, single index is: %u", in_pre, in_pos);
    FUSION_PASS_CHECK(req_s16_node->GetInDataAnchor(in_pos)  == nullptr,
                      OP_LOGD(fused_op_type_.c_str(), "get anchor failed"),
                      return SUCCESS);
    auto input_out = req_s16_node->GetInDataAnchor(in_pos)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(input_out == nullptr,
                      OP_LOGD(fused_op_type_.c_str(), "node %s input is null", req_s16_node->GetName().c_str()),
                      return SUCCESS);
    size_t peer_inputs = input_out->GetPeerInDataAnchors().size();
    FUSION_PASS_CHECK(peer_inputs > 1,
                      OP_LOGD(fused_op_type_.c_str(), "memory reuse only support requants16 input single-refer scene"),
                      return SUCCESS);
    FUSION_PASS_CHECK(req_s16_desc->GetOutputsSize() < 2,
                      OP_LOGD(fused_op_type_.c_str(), "memory reuse only support requants16 double-out scene"),
                      return SUCCESS);
    // bind output reuse tensor desc with input
    for (uint32_t out_pos = 0; out_pos < req_s16_desc->GetOutputsSize(); ++out_pos) {
      auto out_desc = req_s16_desc->MutableOutputDesc(out_pos);
      if(out_desc == nullptr) {
        OP_LOGD(fused_op_type_.c_str(), "out_desc %u is null", out_pos);
        continue;
      }
      if (out_desc->GetDataType() == DT_INT16) {
        if (!IsShapeEqual(req_s16_node, in_pos, out_pos)) {
          OP_LOGD(fused_op_type_.c_str(),
                  "[Node:%s type:%s] input memory size is not equal with output",
                  req_s16_node->GetName().c_str(), req_s16_node->GetType().c_str());
          break;
        }
        // reuse rollback if compile failed
        std::vector<string> roll_back_attrs = {"reuse_input"};
        if (!ge::AttrUtils::SetListStr(req_s16_desc, "_rollback_if_failed", roll_back_attrs)) {
          OP_LOGD(fused_op_type_.c_str(), "set reuse rollback attr failed");
          break;
        }
        TensorUtils::SetReuseInput(*out_desc.get(), true);
        TensorUtils::SetReuseInputIndex(*out_desc.get(), in_pre);
        OP_LOGD(fused_op_type_.c_str(),
                "set reuse tags over, output position is %u, index is: %zu", out_pos, in_pre);
        break;
      }
    }
  }
  OP_LOGD(fused_op_type_.c_str(), "End to do TbeConvDequantS16FusionPass!");
  return SUCCESS;
}

Status ConvDequantS16FusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
{
  OP_LOGD(fused_op_type_.c_str(), "start calc slice info.");
  std::unique_ptr<ConvSliceInfoCalBase> pConvSliceInfoCal = nullptr;
  pConvSliceInfoCal.reset(new (std::nothrow) ConvSliceInfoCalBase());
  CONV_RET_IF_SMART_PTR_IS_NULL(pConvSliceInfoCal);
  Status ret = pConvSliceInfoCal->ConvCalcFusionOpSliceInfo(fusion_nodes, op_slice_info, fused_op_type_);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(fused_op_type_.c_str(), "calc fusion op slice info failed."), return FAILED);
  OP_LOGD(fused_op_type_.c_str(), "end calc slice info.");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvDequantS16FusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvDequantS16FusionPass);
}  // namespace fe
