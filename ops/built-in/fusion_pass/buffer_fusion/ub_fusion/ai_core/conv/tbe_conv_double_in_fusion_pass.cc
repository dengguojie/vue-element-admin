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

#include <memory>
#include "tbe_conv_double_in_fusion_pass.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {
using std::vector;
static const string PATTERN_CONV = "convolution";
static const string PATTERN_ELTWISE1 = "eltwise1";
static const string PATTERN_ELTWISE2 = "eltwise2";
static const string PATTERN_OTHER_INPUT = "otherInput";
static const int NCHW_INDEX_C = 1;
static const int HWCN_INDEX_C = 2;
static const int NHWC_INDEX_C = 3;
static const string FUSED_OP_TYPE = "FusedOp";

vector<BufferFusionPattern*> ConvDoubleInFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "TbeConvElemwiseReluFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_ELTWISE2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .SetHead({PATTERN_CONV})
          .SetOutputs(PATTERN_CONV, {PATTERN_ELTWISE1})
          .SetOutputs(PATTERN_ELTWISE1, {PATTERN_ELTWISE2})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELTWISE1});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pattern_name.c_str());

  return patterns;
}

static Status AddReadSelectFromGraph(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes,
                                     bool &use_common_rules_flag) {
  for (auto &item : mapping) {
    const BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr && op_desc->desc_name == PATTERN_ELTWISE1) {
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
      OP_LOGD(FUSED_OP_TYPE.c_str(), "parse node's dim c is %ld.", c);
      break;
    case ge::FORMAT_HWCN:
      c = shape.GetDim(HWCN_INDEX_C);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "parse node's dim c is %ld.", c);
      break;
    case ge::FORMAT_NHWC:
      c = shape.GetDim(NHWC_INDEX_C);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "parse node's dim c is %ld.", c);
      break;
    default:
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Just support format NCHW, NHWC, HWCN now, but actually is %d.", format);
      return FAILED;
  }

  return SUCCESS;
}

bool PathExistsBetweenTwoNode(ge::NodePtr &from, ge::NodePtr &to) {
  int count = 100;
  ge::NodePtr parent_node;
  auto input_nodes = to->GetInDataNodes();
  if (input_nodes.empty()) {
    return false;
  }

  parent_node = input_nodes.at(0);
  while (parent_node != nullptr && count >= 0) {
    count--;
    if (parent_node == from) {
      return true;
    }
    input_nodes = parent_node->GetInDataNodes();
    if (input_nodes.empty()) {
      return false;
    } else {
      parent_node = input_nodes.at(0);
    }

  }
  return false;
}

Status ConvDoubleInFusionPass::GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do ConvDoubleInFusionPass.");
  bool use_common_rules_flag = true;
  fusion_nodes = GetMatchedNodes(mapping);
  vector<ge::NodePtr> matched_elem_node = GetMatchedNodesByDescName(PATTERN_ELTWISE1, mapping);
  vector<ge::NodePtr> matched_conv_node = GetMatchedNodesByDescName(PATTERN_CONV, mapping);
  vector<ge::NodePtr> conv_nodes;
  if (SUCCESS != AddReadSelectFromGraph(mapping, fusion_nodes, use_common_rules_flag)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Parse Parameter failed.");
    return PARAM_INVALID;
  }
  bool check_node_valid = !matched_elem_node.empty() && !matched_conv_node.empty();
  if (check_node_valid) {
    for (size_t i = 0; i < matched_elem_node[0]->GetAllInDataAnchors().size(); i++) {
      ge::NodePtr conv_node = matched_elem_node[0]->GetAllInDataAnchors().at(i)->GetPeerOutAnchor()->GetOwnerNode();
      if (conv_node->GetType() != "Conv2D") {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s, %s] not match to Conv2D.",
                conv_node->GetName().c_str(), conv_node->GetType().c_str());
        continue;
      }
      conv_nodes.push_back(conv_node);
    }
  }
  if (conv_nodes.size() > 1) {
    int64_t matched_conv_node_c_in = -1;
    ge::GeTensorDesc matched_conv_node_c_in_anchor = matched_conv_node[0]->GetOpDesc()->GetAllInputsDesc().at(0);
    if (GetDimSizes(matched_conv_node_c_in_anchor, matched_conv_node_c_in) != SUCCESS) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s]'s format just support format NCHW, HWCN, NHWC now.",
              matched_conv_node[0]->GetName().c_str());
    } else {
      for (ge::NodePtr each_conv_node : conv_nodes) {
        int64_t conv_node_c_in = -1;
        ge::GeTensorDesc conv_node_c_in_anchor = each_conv_node->GetOpDesc()->GetAllInputsDesc().at(0);
        if (GetDimSizes(conv_node_c_in_anchor, conv_node_c_in) != SUCCESS) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s]'s format just support NCHW, HWCN, NHWC now.",
                  each_conv_node->GetName().c_str());
          continue;
        }
        if (conv_node_c_in > matched_conv_node_c_in) {
          if (!PathExistsBetweenTwoNode(each_conv_node, matched_conv_node[0])) {
            use_common_rules_flag = false;
            EraseNodeFromMapping(mapping, fusion_nodes, OP_PATTERN_CONV);
            fusion_nodes.push_back(each_conv_node);
          }
        }
      }
    }
  }

  if (use_common_rules_flag) {
    fusion_nodes.clear();
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do ConvDoubleInFusionPass.");
  return SUCCESS;
}

Status ConvDoubleInFusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
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

REGISTER_BUFFER_FUSION_PASS("TbeConvDoubleInFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, ConvDoubleInFusionPass);
}  // namespace fe