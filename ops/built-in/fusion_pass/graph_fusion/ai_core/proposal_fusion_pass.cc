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
 * \file proposal_fusion_pass.cpp
 * \brief
 */
#include "proposal_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
#include "common/debug/log.h"
#include "fp16_t.hpp"

namespace fe {
static const float FLOAT_NUM_ZERO = 0;
static const uint16_t UINT_NUM_ZERO = 0;
static const char PATTERN_PROPOSAL[] = "Proposal";
static const char PROPOSAL[] = "Proposal";

// define ROUND(x) ((int)((x) + (float)0.5))
int Round(float x) {
    return static_cast<int>(x + static_cast<float>(0.5));
}

void Anchor2WHCtrXY(const vector<float>& anchor, vector<float>& w_h_ctrxy) {
  GE_CHK_BOOL_EXEC((4 == anchor.size()), return, "Input vector's size is not 4.");
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);
  w_h_ctrxy.push_back(w);
  w_h_ctrxy.push_back(h);
  w_h_ctrxy.push_back(x_ctr);
  w_h_ctrxy.push_back(y_ctr);
}

void WHCtrXY2Anchor(const vector<float>& w_h_ctrxy, vector<float>& anchor) {
  GE_CHK_BOOL_EXEC((4 == w_h_ctrxy.size()), return, "Input vector's size is not 4.");
  anchor.push_back(w_h_ctrxy[2] - 0.5 * (w_h_ctrxy[0] - 1));
  anchor.push_back(w_h_ctrxy[3] - 0.5 * (w_h_ctrxy[1] - 1));
  anchor.push_back(w_h_ctrxy[2] + 0.5 * (w_h_ctrxy[0] - 1));
  anchor.push_back(w_h_ctrxy[3] + 0.5 * (w_h_ctrxy[1] - 1));
}

void GenerateAnchors(const vector<float>& anchor_scale, const vector<float>& anchor_ratio, const float anchor_base_size,
                     vector<float>& anchor_boxes) {
  vector<float> base_anchor;
  base_anchor.push_back(0);
  base_anchor.push_back(0);
  base_anchor.push_back(anchor_base_size - 1);
  base_anchor.push_back(anchor_base_size - 1);
  vector<float> ratio_anchors;

  vector<float> w_h_ctrxy;
  Anchor2WHCtrXY(base_anchor, w_h_ctrxy);
  GE_CHK_BOOL_EXEC((2 <= w_h_ctrxy.size()), return, "Vector's size less than 2.");

  float size = w_h_ctrxy[0] * w_h_ctrxy[1];
  int ratio_num = anchor_ratio.size();
  for (int i = 0; i < ratio_num; i++) {
    if (fabsf(anchor_ratio[i]) < 1e-6) {
      continue;
    }
    float ratio = size / anchor_ratio[i];
    float ws_ori = static_cast<float>(Round(sqrt(ratio)));
    float hs_ori = static_cast<float>(Round(ws_ori * anchor_ratio[i]));

    int scale_num = anchor_scale.size();
    for (int j = 0; j < scale_num; j++) {
      float ws = ws_ori * anchor_scale[j];
      float hs = hs_ori * anchor_scale[j];
      vector<float> w_h_ctrxy_tmp;
      w_h_ctrxy_tmp.push_back(ws);
      w_h_ctrxy_tmp.push_back(hs);
      w_h_ctrxy_tmp.push_back(w_h_ctrxy[2]);
      w_h_ctrxy_tmp.push_back(w_h_ctrxy[3]);
      WHCtrXY2Anchor(w_h_ctrxy_tmp, anchor_boxes);
    }
  }
}

void ProposalFusionPass::GenerateShifts(int height, int width, float feat_stride, vector<float>& shifts) {
  // vector<float> shift_x;
  // vector<float> shift_y;
  // vector<float> shifts_tmp;
  float* shift_x = new float[height * width];
  float* shift_y = new float[height * width];
  float* shifts_tmp = new float[4 * height * width];
  int i = 0;
  int j = 0;
  int k = 0;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass GenerateShifts, height = %d, width = %d", height, width);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      shift_x[k] = j * feat_stride;
      shift_y[k] = i * feat_stride;
      k++;
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass GenerateShifts1");
  k = 0;
  for (i = 0; i < 2; i++) {
    for (j = 0; j < height * width; j++) {
      shifts_tmp[k] = shift_x[j];
      k++;
    }
    for (j = 0; j < height * width; j++) {
      shifts_tmp[k] = shift_y[j];
      k++;
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass GenerateShifts2");
  for (i = 0; i < height * width; i++) {
    for (j = 0; j < 4; j++) {
      shifts.push_back(shifts_tmp[i + j * height * width]);
    }
  }

  delete[] shift_x;
  delete[] shift_y;
  delete[] shifts_tmp;
}

Status ProposalFusionPass::GenerateAnchorsFp16(uint16_t* output1, ge::NodePtr proposalVNode) {
  ge::GeTensorDesc proposalInputTensor = proposalVNode->GetOpDesc()->GetInputDesc(1);
  int batch = proposalInputTensor.GetShape().GetDim(0);
  int channel = proposalInputTensor.GetShape().GetDim(1);
  int height = proposalInputTensor.GetShape().GetDim(2);
  int width = proposalInputTensor.GetShape().GetDim(3);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass GenerateAnchorsFp16");

  if (PatternFusionUtil::IsUnknownShape(channel) ||
      PatternFusionUtil::IsUnknownShape(height) ||
      PatternFusionUtil::IsUnknownShape(width)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ProposalFusionPass cannot be applied for unknown shape.");
    return FAILED;
  }

  // get anchor_base_size and feat_stride
  float anchor_base_size = 0;
  ge::AttrUtils::GetFloat(proposalVNode->GetOpDesc(), "base_size", anchor_base_size);
  float feat_stride = 0;
  ge::AttrUtils::GetFloat(proposalVNode->GetOpDesc(), "feat_stride", feat_stride);

  // get anchor_scale and anchor_ratio
  vector<float> scale_list;
  ge::AttrUtils::GetListFloat(proposalVNode->GetOpDesc(), "scale", scale_list);

  vector<float> ratio_list;
  ge::AttrUtils::GetListFloat(proposalVNode->GetOpDesc(), "ratio", ratio_list);

  GE_CHECK_POSITIVE_SIZE_RANGE(scale_list.size());
  GE_CHECK_POSITIVE_SIZE_RANGE(ratio_list.size());

  vector<float> anchor_scale, anchor_ratio;
  for (uint32_t i = 0; i < scale_list.size(); ++i) {
    anchor_scale.push_back(scale_list[i]);
  }
  for (uint32_t i = 0; i < ratio_list.size(); ++i) {
    anchor_ratio.push_back(ratio_list[i]);
  }

  if ((int)anchor_scale.size() * (int)anchor_ratio.size() * 4 != channel) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "Proposal input channel is invalid, scale.size*ratio.size*4 must equal channel,"
            "scale.size:%lu, ratio.size:%lu, channel:%d",
            anchor_scale.size(), anchor_ratio.size(), channel);
    return FAILED;
  }

  // generate the original anchor boxes, there are anchor_scale.size() * anchor_ratio.size() boxes
  vector<float> anchor_boxes;
  GenerateAnchors(anchor_scale, anchor_ratio, anchor_base_size, anchor_boxes);

  vector<float> shifts;
  GenerateShifts(height, width, feat_stride, shifts);

  int anchor_num = anchor_scale.size() * anchor_ratio.size();
  for (int batch_index = 0; batch_index < batch; batch_index++) {
    int offset = batch_index * channel * height * width;
    for (int i = 0; i < height * width; i++) {
      for (int j = 0; j < anchor_num; j++) {
        fp16_t t;
        t.val = 0;
        t = anchor_boxes[4 * j] + shifts[4 * i];
        output1[offset + j * 4 * height * width + i] = t.val;

        t = anchor_boxes[4 * j + 1] + shifts[4 * i + 1];
        output1[offset + j * 4 * height * width + height * width + i] = t.val;

        t = anchor_boxes[4 * j + 2] + shifts[4 * i + 2];
        output1[offset + j * 4 * height * width + 2 * height * width + i] = t.val;

        t = anchor_boxes[4 * j + 3] + shifts[4 * i + 3];
        output1[offset + j * 4 * height * width + 3 * height * width + i] = t.val;
      }
    }
  }

  return SUCCESS;
}
vector<FusionPattern*> ProposalFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // proposal->proposal_d
  // define DiagFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ProposalFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_PROPOSAL, {PROPOSAL}).SetOutput(PATTERN_PROPOSAL);

  patterns.push_back(pattern);

  return patterns;
}

Status ProposalFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into ProposalFusionPass");
  // proposal node
  ge::NodePtr proposalVNode = GetNodeFromMapping(PATTERN_PROPOSAL, mapping);
  FUSION_PASS_CHECK(proposalVNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "proposalVNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of proposal
  ge::OpDescPtr proposalDesc = proposalVNode->GetOpDesc();
  FUSION_PASS_CHECK(proposalDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "proposalVNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // get the input desc of the entance of proposal node to differentiate between const and var
  ge::GeTensorDesc proposalInputTensor = proposalVNode->GetOpDesc()->GetInputDesc(1);

  // get the shape info
  ge::GeShape proposalInputShape = proposalInputTensor.GetShape();

  // get the data type
  DataType dataType = proposalInputTensor.GetDataType();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass, dataType = %d\n", dataType);

  // multiplies of dims
  int64_t dimNums = proposalInputShape.GetShapeSize();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass, dimNums = %d\n", dimNums);

  ge::GeTensorPtr assitPtr = nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into ProposalFusionPass DT_FLOAT16");
  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);

  Status ret = NnSet(dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

  ret = GenerateAnchorsFp16(inputAssit.get(), proposalVNode);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateAnchorsFp32 failed."), return NOT_CHANGED);

  // define the shape of auxiliary matrix
  ge::GeShape assitShape = proposalInputShape;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDesc.SetShape(assitShape);
  tensorDesc.SetOriginShape(assitShape);
  tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);

  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(uint16_t))),
                          assitPtr = nullptr;
                          return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(proposalVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(proposalVNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType("Const");
  proposalDesc->SetType("ProposalD");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProposalFusionPass pass handle success!!!!");
  return SUCCESS;
}
REGISTER_PASS("ProposalFusionPass", BUILT_IN_GRAPH_PASS, ProposalFusionPass);
}  // namespace fe
