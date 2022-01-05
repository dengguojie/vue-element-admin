/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain new_offset_name copy of the License at
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
 * \file deformable_offsets_grad_fusion_pass.cpp
 * \brief add const node
 */
#include "deformable_offsets_grad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {
static const std::string CONSTANTOP = "Constant";
static const char* FUSED_NODE = "DeformableOffsetsGrad";
static const std::string PATTERN_FUSEDNODE = "DeformableOffsetsGrad";

vector<FusionPattern*> DeformableOffsetsGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DeformableOffsetsGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status MakeHelperTensor(const vector<int64_t>& offsets_shape, const vector<int64_t>& kernel_sizes,
                        const vector<int64_t>& strides, const vector<int64_t>& pads, const vector<int64_t>& dilations,
                        float& grid_base) {
  float* helper_tensor = &grid_base;

  int64_t H_OUT = offsets_shape[1];
  int64_t W_OUT = offsets_shape[2];
  int64_t K_H = kernel_sizes[0];
  int64_t K_W = kernel_sizes[1];

  int64_t stride_h = strides[1];
  int64_t stride_w = strides[2];
  int64_t dilation_h = dilations[1];
  int64_t dilation_w = dilations[2];
  int64_t group = offsets_shape[3] / (3 * K_H * K_W);
  int64_t pad_top = pads[0];
  int64_t pad_left = pads[2];
  int64_t h_index = 0;
  int64_t w_index = 0;

  for (int64_t h = 0; h < H_OUT; h++) {
    for (int64_t w = 0; w < W_OUT; w++) {
      for (int64_t g = 0; g < group; g++) {
        for (int64_t k_h = 0; k_h < K_H; k_h++) {
          for (int64_t k_w = 0; k_w < K_W; k_w++) {
            w_index = h * W_OUT * 3 * group * K_H * K_W + w * 3 * group * K_H * K_W + 0 * group * K_H * K_W +
                      g * K_H * K_W + k_h * K_W + k_w;
            h_index = h * W_OUT * 3 * group * K_H * K_W + w * 3 * group * K_H * K_W + 1 * group * K_H * K_W +
                      g * K_H * K_W + k_h * K_W + k_w;
            float w_val = static_cast<float>(w * stride_w - pad_left + k_w * dilation_w);
            float h_val = static_cast<float>(h * stride_h - pad_top + k_h * dilation_h);
            helper_tensor[w_index] = w_val;
            helper_tensor[h_index] = h_val;
          }
        }
      }
    }
  }

  return SUCCESS;
}

Status DeformableOffsetsGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                               std::vector<NodePtr>& newNodes) {
  // get node
  ge::NodePtr deformableOffsetsGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(deformableOffsetsGradNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get DeformableOffsetsGrad Node Failed, fusion failed."),
                    return PARAM_INVALID);
  // get desc
  ge::OpDescPtr deformableOffsetsGradDesc = deformableOffsetsGradNode->GetOpDesc();
  FUSION_PASS_CHECK(deformableOffsetsGradDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DeformableOffsetsGrad's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // get op
  Operator deformableOffsetsGradOp = OpDescUtils::CreateOperatorFromNode(deformableOffsetsGradNode);

  // check op  supported
  FUSION_PASS_CHECK(!CheckOpSupported(deformableOffsetsGradDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);
  // get input shape
  auto x_desc = deformableOffsetsGradDesc->GetInputDesc(1);
  auto offsets_desc = deformableOffsetsGradDesc->GetInputDesc(2);
  vector<int64_t> x_shape = x_desc.GetShape().GetDims();
  vector<int64_t> offsets_shape = offsets_desc.GetShape().GetDims();
  vector<int64_t> helper_shape = offsets_desc.GetShape().GetDims();
  // get attrs
  int64_t dfm_groups;
  FUSION_PASS_CHECK(!AttrUtils::GetInt(deformableOffsetsGradDesc, "deformable_groups", dfm_groups),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "get deformable_groups attr failed."), return PARAM_INVALID);
  vector<int64_t> ksize;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(deformableOffsetsGradDesc, "ksize", ksize),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "get ksize attr failed."), return PARAM_INVALID);
  vector<int64_t> strides;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(deformableOffsetsGradDesc, "strides", strides),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "get strides attr failed."), return PARAM_INVALID);
  vector<int64_t> pads;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(deformableOffsetsGradDesc, "pads", pads),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "get pads attr failed."), return PARAM_INVALID);
  vector<int64_t> dilations;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(deformableOffsetsGradDesc, "dilations", dilations),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "get dilations attr failed."), return PARAM_INVALID);

  Format helperMatrixFormat = x_desc.GetFormat();
  FUSION_PASS_CHECK(
      helperMatrixFormat != FORMAT_NHWC,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The format of input is %d, not NHWC, cannot do fusion.", helperMatrixFormat),
      return NOT_CHANGED);

  helper_shape[0] = 1;
  ge::GeTensorPtr helpMatrixPtr = nullptr;
  int64_t matrixSize = offsets_shape[1] * offsets_shape[2] * offsets_shape[3];
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[matrixSize]());
  auto retMem = memset_s(inputAssit.get(), matrixSize, 0, matrixSize);
  FUSION_PASS_CHECK(retMem != EOK, OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to operate memset_s."), return NOT_CHANGED);
  Status ret = MakeHelperTensor(offsets_shape, ksize, strides, pads, dilations, *inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "MakeHelperTensor failed."), return NOT_CHANGED);

  // define the shape of auxiliary matrix
  ge::GeTensorDesc tensorDesc;
  ge::GeShape helperShape(helper_shape);
  tensorDesc.SetShape(helperShape);
  tensorDesc.SetOriginShape(helperShape);
  tensorDesc.SetFormat(helperMatrixFormat);
  tensorDesc.SetOriginFormat(helperMatrixFormat);
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetOriginDataType(ge::DT_FLOAT);
  FUSION_PASS_MAKE_SHARED((helpMatrixPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), matrixSize * sizeof(float))),
                          helpMatrixPtr = nullptr;
                          return PARAM_INVALID);
  // add const node
  ge::OpDescPtr const_ptr = ge::OpDescUtils::CreateConstOp(helpMatrixPtr);
  ge::NodePtr const_node = graph.AddNode(const_ptr);
  FUSION_PASS_CHECK(const_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to add const node."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(deformableOffsetsGradNode->AddLinkFrom(3, const_node) != ge::GRAPH_SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to link const node with DeformableOffsetsGrad node."),
                    return NOT_CHANGED);
  auto constInputNodes = OpDescUtils::GetConstInputs(deformableOffsetsGradNode);
  NodePtr constInput = nullptr;
  if (constInputNodes.size() != 0) {
    constInput = constInputNodes[0];
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "constInputNodes is null, please check!");
    return NOT_CHANGED;
  }
  constInput->GetOpDesc()->SetType(CONSTANTOP);

  return SUCCESS;
}

REGISTER_PASS("DeformableOffsetsGradFusionPass", BUILT_IN_GRAPH_PASS, DeformableOffsetsGradFusionPass);
}  // namespace fe
