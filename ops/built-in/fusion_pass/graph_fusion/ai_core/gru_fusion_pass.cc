/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file gru_fusion_pass.cpp
 * \brief GRU fusion pass
 *   (CommonGRU --> DynamicGRUV2)
 */
#include "gru_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "CommonGRU";
static const std::string PATTERN_FUSEDNODE = "CommonGRU";

vector<FusionPattern*> GRUFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern *pattern = new(std::nothrow) FusionPattern("GRUFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
          .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status GRUFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) {
  // get the NodePtr of GRU
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null."), return PARAM_INVALID);

  // get the OpDescPtr of GRU
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null."), return PARAM_INVALID);

  auto gruOp = ge::OperatorFactory::CreateOperator(fusedDesc->GetName() + "_splitD_layer", "DynamicGRUV2");
  FUSION_PASS_CHECK(gruOp.IsEmpty(),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "create DynamicGRUV2 operator error"),
                        return FAILED);

  // create DynamicGRUV2 OpDesc
  std::shared_ptr<ge::OpDesc> gruOpDesc = nullptr;
  gruOpDesc = ge::OpDescUtils::GetOpDescFromOperator(gruOp);
  gruOp.BreakConnect();
  FUSION_PASS_CHECK(gruOpDesc == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "gruOpDesc is null, DynamicGRUV2 failed."),
                      return PARAM_INVALID);

  // process x
  GeTensorDesc xInput = fusedDesc->GetInputDesc(0);
  std::vector<int64_t> xInputDims = xInput.GetShape().GetDims();

  GeShape xInputOriginShape(xInputDims);
  xInput.SetOriginShape(xInputOriginShape);
  (void)ProcessNZFormat(xInputDims);
  GeShape xInputShape(xInputDims);
  xInput.Update(xInputShape, ge::FORMAT_FRACTAL_NZ, xInput.GetDataType());
  gruOpDesc->UpdateInputDesc("x", xInput);

  // process weight_input
  GeTensorDesc weightInput = fusedDesc->GetInputDesc(1);
  std::vector<int64_t> weightInputDims = RemoveNumDirectionsDim(weightInput.GetShape().GetDims(), true);

  GeShape weightInputOriginShape(weightInputDims);
  weightInput.SetOriginShape(weightInputOriginShape);
  (void)ProcessZFormat(weightInputDims);
  GeShape weightInputShape(weightInputDims);
  weightInput.Update(weightInputShape, ge::FORMAT_FRACTAL_Z, weightInput.GetDataType());
  gruOpDesc->UpdateInputDesc("weight_input", weightInput);

  // process weight_hidden
  GeTensorDesc weightHidden = fusedDesc->GetInputDesc(2);
  std::vector<int64_t> weightHiddenDims = RemoveNumDirectionsDim(weightHidden.GetShape().GetDims(), true);

  GeShape weightHiddenOriginShape(weightHiddenDims);
  weightHidden.SetOriginShape(weightHiddenOriginShape);
  (void)ProcessZFormat(weightHiddenDims);
  GeShape weightHiddenShape(weightHiddenDims);
  weightHidden.Update(weightHiddenShape, ge::FORMAT_FRACTAL_Z, weightHidden.GetDataType());
  gruOpDesc->UpdateInputDesc("weight_hidden", weightHidden);

  bool hasSeqLength = fusedDesc->MutableInputDesc("sequence_lens") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "hasSeqLength");
  if (hasSeqLength) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasSeqLength");
    gruOpDesc->UpdateInputDesc("seq_length", *fusedDesc->MutableInputDesc("sequence_lens"));
  }

  bool hasInitH = fusedDesc->MutableInputDesc("initial_h") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "init_h");
  if (hasInitH) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasInitH");
    GeTensorDesc initialH = *fusedDesc->MutableInputDesc("initial_h");
    std::vector<int64_t> initialHDims = RemoveNumDirectionsDim(initialH.GetShape().GetDims(), false);

    GeShape initialHOriginShape(initialHDims);
    initialH.SetOriginShape(initialHOriginShape);
    (void)ProcessNZFormat(initialHDims);
    GeShape initialHShape(initialHDims);
    initialH.Update(initialHShape, ge::FORMAT_FRACTAL_NZ, initialH.GetDataType());
    gruOpDesc->UpdateInputDesc("init_h", initialH);
  }

  GeTensorDesc y = fusedDesc->GetOutputDesc(0);
  std::vector<int64_t> yDims = ProcessOutputDim(y.GetShape().GetDims());
  GeShape yOriginShape(yDims);
  y.SetOriginShape(yOriginShape);
  (void)ProcessNZFormat(yDims);
  GeShape yShape(yDims);
  y.Update(yShape, ge::FORMAT_FRACTAL_NZ, y.GetDataType());
  gruOpDesc->UpdateOutputDesc("y", y);

  GeTensorDesc outputH = fusedDesc->GetOutputDesc(1);
  std::vector<int64_t> outputHDims = ProcessOutputDim(outputH.GetShape().GetDims());

  GeShape outputHOriginShape(outputHDims);
  outputH.SetOriginShape(outputHOriginShape);
  (void)ProcessNZFormat(outputHDims);
  GeShape outputHShape(outputHDims);
  outputH.Update(outputHShape, ge::FORMAT_FRACTAL_NZ, outputH.GetDataType());
  gruOpDesc->UpdateOutputDesc("output_h", outputH);

  gruOpDesc->UpdateOutputDesc("update", outputH);
  gruOpDesc->UpdateOutputDesc("reset", outputH);
  gruOpDesc->UpdateOutputDesc("new", outputH);
  gruOpDesc->UpdateOutputDesc("hidden_new", outputH);

  // create a splitD Op
  OpDescPtr splitDesc = nullptr;
  splitDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/DynamicGRUV2_split", "SplitD");
  FUSION_PASS_CHECK(splitDesc == nullptr, 
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "splitD is null, SplitD failed."), return PARAM_INVALID);

  int groups = 2;
  AttrUtils::SetInt(splitDesc, "split_dim", 1);
  AttrUtils::SetInt(splitDesc, "num_split", groups);

  // add input
  bool hasB = fusedDesc->MutableInputDesc("b") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "hasB");
  if (hasB) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasB");
    ge::GeTensorDesc bias = fusedDesc->GetInputDesc(3);
    FUSION_PASS_CHECK(splitDesc->AddInputDesc(bias) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add SplitD input"), return FAILED);
  }

  GeTensorDesc inputDesc = fusedDesc->GetInputDesc(3);
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "The original format of the gru node[name=%s, type=%s]'s input is %s, which is unsupportable.",
                fusedDesc->GetName().c_str(), fusedDesc->GetType().c_str(),
                ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
        return FAILED);

  // add output
  GeShape inputShape = inputDesc.GetShape();
  int newInputChn = inputShape.GetDim(inChannelIdx);
  GeShape splitOutShape = inputShape;
  splitOutShape.SetDim(inChannelIdx, newInputChn / 2);

  std::vector<int64_t> splitOutDims = RemoveNumDirectionsDim(splitOutShape.GetDims(), false);
  GeShape splitDShape(splitOutDims);
  GeTensorDesc splitOutDesc = inputDesc;
  splitOutDesc.Update(splitDShape, ge::FORMAT_ND, DT_FLOAT16);
  splitOutDesc.SetOriginShape(splitDShape);
  for (int i = 0; i < groups; i++) {
    splitDesc->AddOutputDesc(splitOutDesc);
  }

  bool hasBias = fusedDesc->MutableInputDesc("b") != nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "hasBias");
  if (hasBias) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasBias");
    gruOpDesc->UpdateInputDesc("bias_input", splitOutDesc);
    gruOpDesc->UpdateInputDesc("bias_hidden", splitOutDesc);
  }

  // create DynamicGRUV2 Node
  ge::NodePtr gruNode = graph.AddNode(gruOpDesc);
  FUSION_PASS_CHECK(gruNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "DynamicGRUV2 node is null, fusion failed."), return FAILED);

  // create SplitD Node
  ge::NodePtr splitNode = graph.AddNode(splitDesc);
  FUSION_PASS_CHECK(splitNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "SplitD node is null, fusion failed."), return FAILED);
    
  graphStatus status = GraphUtils::AddEdge(fusedNode->GetInDataAnchor(3)->GetPeerOutAnchor(), splitNode->GetInDataAnchor(0));

  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add data to Split edge fail"), return false);
    
  for (int i = 0; i < groups; i++) {
    status = GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i), gruNode->GetInDataAnchor(i + 3));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"), return false);
  }

  // connect x
  FUSION_PASS_CHECK(
    SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(), gruNode->GetInDataAnchor(0)),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node x failed."), return FAILED);

  // connect weight_input
  FUSION_PASS_CHECK(
    SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(), gruNode->GetInDataAnchor(1)),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node weight_input failed."), return FAILED);

  // connect weight_hidden
  FUSION_PASS_CHECK(
    SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(2)->GetPeerOutAnchor(), gruNode->GetInDataAnchor(2)),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node weight_hidden failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "hasSeqLength");
  if (hasSeqLength) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasSeqLength");
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                            gruNode->GetInDataAnchor(5));
  }

  ge::OutDataAnchorPtr outputY = fusedNode->GetOutDataAnchor(0);
  auto yOriTopPeerAnchors = outputY->GetPeerInDataAnchors();
  ge::OutDataAnchorPtr outputYH = fusedNode->GetOutDataAnchor(1);
  auto yhOriTopPeerAnchors = outputYH->GetPeerInDataAnchors();

  OP_LOGI(FUSED_OP_TYPE.c_str(), "init_h");
  if (hasInitH) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "yes hasInitH");
    ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                            gruNode->GetInDataAnchor(6));
  }

  // unlink all input of CommonGRU
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // unlink all output of CommonGRU
  for (auto outAnchor : fusedNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  for (uint64_t i = 0; i < yOriTopPeerAnchors.size(); ++i) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = yOriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(gruNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node output y failed."), return FAILED);
  }

  for (uint64_t i = 0; i < yhOriTopPeerAnchors.size(); ++i) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = yhOriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(gruNode->GetOutDataAnchor(1), oriTopPeerAnchorPtri),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add DynamicLSTMV2 edge to fusion node output y_h failed."), return FAILED);
  }

  return SUCCESS;
}

std::vector<int64_t> GRUFusionPass::RemoveNumDirectionsDim(const std::vector<int64_t> dims, bool isReverse) {
  std::vector<int64_t> res;
  if (isReverse) {
    for (int i = dims.size() - 1; i > 0; --i) {
      res.push_back(dims[i]);
    }
    return res;
  }
  for (int i = 1; i < dims.size(); ++i) {
    res.push_back(dims[i]);
  }
  return res;
}

std::vector<int64_t> GRUFusionPass::ProcessOutputDim(const std::vector<int64_t> dims) {
  std::vector<int64_t> res;
  int n = dims.size();
  FUSION_PASS_CHECK(n < 2, OP_LOGE(FUSED_OP_TYPE.c_str(), "dim size less then 2."), return res);
  int64_t numStep = dims[0];
  int64_t last = dims[n - 1];
  int64_t second = dims[n - 2];
  res.push_back(numStep);
  res.push_back(second);
  res.push_back(last);
  return res;
}

void GRUFusionPass::ProcessNZFormat(std::vector<int64_t> &dims) {
  int n = dims.size();
  FUSION_PASS_CHECK(n < 2, OP_LOGE(FUSED_OP_TYPE.c_str(), "dim size less then 2."), return);
  int64_t first = dims[n - 1];
  int64_t second = dims[n - 2];
  dims[n - 1] = (second + 15) / 16;
  dims[n - 2] = (first + 15) / 16;
  dims.push_back(16);
  dims.push_back(16);
}

void GRUFusionPass::ProcessZFormat(std::vector<int64_t> &dims) {
  for (auto &elem : dims) {
    elem = (elem + 15) / 16;
  }
  dims.push_back(16);
  dims.push_back(16);
}

REGISTER_PASS("GRUFusionPass", BUILT_IN_GRAPH_PASS, GRUFusionPass);
}