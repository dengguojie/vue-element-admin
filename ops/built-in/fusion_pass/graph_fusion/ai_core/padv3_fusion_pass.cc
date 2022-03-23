/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file pad_v3_fusion_pass.cpp
 * \brief split fusion pass(padv3 --> padv3 + strideslice)
 */
#include "padv3_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_PADV3 = "PadV3";
static const char* PADV3 = "PadV3";
static const int LEN = 2;

bool PadV3FusionPass::GetConstValue(const Tensor& constTensor, const DataType& dtype,
                                    std::vector<int64_t>& constData) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* constdataPtr = (int32_t*)constTensor.GetData();
    FUSION_PASS_CHECK(constdataPtr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Get const data failed."),
                      return false);
    if (constdataPtr == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constdataPtr is null");
    }
    size = constTensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      constData.push_back((int32_t)((*(constdataPtr + i))));
      OP_LOGD(FUSED_OP_TYPE.c_str(), "const data int32 fusion pass ====== %d", (int32_t)(*(constdataPtr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* constdataPtr = (int64_t*)constTensor.GetData();
    FUSION_PASS_CHECK(constdataPtr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Get const data failed."),
                      return false);
    size = constTensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      constData.push_back(((int64_t)(*(constdataPtr + i))));
      OP_LOGD(FUSED_OP_TYPE.c_str(), "const data int64 fusion pass ====== %ld", (int64_t)(*(constdataPtr + i)));
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "not support this type");
    return false;
  }
  return true;
}

vector<FusionPattern*> PadV3FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // pad fusion to pad_d
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PadV3Fusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_PADV3, {PADV3}).SetOutput(PATTERN_PADV3);

  patterns.push_back(pattern);

  return patterns;
}

Status PadV3FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get pad node and node-desc
  ge::NodePtr padNode = GetNodeFromMapping(PATTERN_PADV3, mapping);
  FUSION_PASS_CHECK(padNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "padNode is null, fusion failed."), return PARAM_INVALID);

  ge::OpDescPtr padDesc = padNode->GetOpDesc();
  FUSION_PASS_CHECK(padDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "padNode's OpDesc is null, fusion failed."), return PARAM_INVALID);
  
  auto padInputDims = padDesc->GetInputDesc(0).GetShape().GetDims();
  if (CheckDynamic(padInputDims)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input shape is dynamic not changed.");
    return NOT_CHANGED;
  }

  auto padOutputDims = padDesc->GetOutputDesc(0).GetShape().GetDims();
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(padNode);
  Tensor constTensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", constTensor)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "paddings input is not const, not need fusionpass");
    return NOT_CHANGED;
  }

  auto dtype = op.GetInputDesc("paddings").GetDataType();
  std::vector<int64_t> padValue;
  if (!GetConstValue(constTensor, dtype, padValue)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get Const Value failed ");
    return GRAPH_FAILED;
  };
  
  std::vector<int64_t> leftPad;
  std::vector<int64_t> rightPad;
  if (!SplitPadding(padValue, leftPad, rightPad)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "paddings input not have negative number, not need fusionpass");
    return NOT_CHANGED;
  }
  if (UpdatePadding(padNode, leftPad, dtype) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdatePadding failed ");
    return GRAPH_FAILED;
  }
  
  bool paddingsContiguous = true;
  op.GetAttr("paddings_contiguous", paddingsContiguous);
  if (Infer(op, leftPad, paddingsContiguous) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Infer failed ");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> begin;
  CacuBegin(rightPad, paddingsContiguous, begin);
  auto outputDims = padDesc->GetOutputDesc(0).GetShape().GetDims();
  std::vector<int64_t> end;
  CacuEnd(rightPad, end, outputDims, paddingsContiguous);
  std::vector<int64_t> strides(begin.size(), 1);
  
  ge::NodePtr sliceNode = nullptr;
  FUSION_PASS_CHECK(CreateStrideSliceDNode(graph, padNode, sliceNode, padOutputDims,
                    begin, end, strides) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "CreateStrideSliceDNode is fail, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(AddEdge(padNode, sliceNode) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddEdge is fail, fusion failed."),
                    return PARAM_INVALID);
  return GRAPH_SUCCESS;
}

bool PadV3FusionPass::CheckDynamic(std::vector<int64_t>& dims) {
  int size = dims.size();
  for (int i = 0; i < size; ++i) {
    if (dims[i] < 0) {
      return true;
    }
  }
  return false;
}

Status PadV3FusionPass::Infer(const ge::Operator& op, std::vector<int64_t>& paddings, bool paddingsContiguous) {
  auto opInfo = OpDescUtils::GetOpDescFromOperator(op);
  auto inputDesc = opInfo->MutableInputDesc("x");
  auto inputShape = inputDesc->MutableShape().GetDims();
  auto inputShapeMax = inputShape.size();
  auto paddingsShape = paddings.size();
  
  // expand paddings by 0
  auto expandNum = inputShapeMax * LEN - paddingsShape;
  for (size_t dim = 0; dim < expandNum; dim++) {
    paddings.push_back(0);
  }

  if (expandNum > 0) {
    std::vector<int64_t> padvec;
    for (int i = inputShapeMax; i > 0; i--) {
      padvec.push_back(paddings[i * LEN - LEN]);
      padvec.push_back(paddings[i * LEN - 1]);
    }
    paddings = padvec;
  }

  if (!paddingsContiguous) {
    std::vector<int64_t> pads;
    int64_t rank = paddings.size() / LEN;
    for (int i = 0; i < rank; i++) {
      pads.push_back(paddings[i]);
      pads.push_back(paddings[i + rank]);
    }
    paddings = pads;
    OP_LOGI(op.GetName().c_str(), "Get attr paddings_contiguous = false");
  }

  if (inputShape.size() * LEN != paddingsShape) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the num of paddings must be double the input dim size");
    return GRAPH_FAILED;
  }
  auto inputDtype = inputDesc->GetDataType();
  auto outputDesc = opInfo->MutableOutputDesc("y");
  outputDesc->SetDataType(inputDtype);
  vector<int64_t> outputShape;
  for (size_t dim = 0; dim < inputShape.size(); dim++) {
    outputShape.push_back(inputShape[dim] + paddings[dim * LEN] + paddings[dim * LEN + 1]);
  }
  outputDesc->SetShape(GeShape(outputShape));
  outputDesc->SetOriginShape(GeShape(outputShape));
  return GRAPH_SUCCESS;
}

bool PadV3FusionPass::SplitPadding(std::vector<int64_t>& padValue, std::vector<int64_t>& leftPad,
                                   std::vector<int64_t>& rightPad) {
  int size = padValue.size();
  leftPad.resize(size, 0);
  rightPad.resize(size, 0);
  bool isNegtive = false;
  for (int i = 0; i < size; ++i) {
    if (padValue[i] >= 0) {
      leftPad[i] = padValue[i];
    } else {
      rightPad[i] = padValue[i];
      isNegtive = true;
    } 
  }

  for (auto val : leftPad) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "left %d", val);
  }

  for (auto val : rightPad) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "right %d", val);
  }
  return isNegtive;
}

Status PadV3FusionPass::UpdatePadding(ge::NodePtr& padNode, std::vector<int64_t>& pads, ge::DataType dtype) {
  auto inAnchor = padNode->GetInDataAnchor(1);
  FUSION_PASS_CHECK(inAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "indata is null, fusion failed."), return PARAM_INVALID);
  auto outAnchor = inAnchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(outAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "outdata is null, fusion failed."), return PARAM_INVALID);
  auto constNode = outAnchor->GetOwnerNode();
  FUSION_PASS_CHECK(constNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "const node is null, fusion failed."), return PARAM_INVALID);
  auto constDesc = constNode->GetOpDesc();
  FUSION_PASS_CHECK(constDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "const desc is null, fusion failed."), return PARAM_INVALID);
  
  int size = pads.size();
  std::vector<int64_t> dims = {size};
  ge::GeShape inputShape(dims);
  ge::GeTensorDesc outTensorDesc(inputShape, ge::FORMAT_ND, dtype);

  GeTensorPtr constValue = std::make_shared<ge::GeTensor>(outTensorDesc, reinterpret_cast<uint8_t*>(pads.data()),
                            size * sizeof(int64_t));
  if (dtype == ge::DT_INT32) {
    std::vector<int32_t> vals(size);
    for (int i = 0; i < size; ++i) {
      vals[i] = pads[i];
    }
    constValue = std::make_shared<ge::GeTensor>(outTensorDesc, reinterpret_cast<uint8_t*>(vals.data()),
                  size * sizeof(int32_t));
  }
  if (!AttrUtils::SetTensor(constDesc, ATTR_NAME_WEIGHTS, constValue)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create ATTR_NAME_WEIGHTS failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void PadV3FusionPass::CacuBegin(std::vector<int64_t>& pads, bool paddingsContiguous, std::vector<int64_t>& begin) {
  int loop = pads.size() / LEN;
  int beginIndex = 0;
  for (int i = 0; i < loop; ++i) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "begin_index %d", beginIndex);
    begin.push_back(abs(pads[beginIndex]));
    if (paddingsContiguous) {
      beginIndex += LEN;
    } else {
      ++beginIndex;
    }
  }

  for (auto val : begin) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "begin %d", val);
  }
}

void PadV3FusionPass::CacuEnd(std::vector<int64_t>& pads, std::vector<int64_t>& end,
                              std::vector<int64_t>& outputDims, bool paddingsContiguous) {
  int loop = pads.size() / LEN;
  int endIndex = 1;
  if (!paddingsContiguous) {
    endIndex = loop;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "endIndex %d", endIndex);
  for (int i = 0; i < loop; ++i) {
    end.push_back(abs(outputDims[i] + pads[endIndex]));
    if (paddingsContiguous) {
      endIndex += LEN;
    } else {
      ++endIndex;
    }
  }

  for (auto val : end) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "end %d", val);
  }
}

Status PadV3FusionPass::CreateStrideSliceDNode(ge::ComputeGraph& graph, ge::NodePtr& padNode,
                                               ge::NodePtr& newNode, std::vector<int64_t>& outputDims,
                                               std::vector<int64_t>& begin,
                                               std::vector<int64_t>& end, std::vector<int64_t>& strides) {
  std::shared_ptr<ge::OpDesc> newDesc = std::make_shared<ge::OpDesc>(padNode->GetName() +
      "_StridedSliceD", "StridedSliceD");
  FUSION_PASS_CHECK(newDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
      "StridedSliceD desc is null, fusion failed."),
                    return PARAM_INVALID);
  auto inputDesc = padNode->GetOpDesc()->GetOutputDesc(0);
  newDesc->AddInputDesc("x", inputDesc);
  ge::AttrUtils::SetListInt(newDesc, "begin", begin);
  ge::AttrUtils::SetListInt(newDesc, "end", end);
  ge::AttrUtils::SetListInt(newDesc, "strides", strides);
  ge::AttrUtils::SetInt(newDesc, "begin_mask", 0);
  ge::AttrUtils::SetInt(newDesc, "end_mask", 0);
  ge::AttrUtils::SetInt(newDesc, "ellipsis_mask", 0);
  ge::AttrUtils::SetInt(newDesc, "new_axis_mask", 0);
  ge::AttrUtils::SetInt(newDesc, "shrink_axis_mask", 0);
  auto outputDesc = padNode->GetOpDesc()->GetOutputDesc(0);
  outputDesc.SetShape(GeShape(outputDims));
  newDesc->AddOutputDesc("y", outputDesc);
  newNode = graph.AddNode(newDesc);
  return GRAPH_SUCCESS;
}

Status PadV3FusionPass::AddEdge(ge::NodePtr& padNode, ge::NodePtr& sliceNode) const {
  for (auto inDataAnchor: padNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(padNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(sliceNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add out data edge failed."), return FAILED);
  }

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(padNode->GetOutDataAnchor(0), sliceNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge between node %s. and node %s failed.",
                    padNode->GetName().c_str(), sliceNode->GetName().c_str()), return FAILED);
  return GRAPH_SUCCESS;
}
REGISTER_PASS("PadV3FusionPass", BUILT_IN_GRAPH_PASS, PadV3FusionPass);
}  // namespace fe
