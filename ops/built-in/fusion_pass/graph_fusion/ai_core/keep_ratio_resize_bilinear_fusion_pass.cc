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
 * \file keep_ratio_resize_bilinear_fusion_pass.cpp
 * \brief KeepRatioResizeBilinear fusion pass
 */
#include "keep_ratio_resize_bilinear_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "KeepRatioResizeBilinear";
static const std::string PATTERN_FUSEDNODE = "KeepRatioResizeBilinear";

vector<FusionPattern*> KeepRatioResizeBilinearFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("KeepRatioResizeBilinearFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(fuseNodeType.c_str(), "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status KeepRatioResizeBilinearFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                 vector<ge::NodePtr>& fusionNodes) {
  // get fused node
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(fuseNodeType.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // fusion KeepRatioResizeBilinear to ResizeBilinear
  std::shared_ptr<ge::OpDesc> resizeBilinearDesc = nullptr;
  std::string resizeBilinearName = fusedNode->GetName() + "_fused_to_resize_bilinear";
  FUSION_PASS_MAKE_SHARED(
      (resizeBilinearDesc = std::make_shared<ge::OpDesc>(resizeBilinearName, "ResizeBilinearV2")),
      return FAILED);
  FUSION_PASS_CHECK(resizeBilinearDesc == nullptr,
                    OP_LOGE(fuseNodeType.c_str(), "resizeBilinearDesc is null, fusion failed."), return FAILED);

  // set ResizeBilinearV2 input desc
  ge::GeTensorDesc inputDesc = fusedNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(resizeBilinearDesc->AddInputDesc("x", inputDesc) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "add input desc of %s failed.", fusedNode->GetName().c_str()),
                    return FAILED);
  // set ResizeBilinearV2 output desc
  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(resizeBilinearDesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "add output desc of %s failed.", fusedNode->GetName().c_str()),
                    return FAILED);

  // get attr from KeepRatioResizeBilinearFusion
  int32_t minDims = 0;
  int32_t maxDims = 0;
  ge::AttrUtils::GetInt(fusedNode->GetOpDesc(), "min_dimension", minDims);
  ge::AttrUtils::GetInt(fusedNode->GetOpDesc(), "max_dimension", maxDims);
  bool alignCorners = false;
  bool halfPixelCenters = false;
  ge::AttrUtils::GetBool(fusedNode->GetOpDesc(), "align_corners", alignCorners);
  ge::AttrUtils::GetBool(fusedNode->GetOpDesc(), "half_pixel_centers", halfPixelCenters);

  // input shape
  int64_t heightDIms = 0;
  int64_t widthDims = 0;
  vector<int64_t> oriinputShape = inputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(oriinputShape.empty(),
                    OP_LOGE(fuseNodeType.c_str(), "Node[%s] input shape is NULL.", fusedNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(oriinputShape.size() != 4,
                    OP_LOGE(fuseNodeType.c_str(), "Node[%s] input shape is not equal 4.", fusedNode->GetName().c_str()),
                    return FAILED);
  if (inputDesc.GetFormat() == ge::FORMAT_NHWC) {
    heightDIms = oriinputShape[1];
    widthDims = oriinputShape[2];
  } else {
    heightDIms = oriinputShape[2];
    widthDims = oriinputShape[3];
  }

  if (PatternFusionUtil::IsUnknownShape(heightDIms) || PatternFusionUtil::IsUnknownShape(widthDims)) {
    OP_LOGE(fuseNodeType.c_str(), "KeepRatioResizeBilinearFusion cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  // calcu size
  float minDimsFloat = static_cast<float>(minDims);
  float maxDimsFloat = static_cast<float>(maxDims);
  int64_t minShapeDims = std::min(heightDIms, widthDims);
  int64_t maxShapeDims = std::max(heightDIms, widthDims);
  float minShapeDimsFloat = static_cast<float>(minShapeDims);
  float maxShapeDimsFloat = static_cast<float>(maxShapeDims);
  if (minDims == 0 || maxDims == 0) {
    OP_LOGE(fuseNodeType.c_str(), "KeepRatioResizeBilinearFusion minDims or maxDims can not be zero.");
    return NOT_CHANGED;
  }
  // get min scale
  float resizeScale = minShapeDimsFloat / minDimsFloat;
  float minNewShapeH = floor((heightDIms / resizeScale) + 0.5);
  float minNewShapeW = floor((widthDims / resizeScale) + 0.5);
  float minNewShapeMaxDim = std::max(minNewShapeH, minNewShapeW);

  // get max scale
  resizeScale = maxShapeDimsFloat / maxDimsFloat;
  float maxNewShapeH = floor((heightDIms / resizeScale) + 0.5);
  float maxNewShapeW = floor((widthDims / resizeScale) + 0.5);

  vector<int32_t> sizeVec;
  if (minNewShapeMaxDim > maxDimsFloat) {
    sizeVec.push_back(static_cast<int32_t>(maxNewShapeH));
    sizeVec.push_back(static_cast<int32_t>(maxNewShapeW));
  } else {
    sizeVec.push_back(static_cast<int32_t>(minNewShapeH));
    sizeVec.push_back(static_cast<int32_t>(minNewShapeW));
  }

  // add edge for node
  // add node to graph
  ge::NodePtr resizeBilinearNode = graph.AddNode(resizeBilinearDesc);
  // add input for ResizeBilinearV2
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            resizeBilinearNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "AddEdge edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                               fusedNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(fuseNodeType.c_str(), "AddEdge edge failed."), return FAILED);
  // add output for ResizeBilinearV2
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(fuseNodeType.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(resizeBilinearNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(fuseNodeType.c_str(), "Add edge failed."), return FAILED);
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    OP_LOGE(fuseNodeType.c_str(), "Remove node:[%s] failed.", fusedNode->GetName().c_str()),
                    return FAILED);
  // new the size const node for ResizeBilinearV2
  std::string sizeName = fusedNode->GetName() + "_size_const";
  ge::GeTensorPtr sizeConstPtr = nullptr;
  ge::GeShape sizeConstShape({2});
  ge::GeTensorDesc sizeConstDesc(sizeConstShape, ge::FORMAT_ND, ge::DT_INT32);
  FUSION_PASS_MAKE_SHARED(
      (sizeConstPtr = std::make_shared<ge::GeTensor>(sizeConstDesc, reinterpret_cast<uint8_t*>(sizeVec.data()),
                                                     sizeVec.size() * sizeof(DT_INT32))),
      sizeConstPtr = nullptr;
      return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {sizeConstPtr};
  (void)ge::OpDescUtils::SetWeights(resizeBilinearNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(resizeBilinearNode);
  if (constInputNodes.size() <= 0) {
    OP_LOGE(fuseNodeType.c_str(), "GetConstInputs Error");
    return PARAM_INVALID;
  }
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType("Const");
  ge::AttrUtils::SetBool(resizeBilinearNode->GetOpDesc(), "align_corners", alignCorners);
  ge::AttrUtils::SetBool(resizeBilinearNode->GetOpDesc(), "half_pixel_centers", halfPixelCenters);
  // set second input desc
  vector<int64_t> dims = {2};
  ge::GeShape inputSizeShape(dims);
  inputDesc.SetShape(inputSizeShape);
  inputDesc.SetOriginShape(inputSizeShape);
  inputDesc.SetFormat(ge::FORMAT_ND);
  inputDesc.SetOriginFormat(ge::FORMAT_ND);
  inputDesc.SetDataType(ge::DT_INT32);
  resizeBilinearDesc->UpdateInputDesc(1, inputDesc);

  // try to trans to ResizeBilinearV2D
  // build attr infos
  std::string fusionOpType = "ResizeBilinearV2D";
  std::vector<PassAttrInfo> attrInfos;
  PassAttrInfo size = {1, "size", "SetListInt"};
  attrInfos.push_back(size);

  // build a fusion node op desc
  bool doConst2Attr = true;
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(resizeBilinearNode, fusionOpType, attrInfos);
  if (fusionDescPtr == nullptr) {
    doConst2Attr = false;
    OP_LOGW(fuseNodeType.c_str(), "Fusion OP ResizeBilinearV2 to ResizeBilinearV2D is nullptr, ignore const to attr.");
  }

  // check op support
  if (doConst2Attr) {
    auto isSupported = CheckOpSupported(fusionDescPtr);
    if (isSupported) {
      // const to attr
      OP_LOGI(fuseNodeType.c_str(), "Op ResizeBilinearV2D Supported will do const2Attr.");
      ge::NodePtr const2AttrNode = nullptr;
      Status ret =
          PatternFusionUtil::ConstToAttrWithNode(graph, resizeBilinearNode, fusionOpType, attrInfos, const2AttrNode);
      if (ret != SUCCESS) {
        OP_LOGI(fuseNodeType.c_str(), "ResizeBilinearV2 do const2Attr failed, ignore.");
      } else {
        fusionNodes.push_back(const2AttrNode);
      }
    } else {
      OP_LOGI(fuseNodeType.c_str(), "Op ResizeBilinearV2D Not Supported, ignore const to attr.");
    }
  }
  OP_LOGI(fuseNodeType.c_str(), "KeepRatioResizeBilinearFusion fusion SUCCESSS!!!!!");
  return SUCCESS;
}
REGISTER_PASS("KeepRatioResizeBilinearFusion", BUILT_IN_GRAPH_PASS, KeepRatioResizeBilinearFusionPass);
}  // namespace fe
