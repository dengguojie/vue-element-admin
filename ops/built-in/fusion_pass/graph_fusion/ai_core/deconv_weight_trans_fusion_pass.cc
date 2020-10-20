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
 * \file deconv_weight_trans_fusion_pass.cpp
 * \brief deconv weight trans fusion pass(weight -> deconv ===> weight ->
 *   reshape -> transpose -> reshape -> reverse -> reshape -> deconv)
 */
#include "deconv_weight_trans_fusion_pass.h"

#include <cmath>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace ge;

namespace fe {
namespace {
const string DECONV = "Deconvolution";
const string CONV2D_TRANSPOSE = "Conv2DTransposeD";
const string PATTERN_DECONV = "DeconvolutionInt8";
static const std::string CONSTANTOP = "Const";
static const std::string CONSTANT = "Constant";
static const int CONST_DIM4_NUM = 4;
static const int CONST_DIM3_NUM = 3;
static const int CONST_DIM2_NUM = 2;
}  // namespace

vector<FusionPattern*> DeconvWeightTransFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(),
          "Enter DeconvWeightTransFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("DeconvWeightTransFusionPass");
  FUSION_PASS_CHECK(
      pattern == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
      return patterns);

  pattern->AddOpDesc(PATTERN_DECONV, {DECONV, CONV2D_TRANSPOSE})
      .SetOutput(PATTERN_DECONV);

  patterns.push_back(pattern);

  return patterns;
}

/* weight not 4D, need to complement 1
 * 1D -> C
 * 2D format: HWCN  -->CN
 *            NCHW/NHWC  -->CH
 * 3D -> CHW
 */
static Status GetShapeByFormat(const ge::Format& format,
                               const ge::GeShape& oldShape, int64_t& number,
                               int64_t& channel, int64_t& height,
                               int64_t& weight) {
  if (oldShape.GetDimNum() == 1) {
    channel = oldShape.GetDim(0);
    number = 1;
    height = 1;
    weight = 1;
  } else if (oldShape.GetDimNum() == CONST_DIM2_NUM) {
    if (format == ge::FORMAT_HWCN) {
      channel = oldShape.GetDim(0);
      number = oldShape.GetDim(1);
      height = 1;
      weight = 1;
    } else {
      channel = oldShape.GetDim(0);
      height = oldShape.GetDim(1);
      number = 1;
      weight = 1;
    }
  } else if (oldShape.GetDimNum() == CONST_DIM3_NUM) {
    channel = oldShape.GetDim(0);
    height = oldShape.GetDim(1);
    weight = oldShape.GetDim(CONST_DIM2_NUM);
    number = 1;
  } else {
    if (format == ge::FORMAT_NCHW) {
      number = oldShape.GetDim(0);
      channel = oldShape.GetDim(1);
      height = oldShape.GetDim(CONST_DIM2_NUM);
      weight = oldShape.GetDim(CONST_DIM3_NUM);
    } else if (format == ge::FORMAT_HWCN) {
      number = oldShape.GetDim(CONST_DIM3_NUM);
      channel = oldShape.GetDim(CONST_DIM2_NUM);
      height = oldShape.GetDim(0);
      weight = oldShape.GetDim(1);
    } else if (format == ge::FORMAT_NHWC) {
      number = oldShape.GetDim(0);
      channel = oldShape.GetDim(CONST_DIM3_NUM);
      height = oldShape.GetDim(1);
      weight = oldShape.GetDim(CONST_DIM2_NUM);
    } else {
      return FAILED;
    }
  }
  return SUCCESS;
}

void DeconvWeightTransFusionPass::
    GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
        const ge::Format& filterFormat, const vector<int64_t>& shapeNCHW,
        vector<int64_t>& dimComp, vector<int64_t>& reshapeIn,
        vector<int64_t>& transPerm, vector<int64_t>& reverseAxis,
        vector<int64_t>& reshapeOut) {
  if (shapeNCHW.size() != CONST_DIM4_NUM) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "size of shapeNCHW not equal 4");
    return;
  }
  dimComp.resize(CONST_DIM4_NUM);
  reshapeIn.resize(CONST_DIM3_NUM);
  transPerm.resize(CONST_DIM4_NUM);
  reverseAxis.resize(1);
  reshapeOut.resize(CONST_DIM4_NUM);
  int64_t number = shapeNCHW[0];
  int64_t channel = shapeNCHW[1];
  int64_t height = shapeNCHW[CONST_DIM2_NUM];
  int64_t weight = shapeNCHW[CONST_DIM3_NUM];
  if (filterFormat == ge::FORMAT_HWCN) {
    dimComp[0] = height;
    dimComp[1] = weight;
    dimComp[CONST_DIM2_NUM] = channel;
    dimComp[CONST_DIM3_NUM] = number;
    transPerm[0] = 0;
    transPerm[1] = 1;
    transPerm[CONST_DIM2_NUM] = CONST_DIM3_NUM;
    transPerm[CONST_DIM3_NUM] = CONST_DIM2_NUM;
    reshapeIn[0] = height * weight;
    reshapeIn[1] = number;
    reshapeIn[CONST_DIM2_NUM] = channel;
    reverseAxis[0] = 0;
    reshapeOut[0] = height;
    reshapeOut[1] = weight;
    reshapeOut[CONST_DIM2_NUM] = number;
    reshapeOut[CONST_DIM3_NUM] = channel;
  } else if (filterFormat == ge::FORMAT_NCHW) {
    dimComp[0] = number;
    dimComp[1] = channel;
    dimComp[CONST_DIM2_NUM] = height;
    dimComp[CONST_DIM3_NUM] = weight;
    transPerm[0] = 1;
    transPerm[1] = 0;
    transPerm[CONST_DIM2_NUM] = CONST_DIM2_NUM;
    transPerm[CONST_DIM3_NUM] = CONST_DIM3_NUM;
    reshapeIn[0] = channel;
    reshapeIn[1] = number;
    reshapeIn[CONST_DIM2_NUM] = height * weight;
    reverseAxis[0] = CONST_DIM2_NUM;
    reshapeOut[0] = channel;
    reshapeOut[1] = number;
    reshapeOut[CONST_DIM2_NUM] = height;
    reshapeOut[CONST_DIM3_NUM] = weight;
  } else if (filterFormat == ge::FORMAT_NHWC) {
    dimComp[0] = number;
    dimComp[1] = height;
    dimComp[CONST_DIM2_NUM] = weight;
    dimComp[CONST_DIM3_NUM] = channel;
    transPerm[0] = 0;
    transPerm[1] = 1;
    transPerm[CONST_DIM2_NUM] = CONST_DIM3_NUM;
    transPerm[CONST_DIM3_NUM] = CONST_DIM2_NUM;
    reshapeIn[0] = channel;
    reshapeIn[1] = height * weight;
    reshapeIn[CONST_DIM2_NUM] = number;
    reverseAxis[0] = 1;
    reshapeOut[0] = channel;
    reshapeOut[1] = height;
    reshapeOut[CONST_DIM2_NUM] = weight;
    reshapeOut[CONST_DIM3_NUM] = number;
  }
}

static Status GenerateTransposeNode(ge::ComputeGraph& graph,
                                    ge::GeTensorDesc& prevOutDesc,
                                    ge::GeTensorDesc& nextInDesc,
                                    const vector<int64_t>& perm,
                                    ge::NodePtr& transposeNode,
                                    const std::string& basename) {
  vector<int64_t> nextInShape(CONST_DIM4_NUM);
  for (size_t i = 0; i < perm.size(); ++i) {
    nextInShape[i] = prevOutDesc.GetShape().GetDim(perm[i]);
  }
  ge::OpDescPtr transposeDesc;
  FUSION_PASS_MAKE_SHARED(
      (transposeDesc = std::make_shared<ge::OpDesc>(
           basename + "_const_fold_transpose_nc", "TransposeD")),
      return FAILED);
  prevOutDesc.SetFormat(ge::FORMAT_ND);
  prevOutDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddInputDesc("x", prevOutDesc);
  nextInDesc.SetFormat(ge::FORMAT_ND);
  nextInDesc.SetOriginFormat(ge::FORMAT_ND);
  nextInDesc.SetShape(ge::GeShape(nextInShape));
  nextInDesc.SetOriginShape(ge::GeShape(nextInShape));
  transposeDesc->AddOutputDesc("y", nextInDesc);
  ge::AttrUtils::SetListInt(transposeDesc, "perm", perm);
  transposeNode = graph.AddNode(transposeDesc);
  return SUCCESS;
}

static Status GenerateReshapeNode(ge::ComputeGraph& graph,
                                  ge::GeTensorDesc& prevOutDesc,
                                  ge::GeTensorDesc& nextInDesc,
                                  const vector<int64_t>& shape,
                                  ge::NodePtr& shapeNode,
                                  const std::string& name,
                                  const std::string& basename) {
  ge::OpDescPtr reshapeDesc;
  FUSION_PASS_MAKE_SHARED((reshapeDesc = std::make_shared<ge::OpDesc>(
                               basename + "_const_fold_" + name, "Reshape")),
                          return FAILED);
  reshapeDesc->AddInputDesc("x", prevOutDesc);
  nextInDesc.SetShape(ge::GeShape(shape));
  nextInDesc.SetOriginShape(ge::GeShape(shape));
  reshapeDesc->AddOutputDesc("y", nextInDesc);
  ge::AttrUtils::SetListInt(reshapeDesc, "shape", shape);
  shapeNode = graph.AddNode(reshapeDesc);
  return SUCCESS;
}

static Status GenerateReverseNode(ge::ComputeGraph& graph,
                                  ge::GeTensorDesc& prevOutDesc,
                                  ge::GeTensorDesc& nextInDesc,
                                  const vector<int64_t>& axis,
                                  ge::NodePtr& reverseNode,
                                  const std::string& basename) {
  ge::OpDescPtr reverseDesc;
  FUSION_PASS_MAKE_SHARED(
      (reverseDesc = std::make_shared<ge::OpDesc>(
           basename + "_const_fold_reverse_hw", "ReverseV2D")),
      return FAILED);
  reverseDesc->AddInputDesc("x", prevOutDesc);
  nextInDesc.SetShape(prevOutDesc.GetShape());
  nextInDesc.SetOriginShape(prevOutDesc.GetShape());
  reverseDesc->AddOutputDesc("y", nextInDesc);
  ge::AttrUtils::SetListInt(reverseDesc, "axis", axis);
  reverseNode = graph.AddNode(reverseDesc);
  return SUCCESS;
}

static Status GenerateReFormatNode(ge::ComputeGraph& graph,
                                   ge::GeTensorDesc& prevOutDesc,
                                   ge::GeTensorDesc& nextInDesc,
                                   const ge::Format& format,
                                   ge::NodePtr& reformatNode,
                                   const std::string& basename) {
  ge::OpDescPtr reformatDesc;
  FUSION_PASS_MAKE_SHARED((reformatDesc = std::make_shared<ge::OpDesc>(
                               basename + "_const_fold_reformat", "ReFormat")),
                          return FAILED);
  reformatDesc->AddInputDesc("x", prevOutDesc);
  nextInDesc.SetShape(prevOutDesc.GetShape());
  nextInDesc.SetOriginShape(prevOutDesc.GetShape());
  nextInDesc.SetFormat(format);
  nextInDesc.SetOriginFormat(format);
  reformatDesc->AddOutputDesc("y", nextInDesc);
  reformatNode = graph.AddNode(reformatDesc);
  return SUCCESS;
}

Status DeconvWeightTransFusionPass::Relink(
    ge::NodePtr filterNode, ge::NodePtr dimCompNode, ge::NodePtr transposeNode,
    ge::NodePtr reformatNode, ge::NodePtr reshapeInNode,
    ge::NodePtr reverseNode, ge::NodePtr reshapeOutNode,
    ge::NodePtr deconvNode) {
  // weight -> Deconvolution
  // weight -> [dimComp] -> transpose -> [reshapeIn -> reverse -> reshapeOut] ->
  // deconv
  int filterAnchor = 1;
  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(filterNode->GetOutDataAnchor(0),
                                 deconvNode->GetInDataAnchor(filterAnchor)) !=
          SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to remove edge between filterNode and deconvNode"),
      return FAILED);

  OutDataAnchorPtr dimCompOutAnchor = nullptr;
  if (dimCompNode != nullptr) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(filterNode->GetOutDataAnchor(0),
                                dimCompNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to add edge between filterNode and dimCompNode"),
        return FAILED);
    dimCompOutAnchor = dimCompNode->GetOutDataAnchor(0);
  } else {
    dimCompOutAnchor = filterNode->GetOutDataAnchor(0);
  }

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(dimCompOutAnchor,
                              transposeNode->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to add edge between dimCompNode and transposeNode"),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0),
                              reformatNode->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to add edge between dimCompNode and transposeNode"),
      return FAILED);

  if (reshapeInNode != nullptr) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reformatNode->GetOutDataAnchor(0),
                                reshapeInNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to add edge between transposeNode and reshapeInNode"),
        return FAILED);

    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reshapeInNode->GetOutDataAnchor(0),
                                reverseNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reshapeInNode and reverseNode"),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reverseNode->GetOutDataAnchor(0),
                                reshapeOutNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reverseNode and reshapeOutNode"),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reshapeOutNode->GetOutDataAnchor(0),
                                deconvNode->GetInDataAnchor(filterAnchor)) !=
            SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reshapeOutNode and deconvNode"),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reformatNode->GetOutDataAnchor(0),
                                deconvNode->GetInDataAnchor(filterAnchor)) !=
            SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to add edge between transposeNode and deconvNode"),
        return FAILED);
  }

  FUSION_PASS_CHECK(
      deconvNode->GetOpDesc()->UpdateInputDesc(
          filterAnchor, reformatNode->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to update input description of deconv"),
      return FAILED);

  return SUCCESS;
}

Status DeconvWeightTransFusionPass::Fusion(ge::ComputeGraph& graph,
                                           Mapping& mapping,
                                           vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvWeightTransFusionPass.");
  ge::NodePtr deconvNode = GetNodeFromMapping(PATTERN_DECONV, mapping);

  // pattern
  // originFormat: NCHW,HWCN,NHWC
  // for example: NCHW
  // weight(NCHW) -> | dim completion -> transpose -> reshape in -> reverse ->
  // reshape out | -> Deconvolution
  //                 | |
  //                 | |
  //                 | |

  // input: x, filter, bias
  int xAnchor = 0;
  int filterAnchor = 1;

  // prerequisite
  ge::NodePtr xNode =
      deconvNode->GetInDataAnchor(xAnchor)->GetPeerOutAnchor()->GetOwnerNode();
  int xIdx = deconvNode->GetInDataAnchor(xAnchor)->GetPeerOutAnchor()->GetIdx();
  ge::NodePtr filterNode = deconvNode->GetInDataAnchor(filterAnchor)
                               ->GetPeerOutAnchor()
                               ->GetOwnerNode();
  int filterIdx =
      deconvNode->GetInDataAnchor(filterAnchor)->GetPeerOutAnchor()->GetIdx();
  if (filterNode->GetOpDesc()->GetOutputDesc(filterIdx).GetDataType() !=
          ge::DT_INT8 ||
      xNode->GetOpDesc()->GetOutputDesc(xIdx).GetDataType() != ge::DT_INT8) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The dtype of weight or x is not int8.");
    return NOT_CHANGED;
  }
  std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(filterNode);
  if (type != CONSTANT && type != CONSTANTOP) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The type of weight is not constant.");
    return NOT_CHANGED;
  }

  int64_t number = 0, channel = 0, height = 0, weight = 0;
  ge::GeTensorDesc deconvWeightInDesc =
      deconvNode->GetOpDesc()->GetInputDesc(filterAnchor);
  ge::GeTensorDesc filterOutDesc =
      filterNode->GetOpDesc()->GetOutputDesc(filterIdx);
  ge::GeShape filterShape = filterOutDesc.GetShape();
  ge::Format filterFormat = filterOutDesc.GetFormat();
  FUSION_PASS_CHECK(GetShapeByFormat(filterFormat, filterShape, number, channel,
                                     height, weight) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "Not support this format %d.", filterFormat),
                    return NOT_CHANGED);

  vector<int64_t> dimComp(CONST_DIM4_NUM), reshapeIn(CONST_DIM3_NUM),
      transPerm(CONST_DIM4_NUM), reverseAxis(1), reshapeOut(CONST_DIM4_NUM);
  GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
      filterFormat, {number, channel, height, weight}, dimComp, reshapeIn,
      transPerm, reverseAxis, reshapeOut);

  ge::NodePtr dimCompNode = nullptr, transposeNode = nullptr,
              reformatNode = nullptr, reshapeInNode = nullptr,
              reverseNode = nullptr, reshapeOutNode = nullptr;

  auto basename = filterNode->GetName();
  // 1. dimension completion
  if (filterShape.GetDimNum() != CONST_DIM4_NUM) {
    FUSION_PASS_CHECK(
        GenerateReshapeNode(graph, filterOutDesc, deconvWeightInDesc, dimComp,
                            dimCompNode, "dimension_completion",
                            basename) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "fail to generate dimension completion node"),
        return FAILED);
  }
  ge::GeTensorDesc dimCompOutDesc;
  if (dimCompNode != nullptr) {
    dimCompOutDesc = dimCompNode->GetOpDesc()->GetOutputDesc(0);
  } else {
    dimCompOutDesc = filterOutDesc;
  }

  // 2. transpose number,channel
  FUSION_PASS_CHECK(
      GenerateTransposeNode(graph, dimCompOutDesc, deconvWeightInDesc,
                            transPerm, transposeNode, basename) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate transpose node"),
      return FAILED);
  ge::GeTensorDesc transposeOutDesc =
      transposeNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(
      GenerateReFormatNode(graph, transposeOutDesc, deconvWeightInDesc,
                           filterFormat, reformatNode, basename) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reformat node"),
      return FAILED);
  ge::GeTensorDesc reformatDesc = reformatNode->GetOpDesc()->GetOutputDesc(0);

  if (height != 1 || weight != 1) {
    // 3. fuse height, weight
    FUSION_PASS_CHECK(
        GenerateReshapeNode(graph, reformatDesc, deconvWeightInDesc, reshapeIn,
                            reshapeInNode, "reshape_in", basename) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reshape in node"),
        return FAILED);
    ge::GeTensorDesc reshapeInOutDesc =
        reshapeInNode->GetOpDesc()->GetOutputDesc(0);

    // 4. reverse height*weight
    FUSION_PASS_CHECK(
        GenerateReverseNode(graph, reshapeInOutDesc, deconvWeightInDesc,
                            reverseAxis, reverseNode, basename) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reverse node"),
        return FAILED);
    ge::GeTensorDesc reverseOutDesc =
        reverseNode->GetOpDesc()->GetOutputDesc(0);

    // 5. anti-fusion height*weight
    FUSION_PASS_CHECK(
        GenerateReshapeNode(graph, reverseOutDesc, deconvWeightInDesc,
                            reshapeOut, reshapeOutNode, "reshape_out",
                            basename) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reshape out node"),
        return FAILED);
  }

  FUSION_PASS_CHECK(
      Relink(filterNode, dimCompNode, transposeNode, reformatNode,
             reshapeInNode, reverseNode, reshapeOutNode, deconvNode) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End DeconvWeightTransFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("DeconvWeightTransFusionPass", BUILT_IN_GRAPH_PASS,
              DeconvWeightTransFusionPass);
}  // namespace fe
