/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief deconv weight trans fusion pass(weight -> deconv ===> weight ->
 * reshape -> transpose -> reshape -> reverse -> reshape -> deconv)
 *
 * @author x00444734
 */

#include "deconv_weight_trans_fusion_pass.h"

#include <cmath>
#include <string>
#include <vector>

#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;

namespace fe {
namespace {
const string DECONV = "Deconvolution";
const string PATTERN_DECONV = "DeconvolutionInt8";
static const std::string CONSTANTOP = "Const";
static const std::string CONSTANT = "Constant";
}

vector<FusionPattern *> DeconvWeightTransFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvWeightTransFusionPass::DefinePatterns.");
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("DeconvWeightTransFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_DECONV, {DECONV}).SetOutput(PATTERN_DECONV);

  patterns.push_back(pattern);

  return patterns;
}

/* weight not 4D, need to complement 1
 * 1D -> C
 * 2D format: HWCN  -->CN
 *            NCHW/NHWC  -->CH
 * 3D -> CHW
 */
static Status GetShapeByFormat(const ge::Format &format,
                               const ge::GeShape &oldShape, int64_t &N,
                               int64_t &C, int64_t &H, int64_t &W) {
  if (oldShape.GetDimNum() == 1) {
    C = oldShape.GetDim(0);
    N = 1;
    H = 1;
    W = 1;
  } else if (oldShape.GetDimNum() == 2) {
    if (format == ge::FORMAT_HWCN) {
      C = oldShape.GetDim(0);
      N = oldShape.GetDim(1);
      H = 1;
      W = 1;
    } else {
      C = oldShape.GetDim(0);
      H = oldShape.GetDim(1);
      N = 1;
      W = 1;
    }
  } else if (oldShape.GetDimNum() == 3) {
    C = oldShape.GetDim(0);
    H = oldShape.GetDim(1);
    W = oldShape.GetDim(2);
    N = 1;
  } else {
    if (format == ge::FORMAT_NCHW) {
      N = oldShape.GetDim(0);
      C = oldShape.GetDim(1);
      H = oldShape.GetDim(2);
      W = oldShape.GetDim(3);
    } else if (format == ge::FORMAT_HWCN) {
      N = oldShape.GetDim(3);
      C = oldShape.GetDim(2);
      H = oldShape.GetDim(0);
      W = oldShape.GetDim(1);
    } else if (format == ge::FORMAT_NHWC) {
      N = oldShape.GetDim(0);
      C = oldShape.GetDim(3);
      H = oldShape.GetDim(1);
      W = oldShape.GetDim(2);
    } else {
      return FAILED;
    }
  }
  return SUCCESS;
}

void DeconvWeightTransFusionPass::GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
    const ge::Format &filterFormat, const vector<int64_t> &shapeNCHW,
    vector<int64_t> &dimComp, vector<int64_t> &reshapeIn,
    vector<int64_t> &transPerm, vector<int64_t> &reverseAxis,
    vector<int64_t> &reshapeOut) {
  if (shapeNCHW.size() != 4) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "size of shapeNCHW not equal 4");
    return;
  }
  dimComp.resize(4);
  reshapeIn.resize(3);
  transPerm.resize(4);
  reverseAxis.resize(1);
  reshapeOut.resize(4);
  int64_t N = shapeNCHW[0];
  int64_t C = shapeNCHW[1];
  int64_t H = shapeNCHW[2];
  int64_t W = shapeNCHW[3];
  if (filterFormat == ge::FORMAT_HWCN) {
    dimComp[0] = H;
    dimComp[1] = W;
    dimComp[2] = C;
    dimComp[3] = N;
    transPerm[0] = 0;
    transPerm[1] = 1;
    transPerm[2] = 3;
    transPerm[3] = 2;
    reshapeIn[0] = H * W;
    reshapeIn[1] = N;
    reshapeIn[2] = C;
    reverseAxis[0] = 0;
    reshapeOut[0] = H;
    reshapeOut[1] = W;
    reshapeOut[2] = N;
    reshapeOut[3] = C;
  } else if (filterFormat == ge::FORMAT_NCHW) {
    dimComp[0] = N;
    dimComp[1] = C;
    dimComp[2] = H;
    dimComp[3] = W;
    transPerm[0] = 1;
    transPerm[1] = 0;
    transPerm[2] = 2;
    transPerm[3] = 3;
    reshapeIn[0] = C;
    reshapeIn[1] = N;
    reshapeIn[2] = H * W;
    reverseAxis[0] = 2;
    reshapeOut[0] = C;
    reshapeOut[1] = N;
    reshapeOut[2] = H;
    reshapeOut[3] = W;
  } else if (filterFormat == ge::FORMAT_NHWC) {
    dimComp[0] = N;
    dimComp[1] = H;
    dimComp[2] = W;
    dimComp[3] = C;
    transPerm[0] = 0;
    transPerm[1] = 1;
    transPerm[2] = 3;
    transPerm[3] = 2;
    reshapeIn[0] = C;
    reshapeIn[1] = H * W;
    reshapeIn[2] = N;
    reverseAxis[0] = 1;
    reshapeOut[0] = C;
    reshapeOut[1] = H;
    reshapeOut[2] = W;
    reshapeOut[3] = N;
  }
}

static Status
GenerateTransposeNode(ge::ComputeGraph &graph, ge::GeTensorDesc &prevOutDesc,
                      ge::GeTensorDesc &nextInDesc, const vector<int64_t> &perm,
                      ge::NodePtr &transposeNode, const std::string &basename) {
  vector<int64_t> nextInShape(4);
  for (size_t i = 0; i < perm.size(); ++i) {
    nextInShape[i] = prevOutDesc.GetShape().GetDim(perm[i]);
  }
  ge::OpDescPtr transposeDesc;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
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

static Status
GenerateReshapeNode(ge::ComputeGraph &graph, ge::GeTensorDesc &prevOutDesc,
                    ge::GeTensorDesc &nextInDesc, const vector<int64_t> &shape,
                    ge::NodePtr &shapeNode, const std::string &name,
                    const std::string &basename) {
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

static Status
GenerateReverseNode(ge::ComputeGraph &graph, ge::GeTensorDesc &prevOutDesc,
                    ge::GeTensorDesc &nextInDesc, const vector<int64_t> &axis,
                    ge::NodePtr &reverseNode, const std::string &basename) {
  ge::OpDescPtr reverseDesc;
  FUSION_PASS_MAKE_SHARED((reverseDesc = std::make_shared<ge::OpDesc>(
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

static Status
GenerateReFormatNode(ge::ComputeGraph &graph, ge::GeTensorDesc &prevOutDesc,
                     ge::GeTensorDesc &nextInDesc, const ge::Format &format,
                     ge::NodePtr &reformatNode, const std::string &basename) {
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

Status DeconvWeightTransFusionPass::Relink(ge::NodePtr filterNode, ge::NodePtr dimCompNode,
                     ge::NodePtr transposeNode, ge::NodePtr reformatNode,
                     ge::NodePtr reshapeInNode, ge::NodePtr reverseNode,
                     ge::NodePtr reshapeOutNode, ge::NodePtr deconvNode) {
  // weight -> Deconvolution
  // weight -> [dimComp] -> transpose -> [reshapeIn -> reverse -> reshapeOut] ->
  // deconv
  int filterAnchor = 1;
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(
               filterNode->GetOutDataAnchor(0),
               deconvNode->GetInDataAnchor(filterAnchor)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to remove edge between filterNode and deconvNode"),
           return FAILED);

  OutDataAnchorPtr dimCompOutAnchor = nullptr;
  if (dimCompNode != nullptr) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(filterNode->GetOutDataAnchor(0),
                                     dimCompNode->GetInDataAnchor(0)) !=
                 SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between filterNode and dimCompNode"),
             return FAILED);
    dimCompOutAnchor = dimCompNode->GetOutDataAnchor(0);
  } else {
    dimCompOutAnchor = filterNode->GetOutDataAnchor(0);
  }

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               dimCompOutAnchor, transposeNode->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between dimCompNode and transposeNode"),
           return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0),
                                   reformatNode->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between dimCompNode and transposeNode"),
           return FAILED);

  if (reshapeInNode != nullptr) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reformatNode->GetOutDataAnchor(0),
                                reshapeInNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between transposeNode and reshapeInNode"),
        return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshapeInNode->GetOutDataAnchor(0),
                                     reverseNode->GetInDataAnchor(0)) !=
                 SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between reshapeInNode and reverseNode"),
             return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reverseNode->GetOutDataAnchor(0),
                                     reshapeOutNode->GetInDataAnchor(0)) !=
                 SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between reverseNode and reshapeOutNode"),
             return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
                 reshapeOutNode->GetOutDataAnchor(0),
                 deconvNode->GetInDataAnchor(filterAnchor)) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between reshapeOutNode and deconvNode"),
             return FAILED);
  } else {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
                 reformatNode->GetOutDataAnchor(0),
                 deconvNode->GetInDataAnchor(filterAnchor)) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to add edge between transposeNode and deconvNode"),
             return FAILED);
  }

  FUSION_PASS_CHECK(deconvNode->GetOpDesc()->UpdateInputDesc(
               filterAnchor, reformatNode->GetOpDesc()->GetOutputDesc(0)) !=
               SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to update input description of deconv"),
           return FAILED);

  return SUCCESS;
}

Status DeconvWeightTransFusionPass::Fusion(ge::ComputeGraph &graph,
                                           Mapping &mapping,
                                           vector<ge::NodePtr> &fusionNodes) {
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
  if (filterNode->GetType() != CONSTANT &&
      filterNode->GetType() != CONSTANTOP) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The type of weight is not constant.");
    return NOT_CHANGED;
  }

  int64_t N = 0, C = 0, H = 0, W = 0;
  ge::GeTensorDesc deconvWeightInDesc =
      deconvNode->GetOpDesc()->GetInputDesc(filterAnchor);
  ge::GeTensorDesc filterOutDesc =
      filterNode->GetOpDesc()->GetOutputDesc(filterIdx);
  ge::GeShape filterShape = filterOutDesc.GetShape();
  ge::Format filterFormat = filterOutDesc.GetFormat();
  FUSION_PASS_CHECK(GetShapeByFormat(filterFormat, filterShape, N, C, H, W) != SUCCESS,
           OP_LOGW(FUSED_OP_TYPE.c_str(), "Not support this format %d.", filterFormat),
           return NOT_CHANGED);

  vector<int64_t> dimComp(4), reshapeIn(3), transPerm(4), reverseAxis(1),
      reshapeOut(4);
  GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
      filterFormat, {N, C, H, W}, dimComp, reshapeIn, transPerm, reverseAxis,
      reshapeOut);

  ge::NodePtr dimCompNode = nullptr, transposeNode = nullptr,
              reformatNode = nullptr, reshapeInNode = nullptr,
              reverseNode = nullptr, reshapeOutNode = nullptr;

  auto basename = filterNode->GetName();
  // 1. dimension completion
  if (filterShape.GetDimNum() != 4) {
    FUSION_PASS_CHECK(GenerateReshapeNode(graph, filterOutDesc, deconvWeightInDesc,
                                 dimComp, dimCompNode, "dimension_completion",
                                 basename) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate dimension completion node"),
             return FAILED);
  }
  ge::GeTensorDesc dimCompOutDesc;
  if (dimCompNode != nullptr) {
    dimCompOutDesc = dimCompNode->GetOpDesc()->GetOutputDesc(0);
  } else {
    dimCompOutDesc = filterOutDesc;
  }

  // 2. transpose N,C
  FUSION_PASS_CHECK(GenerateTransposeNode(graph, dimCompOutDesc, deconvWeightInDesc,
                                 transPerm, transposeNode, basename) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate transpose node"), return FAILED);
  ge::GeTensorDesc transposeOutDesc =
      transposeNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(GenerateReFormatNode(graph, transposeOutDesc, deconvWeightInDesc,
                                filterFormat, reformatNode,
                                basename) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reformat node"), return FAILED);
  ge::GeTensorDesc reformatDesc = reformatNode->GetOpDesc()->GetOutputDesc(0);

  if (H != 1 || W != 1) {
    // 3. fuse H, W
    FUSION_PASS_CHECK(GenerateReshapeNode(graph, reformatDesc, deconvWeightInDesc,
                                 reshapeIn, reshapeInNode, "reshape_in",
                                 basename) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reshape in node"), return FAILED);
    ge::GeTensorDesc reshapeInOutDesc =
        reshapeInNode->GetOpDesc()->GetOutputDesc(0);

    // 4. reverse H*W
    FUSION_PASS_CHECK(GenerateReverseNode(graph, reshapeInOutDesc, deconvWeightInDesc,
                                 reverseAxis, reverseNode, basename) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reverse node"), return FAILED);
    ge::GeTensorDesc reverseOutDesc =
        reverseNode->GetOpDesc()->GetOutputDesc(0);

    // 5. anti-fusion H*W
    FUSION_PASS_CHECK(GenerateReshapeNode(graph, reverseOutDesc, deconvWeightInDesc,
                                 reshapeOut, reshapeOutNode, "reshape_out",
                                 basename) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate reshape out node"), return FAILED);
  }

  FUSION_PASS_CHECK(Relink(filterNode, dimCompNode, transposeNode, reformatNode,
                  reshapeInNode, reverseNode, reshapeOutNode,
                  deconvNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End DeconvWeightTransFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("DeconvWeightTransFusionPass", BUILT_IN_GRAPH_PASS,
              DeconvWeightTransFusionPass);
} // namespace fe
