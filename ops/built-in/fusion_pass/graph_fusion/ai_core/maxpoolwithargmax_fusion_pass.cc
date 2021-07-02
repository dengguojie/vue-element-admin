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
 * \file maxpoolwithargmax_fusion_pass.cpp
 * \brief MaxPoolWithArgmax fusion pass
 *   (MaxPoolWithArgmax --> MaxPoolWithArgmax & Mask2Argmax)
 */
#include "maxpoolwithargmax_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "MaxPoolWithArgmax";

static const std::string PATTERN_FUSEDNODE = "MaxPoolWithArgmax";

static const int32_t INT_NUM_ZERO = 0;

vector<FusionPattern*> MaxPoolWithArgmaxFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("MaxPoolWithArgmaxFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

int64_t CeilDev(int64_t value, int64_t factor) {
  int64_t value_num = 0;
  if (value % factor == 0) {
    value_num = value / factor;
  } else {
    value_num = value / factor + 1;
  }
  return value_num;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status MaxPoolWithArgmaxFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                           vector<ge::NodePtr>& fusionNodes) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(fusedDesc);
  ge::GeTensorDesc tmpMaskDesc = fusionDesc->GetOutputDesc(1);
  tmpMaskDesc.SetDataType(ge::DT_UINT16);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != fusionDesc->UpdateOutputDesc("argmax", tmpMaskDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc node %s failed", fusionDesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  ge::GeTensorDesc outputMaskDesc = fusedDesc->GetOutputDesc(1);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  ge::GeTensorDesc inputTensorDesc = fusedDesc->GetInputDesc(0);
  Format input_format = inputTensorDesc.GetFormat();
  ge::GeShape shape = inputTensorDesc.GetShape();
  int64_t in_size_n = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;
  int64_t in_size_c = 0;
  int64_t in_size_c1 = 0;
  if (input_format == ge::FORMAT_NHWC) {
    in_size_n = shape.GetDim(0);
    in_size_h = shape.GetDim(1);
    in_size_w = shape.GetDim(2);
    in_size_c = shape.GetDim(3);
  } else {
    in_size_n = shape.GetDim(0);
    in_size_c = shape.GetDim(1);
    in_size_h = shape.GetDim(2);
    in_size_w = shape.GetDim(3);
  }
  int64_t BLOCKSIZE = 16;
  in_size_c1 = CeilDev(in_size_c, BLOCKSIZE);
  // get input ksize
  std::vector<int64_t> ksizeList;
  if (op.GetAttr("ksize", ksizeList) != ge::GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "GetOpAttr ksizeList failed!");
    return NOT_CHANGED;
  }

  if (ksizeList.size() != DIM_SIZE4) {
    OP_LOGW(op.GetName().c_str(),
            "length of ksize must be equal to"
            "the length of shape!");
    return NOT_CHANGED;
  }

  // get input strides
  std::vector<int64_t> stridesList;
  if (op.GetAttr("strides", stridesList) != ge::GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "GetOpAttr stridesList failed!");
    return NOT_CHANGED;
  }

  if (stridesList.size() != DIM_SIZE4) {
    OP_LOGW(op.GetName().c_str(),
            "length of strides must be equal to"
            "the length of shape!");
    return NOT_CHANGED;
  }
  if ((ksizeList[0] != 1) || (ksizeList[3] != 1) || (stridesList[0] != 1) || (stridesList[3] != 1)) {
    OP_LOGW(op.GetName().c_str(),
            "MaxPoolWithArgmax only supports pooling "
            "across width/height, and other ksize "
            "dimension should be one");
    return NOT_CHANGED;
  }
  if ((ksizeList[1] * ksizeList[2]) > 255) {
    OP_LOGW(op.GetName().c_str(),
            "invalid window params, window_h*window_w "
            "should be <= 255");
    return NOT_CHANGED;
  }
  // get input paddingMode
  std::string paddingMode;
  if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "GetOpAttr padding failed!");
    return NOT_CHANGED;
  }

  if (paddingMode != "SAME" && paddingMode != "VALID") {
    OP_LOGW(op.GetName().c_str(),
            "MaxPoolWithArgmax can only support"
            "SAME or VALID padding mode!");
    return NOT_CHANGED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  for (auto dim : dims_input) {
    if (PatternFusionUtil::IsUnknownShape(dim)) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "MaxPoolWithArgmaxFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  }
  // set output max shape
  std::vector<int64_t> dimVector;
  if (input_format == ge::FORMAT_NHWC) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (stridesList[i] <= INT_NUM_ZERO) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Stride must be > 0.");
            return FAILED; 
          }
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE1) || (i == DIM_SIZE2)) {
          if (stridesList[i] <= INT_NUM_ZERO) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Stride must be > 0.");
            return FAILED;
          }
          int64_t dims = (dims_input[i] - ksizeList[i] + 1 + (stridesList[i] - 1)) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  } else {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (stridesList[i - 1] <= INT_NUM_ZERO) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Stride must be > 0.");
            return FAILED;
          }
          int64_t dims = (dims_input[i] + stridesList[i - 1] - 1) / stridesList[i - 1];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if ((i == DIM_SIZE2) || (i == DIM_SIZE3)) {
          if (stridesList[i - 1] <= INT_NUM_ZERO) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Stride must be > 0.");
            return FAILED;
          }
          int64_t dims = (dims_input[i] - ksizeList[i - 1] + 1 + (stridesList[i - 1] - 1)) / stridesList[i - 1];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    }
  }
  // set output mask shape
  std::vector<int64_t> dimVectorMask;
  if (input_format == ge::FORMAT_NHWC) {
    for (size_t i = 0; i < dims_input.size(); i++) {
      if (i == DIM_SIZE1) {
        int64_t dims = ksizeList[i] * ksizeList[i + 1];
        dimVectorMask.push_back(dims);
      } else if (i == DIM_SIZE2) {
        int64_t dimsTmp = CeilDev(dimVector[i - 1] * dimVector[i], BLOCKSIZE);
        int64_t dims = dimsTmp + 1;
        dimVectorMask.push_back(dims);
      } else {
        int64_t dims = dims_input[i];
        dimVectorMask.push_back(dims);
      }
    }
  } else {
    for (size_t i = 0; i < dims_input.size(); i++) {
      if (i == DIM_SIZE2) {
        int64_t dims = ksizeList[i - 1] * ksizeList[i];
        dimVectorMask.push_back(dims);
      } else if (i == DIM_SIZE3) {
        int64_t dimsTmp = CeilDev(dimVector[i - 1] * dimVector[i], BLOCKSIZE);
        int64_t dims = dimsTmp + 1;
        dimVectorMask.push_back(dims);
      } else {
        int64_t dims = dims_input[i];
        dimVectorMask.push_back(dims);
      }
    }
  }
  ge::GeShape outputMaskShape(dimVectorMask);
  outputMaskDesc.SetOriginShape(outputMaskShape);
  outputMaskDesc.SetShape(outputMaskShape);
  outputMaskDesc.SetOriginDataType(ge::DT_UINT16);
  outputMaskDesc.SetDataType(ge::DT_UINT16);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of MaxpoolWithArgmaxNode is [%d].",
          fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size());
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != fusedDesc->UpdateOutputDesc("argmax", outputMaskDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc node %s failed", fusedDesc->GetName().c_str()),
                    return FAILED);
  if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of MaxpoolWithArgmaxNode is [%d].",
            fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr outMaskAnchorPtr : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      ge::NodePtr fusedNextNode = outMaskAnchorPtr->GetOwnerNode();
      if ((fusedNextNode->GetType() == "MaxPoolGradWithArgmax") || (fusedNextNode->GetType() == "MaxPoolGradGradWithArgmax")) {
        ge::OpDescPtr fusedNextDesc = fusedNextNode->GetOpDesc();
        ge::GeTensorDesc gradInputMaskDesc = fusedNextDesc->GetInputDesc(2);
        gradInputMaskDesc.SetOriginShape(outputMaskShape);
        gradInputMaskDesc.SetShape(outputMaskShape);
        gradInputMaskDesc.SetOriginDataType(ge::DT_UINT16);
        gradInputMaskDesc.SetDataType(ge::DT_UINT16);
        FUSION_PASS_CHECK(
            ge::GRAPH_SUCCESS != fusedNextDesc->UpdateInputDesc("argmax", gradInputMaskDesc),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc node %s failed", fusedNextDesc->GetName().c_str()),
            return FAILED);
      }
      if ((fusedNextNode->GetType() != "MaxPoolGradWithArgmax") && (fusedNextNode->GetType() != "MaxPoolGradGradWithArgmax")) {
        ge::OpDescPtr Mask2ArgmaxDesc = AttrUtils::CloneOpDesc(fusedDesc);
        FUSION_PASS_CHECK(
            Mask2ArgmaxDesc == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedDesc->GetName().c_str()),
            return PARAM_INVALID);
        Mask2ArgmaxDesc->SetName(fusedDesc->GetName() + "/Mask2Argmax");
        Mask2ArgmaxDesc->SetType("Mask2Argmax");
        GeTensorDesc inputMaskDesc = fusedDesc->GetOutputDesc(1);
        OpDescUtils::ClearOutputDesc(Mask2ArgmaxDesc, 1);
        Mask2ArgmaxDesc->AddInputDesc(1, inputMaskDesc);
        GeTensorDesc outputMaxDesc = Mask2ArgmaxDesc->GetOutputDesc(0);
        outputMaxDesc.SetDataType(ge::DT_FLOAT);
        outputMaxDesc.SetOriginDataType(ge::DT_FLOAT);
        FUSION_PASS_CHECK(
            ge::GRAPH_SUCCESS != Mask2ArgmaxDesc->UpdateOutputDesc(0, outputMaxDesc),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc node %s failed", Mask2ArgmaxDesc->GetName().c_str()),
            return FAILED);
        ge::NodePtr Mask2ArgmaxNode = graph.AddNode(Mask2ArgmaxDesc);
        FUSION_PASS_CHECK(
            Mask2ArgmaxNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", Mask2ArgmaxDesc->GetName().c_str()),
            return PARAM_INVALID);
        Operator op_mask2argmax = ge::OpDescUtils::CreateOperatorFromNode(Mask2ArgmaxNode);
        // set attr of Mask2Argmax:originshape
        std::vector<int64_t> originshape;
        originshape.push_back(in_size_n);
        originshape.push_back(in_size_h);
        originshape.push_back(in_size_w);
        originshape.push_back(in_size_c);
        op_mask2argmax.SetAttr("originshape", originshape);
        /* Add new Opdesc of MaxPoolWithArgmax*/
        std::shared_ptr<ge::OpDesc> transDataOpdesc = nullptr;
        std::string transDataNodeName = fusedDesc->GetName() + "/TransData";
        FUSION_PASS_MAKE_SHARED(
            (transDataOpdesc = std::make_shared<ge::OpDesc>(transDataNodeName, "TransData")),
            return PARAM_INVALID);
        GeTensorDesc outputTransDataDesc = Mask2ArgmaxDesc->GetOutputDesc(0);
        outputTransDataDesc.SetFormat(input_format);
        // set src shape
        int64_t out_size_h = 0;
        int64_t out_size_w = 0;
        std::string tarnsDataSrcFormat = "NC1HWC0";
        std::string tarnsDataDstFormat = "";
        if (input_format == ge::FORMAT_NHWC) {
          tarnsDataDstFormat = "NHWC";
          out_size_h = dimVector[1];
          out_size_w = dimVector[2];
        } else {
          tarnsDataDstFormat = "NCHW";
          out_size_h = dimVector[2];
          out_size_w = dimVector[3];
        }
        std::vector<int64_t> input_tshape;
        input_tshape.push_back(in_size_n);
        input_tshape.push_back(in_size_c1);
        input_tshape.push_back(out_size_h);
        input_tshape.push_back(out_size_w);
        input_tshape.push_back(BLOCKSIZE);
        ge::GeShape inputTransDataShape(input_tshape);
        ge::GeTensorDesc inputTransDataDesc = ge::GeTensorDesc(inputTransDataShape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
        inputTransDataDesc.SetShape(inputTransDataShape);
        inputTransDataDesc.SetOriginFormat(outputMaskDesc.GetOriginFormat());
        inputTransDataDesc.SetOriginShape(outputMaxDesc.GetOriginShape());
        inputTransDataDesc.SetOriginDataType(outputMaxDesc.GetOriginDataType());
        transDataOpdesc->AddInputDesc("src", inputTransDataDesc);
        transDataOpdesc->AddOutputDesc("dst", outputTransDataDesc);
        /* Add Node into graph */
        ge::NodePtr transDataNode = graph.AddNode(transDataOpdesc);
        FUSION_PASS_CHECK(
            transDataNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", transDataOpdesc->GetName().c_str()),
            return PARAM_INVALID);
        Operator op_transdata = ge::OpDescUtils::CreateOperatorFromNode(transDataNode);
        op_transdata.SetAttr("src_format", tarnsDataSrcFormat);
        op_transdata.SetAttr("dst_format", tarnsDataDstFormat);
        // link MaxPoolWithArgmax->Mask2Argmax->TransData
        if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
          OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of MaxPoolWithArgmax is [%d].",
                  fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size());
          for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
            inAnchorPtr->UnlinkAll();
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(transDataNode->GetOutDataAnchor(0), inAnchorPtr),
                              OP_LOGE(FUSED_OP_TYPE.c_str(),
                                      "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                                      transDataOpdesc->GetName().c_str(), fusedDesc->GetName().c_str()),
                              return FAILED);
            OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
                    transDataOpdesc->GetName().c_str(), fusedDesc->GetName().c_str());
          }
        }
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                             Mask2ArgmaxNode->GetInDataAnchor(0)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                  fusedDesc->GetName().c_str(), Mask2ArgmaxDesc->GetName().c_str()),
                          return FAILED);
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetOutDataAnchor(1), Mask2ArgmaxNode->GetInDataAnchor(1)),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                    fusedDesc->GetName().c_str(), Mask2ArgmaxDesc->GetName().c_str()),
            return FAILED);
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(Mask2ArgmaxNode->GetOutDataAnchor(0), transDataNode->GetInDataAnchor(0)),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                    Mask2ArgmaxDesc->GetName().c_str(), transDataOpdesc->GetName().c_str()),
            return FAILED);
      }
    }
  }
  return SUCCESS;
}
REGISTER_PASS("MaxPoolWithArgmaxFusionPass", BUILT_IN_GRAPH_PASS, MaxPoolWithArgmaxFusionPass);
}  // namespace fe
