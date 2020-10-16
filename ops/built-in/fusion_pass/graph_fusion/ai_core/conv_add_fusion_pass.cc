/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  biasadd_conv_fusion_pass.cpp
 *
 * @brief conv-add fusion pass(conv + add --> conv)
 *
 */

#include "conv_add_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const string PATTERN_SRC = "src";
static const string PATTERN_BIASADD = "Convadd";
static const char *CONVOLUTION_3D = "Conv3D";
static const char *CONVOLUTION_2D = "Conv2D";
static const char *ADD_3D = "Add";
static const std::set<string> NEW_ADD_IN = {"Const", "Constant", "Mul"};
static const int64_t DIM1 = 1;
static const int64_t DIM_COUNT = 4;
static const int64_t BIAS_INDEX = 2;

vector<FusionPattern *> ConvaddFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  string passName = "TBEConv3daddFusion";
  FusionPattern *pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "TBEConv3daddFusion new an object failed."),
           return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_BIASADD, {ADD_3D})
          .AddOpDesc(PATTERN_SRC, {CONVOLUTION_3D, CONVOLUTION_2D})
          .SetInputs(PATTERN_BIASADD, {PATTERN_SRC})
          .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConvaddFusionPass::Fusion(ge::ComputeGraph &graph,
                                     Mapping &mapping,
                                     vector<ge::NodePtr> &fusionNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter ConvaddFusionPass");
  ge::NodePtr src_node = GetNodeFromMapping(PATTERN_SRC, mapping);
  ge::NodePtr biasadd_node = GetNodeFromMapping(PATTERN_BIASADD, mapping);
  FUSION_PASS_CHECK(src_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node Conv3d is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(biasadd_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node Add is null, fusion failed."),
           return PARAM_INVALID);

  if (src_node->GetOutDataNodes().size() > 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "out data size is invalid.");
    return NOT_CHANGED;
  }

  ge::OpDescPtr src_op = src_node->GetOpDesc();
  FUSION_PASS_CHECK(src_op == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", src_node->GetName().c_str()),
           return PARAM_INVALID);

  // get conv2d's weights & bias
  bool hasBias = true;
  if (src_op->GetType() == "Conv2D") {
    int64_t in_edges_size = src_node->GetInDataNodes().size();
    OP_LOGI(FUSED_OP_TYPE.c_str(), "op [%s]: Conv2d has %d input nodes.",
          src_node->GetName().c_str(), in_edges_size);
    if (in_edges_size <= BIAS_INDEX) {
      hasBias = false;
    }
    FUSION_PASS_CHECK(hasBias,
            OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv_add graph_fusion not support with_bias for now."),
            return NOT_CHANGED);
    OutDataAnchorPtr filterAnchor = src_node->GetInDataAnchor(1)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(filterAnchor == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "filter output anchor is null"),
            return PARAM_INVALID);
    NodePtr filterNode = filterAnchor->GetOwnerNode();
    FUSION_PASS_CHECK(filterNode == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DAddFusionPass: filterNode is not exist."),
            return PARAM_INVALID);
    FUSION_PASS_CHECK(NEW_ADD_IN.find(filterNode->GetOpDesc()->GetType()) == NEW_ADD_IN.end(),
              OP_LOGW(FUSED_OP_TYPE.c_str(),"middle layer's of filterNode is not const type"),
              return false);
    if (hasBias) {
        OutDataAnchorPtr biasAnchor = src_node->GetInDataAnchor(2)->GetPeerOutAnchor();
        FUSION_PASS_CHECK(biasAnchor == nullptr,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "bias anchor is null"),
                return PARAM_INVALID);
        auto biasNode = biasAnchor->GetOwnerNode();
        FUSION_PASS_CHECK(biasNode == nullptr,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DAddFusionPass: biasNode is not exist."),
                return PARAM_INVALID);
        FUSION_PASS_CHECK(NEW_ADD_IN.find(biasNode->GetOpDesc()->GetType()) == NEW_ADD_IN.end(),
                OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's of biasNode is not const type"),
                return false);
    }
  }

  vector<ge::ConstGeTensorPtr> weights =
      ge::OpDescUtils::GetWeights(biasadd_node);
  auto biasAddWeightSize = weights.size();
  if (biasAddWeightSize == 0 && biasadd_node->GetType() == ADD_3D &&
      (src_node->GetType() == CONVOLUTION_3D ||
       src_node->GetType() == CONVOLUTION_2D)) {
    bool checkNull = biasadd_node->GetInDataAnchor(1) == nullptr ||
        biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor() == nullptr ||
        biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() == nullptr;

    FUSION_PASS_CHECK(checkNull, OP_LOGI(FUSED_OP_TYPE.c_str(), "The input of add %s is null!",
                                biasadd_node->GetName().c_str()),
             return NOT_CHANGED);
    auto nodeInfrontOfAdd =
        biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();

    if (nodeInfrontOfAdd->GetType() == "Reshape") {
      /* This case, the BiasAdd is Add and the input of Add is Reshape,
       * we just get the first weight of reshape as the bias. */
      weights = ge::OpDescUtils::GetWeights(nodeInfrontOfAdd);
      FUSION_PASS_CHECK(weights.empty(),
               OP_LOGI(FUSED_OP_TYPE.c_str(), "Node Add:[%s]'s weight size %u is invalid.",
                       nodeInfrontOfAdd->GetName().c_str(), weights.size()),
               return NOT_CHANGED);
    } else {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The input of biasadd %s is invalid.",
              biasadd_node->GetName().c_str());
      return NOT_CHANGED;
    }
  } else {
    FUSION_PASS_CHECK(biasAddWeightSize != 1,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Node BiasAdd:[%s]'s weight size %u is invalid.",
                     biasadd_node->GetName().c_str(), biasAddWeightSize),
             return NOT_CHANGED);
    /* The weights will be the weight of BiasAdd node */
  }

  Status result = PatternFusionUtil::CopyMultiReferenceConstNode(graph, src_node);
  FUSION_PASS_CHECK(result != SUCCESS,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Src_node[%s]: can not copy multiReference const node.",
                    src_node->GetName().c_str()),
           return result);

  std::map<string, uint32_t> inputNameMap = src_op->GetAllInputName();
  ge::OpDescPtr biasadd_op = biasadd_node->GetOpDesc();
  FUSION_PASS_CHECK(biasadd_op == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", biasadd_node->GetName().c_str()),
           return PARAM_INVALID);

  GeTensorDesc inputDesc0 = biasadd_op->GetInputDesc(0);
  GeTensorDesc inputDesc1 = biasadd_op->GetInputDesc(1);
  auto dims0 = inputDesc0.GetShape().GetDims();
  auto dims1 = inputDesc1.GetShape().GetDims();

  if (src_op->GetType() == "Conv2D") {
    if (dims1.size() == 1) {
        if (dims0.at(3) != dims1.at(0)) {
            return NOT_CHANGED;
        }
    } else if (dims1.size() == 4) {
        // NHWC, channel wise  exemple: add input [1,1,1,256] and conv input [x,x,x,256]
        if (dims1.at(0) != 1 or dims1.at(1) != 1 or dims1.at(2) != 1 or\
             dims0.at(3) != dims1.at(3)) {
          OP_LOGW(FUSED_OP_TYPE.c_str(), "Match failed! tensor is not ChannelWise");
          return NOT_CHANGED;
        }
    }
  } else if (src_op->GetType() == "Conv3D") {
   if (dims1.size() == 1) {
      if (dims0.at(4) != dims1.at(0)) {
          return NOT_CHANGED;
      }
  } else if (dims1.size() == 5) {
      // NDHWC, channel wise  exemple: add input [1,1,1,1,256] and conv input [x,x,x,x,256]
      if (dims1.at(0) != 1 or dims1.at(1) != 1 or dims1.at(2) != 1 or dims1.at(3) != 1 or dims0.at(4) != dims1.at(4)) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Match failed! tensor is not ChannelWise");
        return NOT_CHANGED;
      }
    }
  }

  bool ret;
  ret = ge::OpDescUtils::ClearInputDesc(src_op, 3);
  if (ret) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvaddFusionPass: clear Conv3d [%s]'s 4nd input desc.",src_node->GetName().c_str());
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(),  "Fail to ConvaddFusionPass: clear Conv3d [%s]'s 4nd input desc.",src_node->GetName().c_str());
  }

  ret = ge::OpDescUtils::ClearInputDesc(src_op, 2);
  if (ret) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvaddFusionPass: clear Conv3d [%s]'s 3nd input desc.", src_node->GetName().c_str());
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to ConvaddFusionPass: clear Conv3d [%s]'s 3nd input desc.", src_node->GetName().c_str());
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvaddFusionPass: Conv3d [%s] has %u input anchor.", src_node->GetName().c_str(), src_node->GetAllInDataAnchors().size());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvaddFusionPass: Conv3d [%s] has %u input desc.", src_node->GetName().c_str(), src_op->GetAllInputsDesc().size());
  int64_t in_edges_size = src_node->GetInDataNodes().size();
  if (in_edges_size < 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inEdges size is invalid.");
    return NOT_CHANGED;
  }

  ge::ConstGeTensorPtr biases = weights[0];
  FUSION_PASS_CHECK(biases == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Biasadd node's weight is null, fusion failed."),
           return PARAM_INVALID);

  int64_t dim1Count = 0;
  int64_t newShape = 1;

  FUSION_PASS_CHECK(biases->GetTensorDesc().GetShape().GetDims().size() == 0,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Dims size is invalid."),
           return NOT_CHANGED);

  if (biases->GetTensorDesc().GetShape().GetDims().size() != 1) {
    for (int64_t dim : biases->GetTensorDesc().GetShape().GetDims()) {
      if (dim == DIM1) {
        dim1Count++;
      } else {
        newShape = dim;
      }
    }
    if (dim1Count < DIM_COUNT) {
      return NOT_CHANGED;
    }
  } else {
    newShape = biases->GetTensorDesc().GetShape().GetDims().at(0);
  }

  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::LinkControlEdge(biasadd_node, src_node),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Link control edge from [%s] to [%s] failed", biasadd_node->GetName().c_str(), src_node->GetName().c_str()),
           return FAILED);

  vector<ge::GeTensorPtr> src_weights_vec =
          ge::OpDescUtils::MutableWeights(src_node);

  ge::GeTensorPtr src_bias_ptr = nullptr;
  src_bias_ptr = std::make_shared<ge::GeTensor>(biases->Clone());

  FUSION_PASS_CHECK(src_bias_ptr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Clone Biasadd node's weight is null, fusion failed."),
           return PARAM_INVALID);

  std::vector<int64_t> newDimVec;
  newDimVec.push_back(newShape);
  ge::GeShape biasShape(newDimVec);
  src_bias_ptr->MutableTensorDesc().SetShape(biasShape);
  src_bias_ptr->MutableTensorDesc().SetOriginShape(biasShape);
  ge::Format inputFormat = biasadd_op->GetInputDesc(0).GetFormat();
  src_bias_ptr->MutableTensorDesc().SetFormat(inputFormat);
  inputFormat = biasadd_op->GetInputDesc(0).GetOriginFormat();
  src_bias_ptr->MutableTensorDesc().SetOriginFormat(inputFormat);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "bias's format is %d, origin format is %d.",
          src_bias_ptr->MutableTensorDesc().GetFormat(),
          src_bias_ptr->MutableTensorDesc().GetOriginFormat());

  OP_LOGI(FUSED_OP_TYPE.c_str(), "size of src node inputNameMap is %u", inputNameMap.size());
  src_weights_vec.push_back(src_bias_ptr);
  ge::OpDescUtils::SetWeights(src_node, src_weights_vec);
  FUSION_PASS_CHECK(true != ge::AttrUtils::SetBool(src_node->GetOpDesc(), ge::MATMUL_HAS_BIAS, true),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Biasadd op weight should be 1-D."), return FAILED);
  GeTensorDesc convbiasTensor = src_op->GetInputDesc(2);
  GeTensorDesc inputTensor = src_op->GetInputDesc(0);
  convbiasTensor.SetOriginShape(convbiasTensor.GetShape());
  convbiasTensor.SetOriginDataType(convbiasTensor.GetDataType());
  convbiasTensor.SetOriginFormat(inputTensor.GetOriginFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input datatype is %d.", src_op->GetInputDesc(2).GetDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input origin datatype is %d.", src_op->GetInputDesc(2).GetOriginDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input format is %d.", src_op->GetInputDesc(2).GetFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input origin format is %d.", src_op->GetInputDesc(2).GetOriginFormat());
  src_op->UpdateInputDesc(2, convbiasTensor);

  ge::GeTensorDesc offsetDesc;
  offsetDesc.SetDataType(ge::DT_UNDEFINED);
  offsetDesc.SetFormat(ge::FORMAT_RESERVED);
  src_op->AddInputDesc(offsetDesc);

  FUSION_PASS_CHECK(false == src_op->UpdateInputName(inputNameMap),
           OP_LOGW(FUSED_OP_TYPE.c_str(), "UpdateInputName conv failed."),
           return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input datatype is %d.", src_op->GetInputDesc(2).GetDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input origin datatype is %d.", src_op->GetInputDesc(2).GetOriginDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input format is %d.", src_op->GetInputDesc(2).GetFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3d's 2nd input origin format is %d.", src_op->GetInputDesc(2).GetOriginFormat());


  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(biasadd_node),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node:[%s] failed", biasadd_node->GetName().c_str()),
           return FAILED);
  fusionNodes.push_back(src_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvaddFusionPass fusion success.");

  return SUCCESS;
}

REGISTER_PASS("TBEConv3daddFusion", BUILT_IN_GRAPH_PASS, ConvaddFusionPass);
}
