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
 * \file batch_multi_class_nms_fusion_pass.cpp
 * \brief batch_multi_class_nms fusion pass
 */
#include "batch_multi_class_nms_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const int32_t INT_NUM_THREE = 3;
static const int32_t INT_NUM_TWO = 2;
static const string PATTERN_FUSEDNODE = "FusedNodeBatchMultiClassNonMaxSuppression";
static const string FUSED_NODE = "BatchMultiClassNonMaxSuppression";

vector<FusionPattern*> BatchMultiClassNonMaxSuppressionFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchMultiClassNonMaxSuppressionFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

bool BatchMultiClassNonMaxSuppressionFusionPass::CheckTransposeBeforeSlice(ge::NodePtr checkNode) {
  auto checkOpType = checkNode->GetType();
  FUSION_PASS_CHECK((checkOpType != "Slice") && (checkOpType != "SliceD"),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op name is not Slice or SliceD, is %s", checkOpType.c_str()),
                    return false);
  if (checkNode->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the Slice op have more than one output");
    return false;
  }
  return true;
}

bool BatchMultiClassNonMaxSuppressionFusionPass::CheckfindSoftmax(ge::NodePtr checkNode) {
  auto ExpandDims_OpType = checkNode->GetType();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "checkNode, is %s", ExpandDims_OpType.c_str());
  FUSION_PASS_CHECK((ExpandDims_OpType != "ExpandDims"),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op name is not ExpandDims, is %s", ExpandDims_OpType.c_str()),
                    return false);
  return true;
}


Status BatchMultiClassNonMaxSuppressionFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                          vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionFusionPass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc boxesInputDesc = fusedNode->GetOpDesc()->GetInputDesc(0);
  vector<int64_t> boxesInputShape = boxesInputDesc.GetShape().GetDims();
  auto boxesSize = boxesInputShape.size();
  bool isNeedTranposeBeforeScore = true;
  vector<int64_t> permBoxesList;

  // insert transpose at input 0
  if (boxesSize != 4) {
    isNeedTranposeBeforeScore = false;
    permBoxesList = {0, 2, 1};
  } else {
    permBoxesList = {0, 2, 3, 1};
  }
  AddTransposeBeforeNode(fusedNode, 0, permBoxesList, graph);
  // insert transpose at input 1

  // get the input 1 peer node op type
  vector<int64_t> permScoreList = {0, 2, 1};
  auto in_anchor_size = fusedNode->GetAllInDataAnchorsSize();
  if (in_anchor_size < INT_NUM_TWO) {
    return FAILED;
  }
  auto peerNode = fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  auto peerOpType = peerNode->GetType();
  bool isNeedTransposeBeforeSlice = false;
  bool isfindSoftmax = false;
  isNeedTransposeBeforeSlice = CheckTransposeBeforeSlice(peerNode);
  isfindSoftmax = CheckfindSoftmax(peerNode);


  if (isNeedTranposeBeforeScore && isNeedTransposeBeforeSlice) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "will insert transpose before Slice + BatchMultiClassNonMaxSuppression");
    // insert transpose before Slice
    AddTransposeBeforeNode(peerNode, 0, permScoreList, graph);
    if (peerOpType == "Slice") {
      Operator op = ge::OpDescUtils::CreateOperatorFromNode(peerNode);
      DataType dtype = op.GetInputDesc("offsets").GetDataType();

      vector<ge::GeTensorPtr> sliceTensorPtr = ge::OpDescUtils::MutableWeights(peerNode);
      FUSION_PASS_CHECK(sliceTensorPtr.size() < 2, OP_LOGW(FUSED_OP_TYPE.c_str(), "slice const num less then 2!"),
                        return NOT_CHANGED);

      if (dtype == ge::DT_INT32) {
        // modify Slice offset const node value
        ge::GeTensorPtr offsetsTensorPtr = sliceTensorPtr[0];
        int32_t* constOffsetData = (int32_t*)(offsetsTensorPtr->GetData().GetData());
        if (constOffsetData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get Offset Data from const node is NULL.");
          return PARAM_INVALID;
        }

        auto sizeConstOffset = offsetsTensorPtr->GetData().GetSize();
        if (sizeConstOffset < INT_NUM_THREE) {
          return PARAM_INVALID;
        }

        vector<int32_t> offsetsNew = {constOffsetData[0], constOffsetData[2], constOffsetData[1]};
        offsetsTensorPtr->SetData(reinterpret_cast<uint8_t*>(offsetsNew.data()), offsetsNew.size() * sizeof(int32_t));

        // modify Slice size const node value
        ge::GeTensorPtr sizeTensorPtr = sliceTensorPtr[1];
        int32_t* constSizeData = (int32_t*)(sizeTensorPtr->GetData().GetData());
        if (constSizeData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get size Data from const node is NULL.");
          return PARAM_INVALID;
        }
        vector<int32_t> sizeNew = {constSizeData[0], constSizeData[2], constSizeData[1]};
        sizeTensorPtr->SetData(reinterpret_cast<uint8_t*>(sizeNew.data()), sizeNew.size() * sizeof(int32_t));
      } else {
        // modify Slice offset const node value
        ge::GeTensorPtr offsetsTensorPtr = sliceTensorPtr[0];
        int64_t* constOffsetData = (int64_t*)(offsetsTensorPtr->GetData().GetData());
        if (constOffsetData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get Offset Data from const node is NULL.");
          return PARAM_INVALID;
        }
        vector<int64_t> offsetsNew = {constOffsetData[0], constOffsetData[2], constOffsetData[1]};
        offsetsTensorPtr->SetData(reinterpret_cast<uint8_t*>(offsetsNew.data()), offsetsNew.size() * sizeof(int64_t));

        // modify Slice size const node value
        ge::GeTensorPtr sizeTensorPtr = sliceTensorPtr[1];
        int64_t* constSizeData = (int64_t*)(sizeTensorPtr->GetData().GetData());
        if (constSizeData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get size Data from const node is NULL.");
          return PARAM_INVALID;
        }
        vector<int64_t> sizeNew = {constSizeData[0], constSizeData[2], constSizeData[1]};
        sizeTensorPtr->SetData(reinterpret_cast<uint8_t*>(sizeNew.data()), sizeNew.size() * sizeof(int64_t));
      }
    } else {
      vector<int64_t> offsets;
      ge::AttrUtils::GetListInt(peerNode->GetOpDesc(), "offsets", offsets);
      FUSION_PASS_CHECK(offsets.size() < 3, OP_LOGW(FUSED_OP_TYPE.c_str(), "sliceD attr offsets is less then 3!"),
                        return NOT_CHANGED);
      vector<int64_t> size;
      ge::AttrUtils::GetListInt(peerNode->GetOpDesc(), "size", size);
      FUSION_PASS_CHECK(size.size() < 3, OP_LOGW(FUSED_OP_TYPE.c_str(), "sliceD attr size is less then 3!"),
                        return NOT_CHANGED);
      vector<int64_t> offsetsNew = {offsets[0], offsets[2], offsets[1]};
      vector<int64_t> sizeNew = {size[0], size[2], size[1]};
      ge::AttrUtils::SetListInt(peerNode->GetOpDesc(), "offsets", offsetsNew);
      ge::AttrUtils::SetListInt(peerNode->GetOpDesc(), "sizeNew", sizeNew);
    }
    // update Slice out desc
    vector<int64_t> oriSliceOutputShape = peerNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
    FUSION_PASS_CHECK(oriSliceOutputShape.empty(), OP_LOGW(FUSED_OP_TYPE.c_str(), "Slice output shape is nullptr!"),
                      return NOT_CHANGED);
    vector<int64_t> outputSliceShapeVec;
    if (oriSliceOutputShape.size() < INT_NUM_THREE) {
        return FAILED;
    }
    outputSliceShapeVec.push_back(oriSliceOutputShape[0]);
    outputSliceShapeVec.push_back(oriSliceOutputShape[2]);
    outputSliceShapeVec.push_back(oriSliceOutputShape[1]);
    ge::GeShape outputSliceShape(outputSliceShapeVec);
    auto opSliceOutputDesc = peerNode->GetOpDesc()->GetOutputDesc(0);
    opSliceOutputDesc.SetShape(outputSliceShape);
    opSliceOutputDesc.SetOriginShape(outputSliceShape);
    peerNode->GetOpDesc()->UpdateOutputDesc(0, opSliceOutputDesc);
    // update BatchMultiClassNonMaxSuppression input 1 desc
    auto in_anchors_size = fusedNode->GetAllInDataAnchorsSize();
    if (in_anchors_size < INT_NUM_TWO) {
      return FAILED;
    }
    auto inputDesc = fusedNode->GetOpDesc()->GetInputDesc(1);
    inputDesc.SetShape(outputSliceShape);
    inputDesc.SetOriginShape(outputSliceShape);
    fusedNode->GetOpDesc()->UpdateInputDesc(1, inputDesc);
  } else if (isNeedTranposeBeforeScore) {
    if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() == 0) {
      // insert directly before the input 1
      ge::GeTensorDesc nmsScorceDesc = fusedNode->GetOpDesc()->GetInputDesc(1);
      ge::OpDescPtr score_thresholdnode = fusedNode->GetOpDesc();
      float score_threshold_value = 0.0;
      float score_threshold_new = 0.0;

      ge::AttrUtils::GetFloat(score_thresholdnode, "score_threshold", score_threshold_value);
      FUSION_PASS_CHECK((65536.0/65535.0) - score_threshold_value < 1e-6,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "score threshold should not be zero!"),
                        return FAILED);
      score_threshold_new = (1 + score_threshold_value) / ((65536.0/65535.0) - score_threshold_value);
      ge::AttrUtils::SetFloat(score_thresholdnode, "score_threshold", score_threshold_new);
      vector<int64_t> oriNmsScorceShape = nmsScorceDesc.GetShape().GetDims();
      FUSION_PASS_CHECK(oriNmsScorceShape.empty(),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "can not get input nms scorce shape. shape is empty!"),
                        return PARAM_INVALID);

      // Add the description (input, output, name, attribute) of the add1 node
      ge::OpDescPtr Adds;
      std::string add1DescName = fusedNode->GetOpDesc()->GetName();

      // Define node name----->BatchMultiClassNonMaxSuppression_Add1
      FUSION_PASS_MAKE_SHARED((Adds = std::make_shared<ge::OpDesc>(add1DescName + "_Add1", "Adds")),
                              Adds = nullptr;
                              return PARAM_INVALID);

      // Define out auxiliary matrix shape of Add1 node
      vector<int64_t> newnodeShapeVec;
      if (oriNmsScorceShape.size() < INT_NUM_THREE) {
          return PARAM_INVALID;
      }
      newnodeShapeVec.push_back(oriNmsScorceShape[0]);
      newnodeShapeVec.push_back(oriNmsScorceShape[1]);
      newnodeShapeVec.push_back(oriNmsScorceShape[2]);
      ge::GeShape add1newShape(newnodeShapeVec);

      // Set node properties
      ge::GeTensorDesc tensorDescadd1(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
      tensorDescadd1.SetShape(add1newShape);
      tensorDescadd1.SetOriginShape(add1newShape);
      int32_t realDimCnt0 = add1newShape.GetDimNum();
      ge::TensorUtils::SetRealDimCnt(tensorDescadd1, realDimCnt0);
      FUSION_PASS_CHECK(Adds->AddInputDesc("x", tensorDescadd1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x for Adds after valid num is null, fusion failed."),
                      return FAILED);
      Adds->AddOutputDesc("y", tensorDescadd1);
      ge::AttrUtils::SetFloat(Adds, "value", 1);
      // add node add1 to graph
      ge::NodePtr add1Node = graph.AddNode(Adds);

      FUSION_PASS_CHECK(
          add1Node == nullptr,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "AddsNode fusionNode is null, fusion failed."),
          return PARAM_INVALID);

      newNodes.push_back(add1Node);

      if (isfindSoftmax) {
        auto inAnchorSizes = fusedNode->GetAllInDataAnchorsSize();
        if (inAnchorSizes < INT_NUM_TWO) {
          return FAILED;
        }
        auto ExpandDims_5_node = fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
        auto strided_slice_10_node = ExpandDims_5_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        auto Reshape_4_node = strided_slice_10_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        auto Softmax_node = Reshape_4_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
        ge::GeTensorDesc softmaxDesc = Softmax_node->GetOpDesc()->GetInputDesc(0);
        vector<int64_t> softmax_shape = softmaxDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(softmax_shape.size() < 2,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "can not get input nms scorce shape. shape is less then 2!"),
                          return PARAM_INVALID);
        // Add the description (input, output, name, attribute) of the cast1 node
        ge::OpDescPtr Cast;
        std::string cast1DescName = fusedNode->GetOpDesc()->GetName();

        // Define node name----->BatchMultiClassNonMaxSuppression_Cast1
        FUSION_PASS_MAKE_SHARED((Cast = std::make_shared<ge::OpDesc>(cast1DescName + "_Cast1", "Cast")),
                                Cast = nullptr;
                                return PARAM_INVALID);

        // Define out auxiliary matrix shape of Cast1 node
        vector<int64_t> castnodeShapeVec;
        castnodeShapeVec.push_back(softmax_shape[0]);
        castnodeShapeVec.push_back(softmax_shape[1]);
        ge::GeShape cast1newShape(castnodeShapeVec);

        // Set node properties
        ge::GeTensorDesc tensorDesccast1(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
        tensorDesccast1.SetShape(cast1newShape);
        tensorDesccast1.SetOriginFormat(ge::FORMAT_ND);
        tensorDesccast1.SetOriginShape(cast1newShape);
        tensorDesccast1.SetOriginDataType(ge::DT_FLOAT);

        int32_t realDimCnt0 = cast1newShape.GetDimNum();
        ge::TensorUtils::SetRealDimCnt(tensorDesccast1, realDimCnt0);
        FUSION_PASS_CHECK(Cast->AddInputDesc("x", tensorDesccast1) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x for Cast after valid num is null, fusion failed."),
                        return FAILED);
        Cast->AddOutputDesc("y", tensorDesccast1);
        ge::AttrUtils::SetFloat(Cast, "dst_type", 0);
        // add node cast1 to graph
        ge::NodePtr cast1Node = graph.AddNode(Cast);

        FUSION_PASS_CHECK(cast1Node == nullptr,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "CastNode fusionNode is null, fusion failed."),
                          return PARAM_INVALID);

        newNodes.push_back(cast1Node);

        // Add the original input side 0 to cast1Node
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(Softmax_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                    cast1Node->GetInDataAnchor(0)),
                  OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                          Softmax_node->GetName().c_str(), 0,
                          cast1Node->GetName().c_str(), 0),
                          return FAILED);
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(Softmax_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                    Softmax_node->GetInDataAnchor(0)) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return FAILED);
        // Add the output edge of cast1Node to Softmax_node
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(cast1Node->GetOutDataAnchor(0),
                                                    Softmax_node->GetInDataAnchor(0)),
                OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                        cast1Node->GetName().c_str(), 0,
                        Softmax_node->GetName().c_str(), 0),
                        return FAILED);
      }

      // Add the description (input, output, name, attribute) of the mul node
      ge::OpDescPtr Muls;
      std::string mulDescName = fusedNode->GetOpDesc()->GetName();

      // Define node name----->BatchMultiClassNonMaxSuppression_Muls
      FUSION_PASS_MAKE_SHARED((Muls = std::make_shared<ge::OpDesc>(mulDescName + "_Muls", "Muls")),
                              return FAILED);

      // Define out auxiliary matrix shape of Muls node
      ge::GeShape mulnewShape(newnodeShapeVec);

      // Set node properties
      ge::GeTensorDesc tensorDescmul(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
      tensorDescmul.SetShape(mulnewShape);
      tensorDescmul.SetOriginShape(mulnewShape);
      int32_t realDimCnt1 = mulnewShape.GetDimNum();
      ge::TensorUtils::SetRealDimCnt(tensorDescmul, realDimCnt1);
      FUSION_PASS_CHECK(Muls->AddInputDesc("x", tensorDescmul) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x for Muls after valid num is null, fusion failed."),
                      return FAILED);
      Muls->AddOutputDesc("y", tensorDescmul);
      ge::AttrUtils::SetFloat(Muls, "value", -1);

      // add node mul to graph
      ge::NodePtr mulNode = graph.AddNode(Muls);
      FUSION_PASS_CHECK(mulNode == nullptr,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "MulsNode fusionNode is null, fusion failed."),
                        return PARAM_INVALID);
      newNodes.push_back(mulNode);

      // Add the description (input, output, name, attribute) of the add_mul node
      ge::OpDescPtr Adds_mul;
      std::string add_mulDescName = fusedNode->GetOpDesc()->GetName();

      // Define node name----->BatchMultiClassNonMaxSuppression_Add_mul
      FUSION_PASS_MAKE_SHARED((Adds_mul = std::make_shared<ge::OpDesc>(add_mulDescName + "_Add_mul", "Adds")),
                              return PARAM_INVALID);

      // Define out auxiliary matrix shape of Add1 node
      ge::GeShape add_mulnewShape(newnodeShapeVec);

      // Set node properties
      ge::GeTensorDesc tensorDescadd_mul(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
      tensorDescadd_mul.SetShape(add_mulnewShape);
      tensorDescadd_mul.SetOriginShape(add_mulnewShape);
      int32_t realDimCnt2 = add_mulnewShape.GetDimNum();
      ge::TensorUtils::SetRealDimCnt(tensorDescadd_mul, realDimCnt2);
      FUSION_PASS_CHECK(Adds_mul->AddInputDesc("x", tensorDescadd_mul) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x for Adds_mul after valid num is null, fusion failed."),
                      return FAILED);
      Adds_mul->AddOutputDesc("y", tensorDescadd_mul);
      ge::AttrUtils::SetFloat(Adds_mul, "value", 65535.0/65534.0);

      // add node add_mul to graph
      ge::NodePtr add_mulNode = graph.AddNode(Adds_mul);
      FUSION_PASS_CHECK(add_mulNode == nullptr,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Adds_mul Node fusionNode is null, fusion failed."),
                        return PARAM_INVALID);
      newNodes.push_back(add_mulNode);


      // Add the description (input, output, name, attribute) of the div node
      ge::OpDescPtr Div;
      std::string divDescName = fusedNode->GetOpDesc()->GetName();

      // Define node name----->BatchMultiClassNonMaxSuppression_Div
      FUSION_PASS_MAKE_SHARED((Div = std::make_shared<ge::OpDesc>(divDescName + "_Div", "Div")),
                              return PARAM_INVALID);

      // Define out auxiliary matrix shape of Div node
      ge::GeShape divnewShape(newnodeShapeVec);

      // Set node properties
      ge::GeTensorDesc tensorDescdiv(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
      tensorDescdiv.SetShape(divnewShape);
      tensorDescdiv.SetOriginShape(divnewShape);
      int32_t realDimCnt3 = divnewShape.GetDimNum();
      ge::TensorUtils::SetRealDimCnt(tensorDescdiv, realDimCnt3);
      FUSION_PASS_CHECK(Div->AddInputDesc("x1", tensorDescdiv) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x1 for Div after valid num is null, fusion failed."),
                      return FAILED);
      FUSION_PASS_CHECK(Div->AddInputDesc("x2", tensorDescdiv) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x2 for Div after valid num is null, fusion failed."),
                      return FAILED);
      Div->AddOutputDesc("y", tensorDescdiv);

      // add node div to graph
      ge::NodePtr divNode = graph.AddNode(Div);
      FUSION_PASS_CHECK(divNode == nullptr,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "DivNode fusionNode is null, fusion failed."),
                        return PARAM_INVALID);
      newNodes.push_back(divNode);

      // Add the original input side 0 to add1node
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                  add1Node->GetInDataAnchor(0)),
                OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                        fusedNode->GetName().c_str(), 0,
                        add1Node->GetName().c_str(), 0),
                        return FAILED);

      // Add the original input edge 0 to mulnode
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                  mulNode->GetInDataAnchor(0)),
                OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                        fusedNode->GetName().c_str(), 0,
                        mulNode->GetName().c_str(), 0),
                        return FAILED);

      // Add the output edge of add1node to divnode
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(add1Node->GetOutDataAnchor(0),
                                                  divNode->GetInDataAnchor(0)),
              OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      add1Node->GetName().c_str(), 0,
                      divNode->GetName().c_str(), 0),
                      return FAILED);

      // Add the output edge of mulnode to add_ mulNode
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0),
                                                  add_mulNode->GetInDataAnchor(0)),
              OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      mulNode->GetName().c_str(), 0,
                      add_mulNode->GetName().c_str(), 0),
                      return FAILED);

      // Add The output edge of mulnode is added to divnode
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(add_mulNode->GetOutDataAnchor(0),
                                                  divNode->GetInDataAnchor(1)),
              OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      add_mulNode->GetName().c_str(), 0,
                      divNode->GetName().c_str(), 0),
                      return FAILED);

      //Delete the original edge
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                  fusedNode->GetInDataAnchor(1)) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return FAILED);

      // Add the output edge of divnode to fusednode
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(divNode->GetOutDataAnchor(0),
                                                  fusedNode->GetInDataAnchor(1)),
              OP_LOGE("Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      divNode->GetName().c_str(), 0,
                      fusedNode->GetName().c_str(), 0),
                      return FAILED);
    }
    AddTransposeBeforeNode(fusedNode, 1, permScoreList, graph);

  }

  // do infer for fused node again, and update fused node output shape
  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(0);
  vector<int64_t> oriOutputShape = outputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(oriOutputShape.size() < 3,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "can not get output shape. shape size less then 3!"),
                    return PARAM_INVALID);
  vector<int64_t> outputShapeVec;
  outputShapeVec.push_back(oriOutputShape[0]);
  outputShapeVec.push_back(oriOutputShape[2]);
  outputShapeVec.push_back(oriOutputShape[1]);
  ge::GeShape outputShape(outputShapeVec);
  outputDesc.SetShape(outputShape);
  outputDesc.SetOriginShape(outputShape);
  // update fused node output info
  auto opOutputDesc = fusedNode->GetOpDesc();
  opOutputDesc->UpdateOutputDesc(0, outputDesc);

  // insert transpose at output 0
  AddTransposeAfterNode(fusedNode, 0, permScoreList, graph);

  ge::AttrUtils::SetBool(fusedNode->GetOpDesc(), "transpose_box", true);

  // for performance change nms_valid_num shape from [batch] to [batch, 8] and insert a SliceD
  ge::GeTensorDesc nmsNumDesc = fusedNode->GetOpDesc()->GetOutputDesc(3);
  vector<int64_t> oriNmsNumShape = nmsNumDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(oriNmsNumShape.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "can not get output nms valid num shape. shape is empty!"),
                    return PARAM_INVALID);
  vector<int64_t> newShapeVec;
  newShapeVec.push_back(oriNmsNumShape[0]);
  newShapeVec.push_back(8);

  // new a slice node
  std::shared_ptr<ge::OpDesc> reduceDesc = nullptr;
  std::string reduceDescName = fusedNode->GetName() + "_Output_3_reduce";
  reduceDesc = std::make_shared<ge::OpDesc>(reduceDescName, "StridedSliceD");
  FUSION_PASS_CHECK(reduceDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add reduce after valid num is null, fusion failed."),
                    return FAILED);
  FUSION_PASS_CHECK(reduceDesc->AddOutputDesc("y", nmsNumDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output y for reduce after valid num is null, fusion failed."),
                    return FAILED);

  ge::GeShape newShape(newShapeVec);
  nmsNumDesc.SetShape(newShape);
  nmsNumDesc.SetOriginShape(newShape);
  opOutputDesc->UpdateOutputDesc(3, nmsNumDesc);
  FUSION_PASS_CHECK(reduceDesc->AddInputDesc("x", nmsNumDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input x for reduce after valid num is null, fusion failed."),
                    return FAILED);
  ge::AttrUtils::SetListInt(reduceDesc, "begin", {0, 0});
  ge::AttrUtils::SetListInt(reduceDesc, "end", {oriNmsNumShape[0], 1});
  ge::AttrUtils::SetListInt(reduceDesc, "strides", {1, 1});
  ge::AttrUtils::SetInt(reduceDesc, "begin_mask", 0);
  ge::AttrUtils::SetInt(reduceDesc, "end_mask", 0);
  ge::AttrUtils::SetInt(reduceDesc, "ellipsis_mask", 0);
  ge::AttrUtils::SetInt(reduceDesc, "new_axis_mask", 0);
  ge::AttrUtils::SetInt(reduceDesc, "shrink_axis_mask", 0);

  // add node to graph
  ge::NodePtr reduceNode = graph.AddNode(reduceDesc);
  FUSION_PASS_CHECK(reduceNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "reduceNode is null, fusion failed."),
                    return FAILED);
  auto out_data_anchor = fusedNode->GetOutDataAnchor(INT_NUM_THREE);
  FUSION_PASS_CHECK(out_data_anchor == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "out_data_anchor is null, fusion failed."),
                    return FAILED);
  // add edge GraphUtils node output with other node input
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(INT_NUM_THREE)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(INT_NUM_THREE), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge failed."), return FAILED);
  }

  // add input for reduce node
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetOutDataAnchor(3), reduceNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge edge failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("BatchMultiClassNonMaxSuppressionFusionPass", BUILT_IN_GRAPH_PASS,
              BatchMultiClassNonMaxSuppressionFusionPass);
}  // namespace fe
