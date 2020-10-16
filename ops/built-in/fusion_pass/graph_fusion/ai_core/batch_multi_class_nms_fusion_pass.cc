/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief batch_multi_class_nms fusion pass
 *
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
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeBatchMultiClassNonMaxSuppression";
static const string FUSED_NODE = "BatchMultiClassNonMaxSuppression";

vector<FusionPattern *> BatchMultiClassNonMaxSuppressionFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = \
  new (std::nothrow) FusionPattern("BatchMultiClassNonMaxSuppressionFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),  return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
          .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

bool BatchMultiClassNonMaxSuppressionFusionPass::CheckTransposeBeforeSlice(ge::NodePtr checkNode) {
  auto checkOpType = checkNode->GetType();
  if ((checkOpType != "Slice") && (checkOpType != "SliceD")) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op name is not Slice or SliceD, is %s", checkOpType.c_str());
    return false;
  }
  if (checkNode->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the Slice op have more than one output");
    return false;
  }
  return true;
}

Status BatchMultiClassNonMaxSuppressionFusionPass::Fusion(ge::ComputeGraph &graph,
                                                          Mapping &mapping,
                                                          vector<ge::NodePtr> &newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionFusionPass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
           return PARAM_INVALID);
  ge::GeTensorDesc boxesInputDesc = fusedNode->GetOpDesc()->GetInputDesc(0);
  vector<int64_t> boxesInputShape = boxesInputDesc.GetShape().GetDims();
  auto boxesSize = boxesInputShape.size();
  bool isNeedTranposeBeforeScore = true;
  vector<int64_t> permBoxesList;

  // insert transpose at input 0
  if (boxesSize != 4) {
    isNeedTranposeBeforeScore = false;
    permBoxesList = {0,2,1};
  } else {
    permBoxesList = {0,2,3,1};
  }
  AddTransposeBeforeNode(fusedNode, 0, permBoxesList, graph);
  // insert transpose at input 1

  // get the input 1 peer node op type
  vector<int64_t> permScoreList = {0,2,1};
  auto peerNode = fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  auto peerOpType = peerNode->GetType();
  bool isNeedTransposeBeforeSlice = false;
  isNeedTransposeBeforeSlice = CheckTransposeBeforeSlice(peerNode);

  if (isNeedTranposeBeforeScore && isNeedTransposeBeforeSlice) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "will insert transpose before Slice + BatchMultiClassNonMaxSuppression");
    // insert transpose before Slice
    AddTransposeBeforeNode(peerNode, 0, permScoreList, graph);
    if (peerOpType == "Slice") {
      Operator op = ge::OpDescUtils::CreateOperatorFromNode(peerNode);
      DataType dtype = op.GetInputDesc("offsets").GetDataType();

      vector<ge::GeTensorPtr> sliceTensorPtr = ge::OpDescUtils::MutableWeights(peerNode);
      FUSION_PASS_CHECK(sliceTensorPtr.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "slice const is nullptr!"), return PARAM_INVALID);

      if (dtype == ge::DT_INT32) {
        // modify Slice offset const node value
        ge::GeTensorPtr offsetsTensorPtr = sliceTensorPtr[0];
        int32_t *constOffsetData = (int32_t *) (offsetsTensorPtr->GetData().GetData());
        if (constOffsetData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get Offset Data from const node is NULL.");
          return PARAM_INVALID;
        }

        vector<int32_t> offsetsNew = {constOffsetData[0], constOffsetData[2], constOffsetData[1]};
        offsetsTensorPtr->SetData(reinterpret_cast<uint8_t *>(offsetsNew.data()), offsetsNew.size() * sizeof(int32_t));

        // modify Slice size const node value
        ge::GeTensorPtr sizeTensorPtr = sliceTensorPtr[1];
        int32_t *constSizeData = (int32_t *) (sizeTensorPtr->GetData().GetData());
        if (constSizeData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get size Data from const node is NULL.");
          return PARAM_INVALID;
        }
        vector<int32_t> sizeNew = {constSizeData[0], constSizeData[2], constSizeData[1]};
        sizeTensorPtr->SetData(reinterpret_cast<uint8_t *>(sizeNew.data()), sizeNew.size() * sizeof(int32_t));
      } else {
        // modify Slice offset const node value
        ge::GeTensorPtr offsetsTensorPtr = sliceTensorPtr[0];
        int64_t *constOffsetData = (int64_t *) (offsetsTensorPtr->GetData().GetData());
        if (constOffsetData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get Offset Data from const node is NULL.");
          return PARAM_INVALID;
        }
        vector<int64_t> offsetsNew = {constOffsetData[0], constOffsetData[2], constOffsetData[1]};
        offsetsTensorPtr->SetData(reinterpret_cast<uint8_t *>(offsetsNew.data()), offsetsNew.size() * sizeof(int64_t));

        // modify Slice size const node value
        ge::GeTensorPtr sizeTensorPtr = sliceTensorPtr[1];
        int64_t *constSizeData = (int64_t *) (sizeTensorPtr->GetData().GetData());
        if (constSizeData == nullptr) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Get size Data from const node is NULL.");
          return PARAM_INVALID;
        }
        vector<int64_t> sizeNew = {constSizeData[0], constSizeData[2], constSizeData[1]};
        sizeTensorPtr->SetData(reinterpret_cast<uint8_t *>(sizeNew.data()), sizeNew.size() * sizeof(int64_t));
      }
    } else {
      vector<int64_t> offsets;
      ge::AttrUtils::GetListInt(peerNode->GetOpDesc(), "offsets", offsets);
      FUSION_PASS_CHECK(offsets.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "sliceD attr offsets is nullptr!"), return PARAM_INVALID);
      vector<int64_t> size;
      ge::AttrUtils::GetListInt(peerNode->GetOpDesc(), "size", size);
      FUSION_PASS_CHECK(size.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "sliceD attr size is nullptr!"), return PARAM_INVALID);
      vector<int64_t> offsetsNew = {offsets[0], offsets[2], offsets[1]};
      vector<int64_t> sizeNew = {size[0], size[2], size[1]};
      ge::AttrUtils::SetListInt( peerNode->GetOpDesc(), "offsets", offsetsNew);
      ge::AttrUtils::SetListInt( peerNode->GetOpDesc(), "sizeNew", sizeNew);
    }
    // update Slice out desc
    vector<int64_t> oriSliceOutputShape = peerNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
    FUSION_PASS_CHECK(oriSliceOutputShape.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "Slice output shape is nullptr!"), return PARAM_INVALID);
    vector<int64_t> outputSliceShapeVec;
    outputSliceShapeVec.push_back(oriSliceOutputShape[0]);
    outputSliceShapeVec.push_back(oriSliceOutputShape[2]);
    outputSliceShapeVec.push_back(oriSliceOutputShape[1]);
    ge::GeShape outputSliceShape(outputSliceShapeVec);
    auto opSliceOutputDesc = peerNode->GetOpDesc()->GetOutputDesc(0);
    opSliceOutputDesc.SetShape(outputSliceShape);
    opSliceOutputDesc.SetOriginShape(outputSliceShape);
    peerNode->GetOpDesc()->UpdateOutputDesc(0, opSliceOutputDesc);
    // update BatchMultiClassNonMaxSuppression input 1 desc
    auto inputDesc = fusedNode->GetOpDesc()->GetInputDesc(1);
    inputDesc.SetShape(outputSliceShape);
    inputDesc.SetOriginShape(outputSliceShape);
    fusedNode->GetOpDesc()->UpdateInputDesc(1, inputDesc);
  } else if (isNeedTranposeBeforeScore) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "will insert transpose before BatchMultiClassNonMaxSuppression");
    // insert directly before the input 1
    AddTransposeBeforeNode(fusedNode, 1, permScoreList, graph);
  }
  // do infer for fused node again, and update fused node output shape
  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(0);
  vector<int64_t> oriOutputShape = outputDesc.GetShape().GetDims();
  if (oriOutputShape.empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "can not get output shape. shape is empty!");
    return PARAM_INVALID;
  }
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
  if (oriNmsNumShape.empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "can not get output nms valid num shape. shape is empty!");
    return PARAM_INVALID;
  }
  vector<int64_t> newShapeVec;
  newShapeVec.push_back(oriNmsNumShape[0]);
  newShapeVec.push_back(8);

  // new a slice node
  std::shared_ptr<ge::OpDesc> reduceDesc = nullptr;
  std::string reduceDescName = fusedNode->GetName() + "_Output_3_reduce";
  reduceDesc = std::make_shared<ge::OpDesc>(reduceDescName, "StridedSliceD");
  FUSION_PASS_CHECK(reduceDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add reduce after valid num is null, fusion failed."),
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
  // add edge GraphUtils node output with other node input
  for (auto inDataAnchor :
       fusedNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(3),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(0),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge failed."), return FAILED);
  }

  // add input for reduce node
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetOutDataAnchor(3),
                                   reduceNode->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge edge failed."), return FAILED);


  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("BatchMultiClassNonMaxSuppressionFusionPass", BUILT_IN_GRAPH_PASS,
              BatchMultiClassNonMaxSuppressionFusionPass);
}

