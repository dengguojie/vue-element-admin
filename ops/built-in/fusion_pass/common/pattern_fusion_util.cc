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
 * \file pattern_fusion_util.cpp
 * \brief add a control edge from source node to dest node Provide some
 *   basic methods for fusion pass (include change a const anchor to attr)
 */
#include "pattern_fusion_util.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "fp16_t.hpp"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

using namespace ge;
using namespace std;

namespace fe {

static const char* STREAMSWITCH = "StreamSwitch";
static const string ATTR_TYPE_SETFLOAT = "SetFloat";
static const string ATTR_TYPE_SETLISTFLOAT = "SetListFloat";
static const string ATTR_TYPE_SETINT = "SetInt";
static const string ATTR_TYPE_SETLISTINT = "SetListInt";


Status PatternFusionUtil::AddInputToOutput(ge::NodePtr node, std::vector<PassInputInfo>& inputInfoVec) {
  string curOpType = node->GetType();
  FUSION_PASS_CHECK(node == nullptr, OP_LOGE(curOpType.c_str(), "node is null, add input to output failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(inputInfoVec.empty(),
                    OP_LOGE(curOpType.c_str(), "inputInfoVec is empty, add input to output failed."),
                    return PARAM_INVALID);

  for (auto& inputInfo : inputInfoVec) {
    if (node->GetOpDesc()->GetInputsSize() <= inputInfo.inputOpDescIndex) {
      OP_LOGE(curOpType.c_str(), "node name:%s, input index:%u is larger than inputs size:%zu.",
              node->GetName().c_str(), inputInfo.inputOpDescIndex, node->GetOpDesc()->GetInputsSize());
      return FAILED;
    }

    ge::GeTensorDesc inputTensorDesc = node->GetOpDesc()->GetInputDesc(inputInfo.inputOpDescIndex);
    if (inputInfo.inputOpDescName != node->GetOpDesc()->GetInputNameByIndex(inputInfo.inputOpDescIndex)) {
      OP_LOGE(curOpType.c_str(), "node name:%s, input name:%s is not same to opdesc input name:%s.",
              node->GetName().c_str(), inputInfo.inputOpDescName.c_str(),
              node->GetOpDesc()->GetInputNameByIndex(inputInfo.inputOpDescIndex).c_str());
      return FAILED;
    }

    if (node->GetOpDesc()->AddOutputDesc(inputInfo.inputOpDescName, inputTensorDesc) != ge::GRAPH_SUCCESS) {
      OP_LOGE(curOpType.c_str(), "node name:%s, input name:%s, inputIdx:%u, add output failed.",
              node->GetName().c_str(), inputInfo.inputOpDescName.c_str(), inputInfo.inputOpDescIndex);
      return FAILED;
    }

    OP_LOGD(curOpType.c_str(), "node name:%s, input name:%s, inputIdx:%u, add output success.", node->GetName().c_str(),
            inputInfo.inputOpDescName.c_str(), inputInfo.inputOpDescIndex);
  }

  return SUCCESS;
}

void PatternFusionUtil::SetConstValueToAttrWithType(ge::OpDescPtr op_desc, const ge::Tensor& const_tensor,
                                                    const DataType& dtype, PassAttrInfo& attrInfo) {
  string curOpType = op_desc->GetType();
  OP_LOGD(curOpType.c_str(), "Begin to convert const data of node[%s] to attribute value.", op_desc->GetName().c_str());
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGE(curOpType.c_str(), "const_data_ptr is null, fusion failed."),
                      return );
    size = const_tensor.GetSize() / sizeof(int32_t);
    if (attrInfo.attrType == ATTR_TYPE_SETINT) {
      int32_t const_data = (int32_t)(*const_data_ptr);
      OP_LOGD(curOpType.c_str(), "Int32 value of const node is %d.", const_data);
      ge::AttrUtils::SetInt(op_desc, attrInfo.attrName, const_data);
    }
    if (attrInfo.attrType == ATTR_TYPE_SETLISTINT) {
      std::vector<int32_t> const_data_vec;
      for (size_t i = 0; i < size; ++i) {
        int32_t const_data = (int32_t)(*(const_data_ptr + i));
        const_data_vec.push_back(const_data);
        OP_LOGD(curOpType.c_str(), "Int32 list value[%u] of const node is %d.", i, const_data);
      }
      ge::AttrUtils::SetListInt(op_desc, attrInfo.attrName, const_data_vec);
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGE(curOpType.c_str(), "const_data_ptr is null, fusion failed."),
                      return );
    size = const_tensor.GetSize() / sizeof(int64_t);
    if (attrInfo.attrType == ATTR_TYPE_SETINT) {
      int64_t const_data = (int64_t)(*const_data_ptr);
      OP_LOGD(curOpType.c_str(), "Int64 value of const node is %ld.", const_data);
      ge::AttrUtils::SetInt(op_desc, attrInfo.attrName, const_data);
    }
    if (attrInfo.attrType == ATTR_TYPE_SETLISTINT) {
      std::vector<int64_t> const_data_vec;
      for (size_t i = 0; i < size; ++i) {
        int64_t const_data = (int64_t)(*(const_data_ptr + i));
        const_data_vec.push_back(const_data);
        OP_LOGD(curOpType.c_str(), "Int64 list value[%u] of const node is %ld.", i, const_data);
      }
      ge::AttrUtils::SetListInt(op_desc, attrInfo.attrName, const_data_vec);
    }
  } else if (dtype == ge::DT_FLOAT) {
    float* const_data_ptr = (float*)const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGE(curOpType.c_str(), "const_data_ptr is null, fusion failed."),
                      return );
    size = const_tensor.GetSize() / sizeof(float);
    if (attrInfo.attrType == ATTR_TYPE_SETFLOAT) {
      float const_data = (float)(*const_data_ptr);
      OP_LOGD(curOpType.c_str(), "Float value of const node is %f.", const_data);
      ge::AttrUtils::SetFloat(op_desc, attrInfo.attrName, const_data);
    }
    if (attrInfo.attrType == ATTR_TYPE_SETLISTFLOAT) {
      std::vector<float> const_data_vec;
      for (size_t i = 0; i < size; ++i) {
        float const_data = (float)(*(const_data_ptr + i));
        const_data_vec.push_back(const_data);
        OP_LOGD(curOpType.c_str(), "Float list value[%u] of const node is %f.", i, const_data);
      }
      ge::AttrUtils::SetListFloat(op_desc, attrInfo.attrName, const_data_vec);
    }
  } else if (dtype == ge::DT_FLOAT16) {
    uint16_t* const_data_ptr = (uint16_t*)const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGE(curOpType.c_str(), "const_data_ptr is null, fusion failed."),
                      return );
    size = const_tensor.GetSize() / sizeof(uint16_t);
    if (attrInfo.attrType == ATTR_TYPE_SETFLOAT) {
      uint16_t const_data = (uint16_t)(*const_data_ptr);
      fp16_t const_data_fp16(const_data);
      float const_data_fp32 = const_data_fp16.toFloat();
      OP_LOGD(curOpType.c_str(), "Fp16 value of const node is %f.", const_data_fp32);
      ge::AttrUtils::SetFloat(op_desc, attrInfo.attrName, const_data_fp32);
    }
    if (attrInfo.attrType == ATTR_TYPE_SETLISTFLOAT) {
      std::vector<float> const_data_vec;
      for (size_t i = 0; i < size; ++i) {
        uint16_t const_data = (uint16_t)(*(const_data_ptr + i));
        fp16_t const_data_fp16(const_data);
        float const_data_fp32 = const_data_fp16.toFloat();
        const_data_vec.push_back(const_data_fp32);
        OP_LOGD(curOpType.c_str(), "Fp16 list value[%u] of const node is %f.", i, const_data_fp32);
      }
      ge::AttrUtils::SetListFloat(op_desc, attrInfo.attrName, const_data_vec);
    }
  } else {
    OP_LOGW(curOpType.c_str(), "Data type of const node of [%s] is not supported.", op_desc->GetName().c_str());
  }
}

Status PatternFusionUtil::ConstToAttrWithNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, std::string fusionOpType,
                                              std::vector<PassAttrInfo>& attrInfos, ge::NodePtr& fusionNode) {
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(fusionOpType.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(fusionOpType.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                  fusedNode->GetName().c_str()),
                    return PARAM_INVALID);

  std::vector<int> attrIndexVec;
  for (size_t i = 0; i < attrInfos.size(); i++) {
    attrIndexVec.push_back(attrInfos[i].attrIndex);
  }
  std::sort(attrIndexVec.begin(), attrIndexVec.end());
  // op desc of fusion node
  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(fusedDesc);

  // update the infer depends
  std::vector<string> depends_vec;
  std::vector<string>::iterator it_depends;
  depends_vec = fusionDesc->GetOpInferDepends();
  for (size_t i = 0; i < attrInfos.size(); i++) {
    it_depends = find(depends_vec.begin(), depends_vec.end(), attrInfos[i].attrName);
    if (it_depends != depends_vec.end()) {
      depends_vec.erase(it_depends);
    }
  }
  fusionDesc->SetOpInferDepends(depends_vec);

  FUSION_PASS_CHECK(fusionDesc == nullptr, OP_LOGE(fusionOpType.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                   fusedNode->GetName().c_str()),
                    return PARAM_INVALID);
  // convert const input node to attribute of fusion node
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  Tensor constTensor;
  PassAttrInfo attr;
  vector<ge::NodePtr> constNodes;
  std::map<string, uint32_t> fusionDescInputName;
  for (size_t i = 0, j = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    if (!FindAttrInfoByIndex(attrInfos, i, attr)) {
      if (fusedDesc->GetInputNameByIndex(i).empty()) {
        OP_LOGE(fusionOpType.c_str(), "Get fuseNode:[%s]'s inputName by index [%d] failed.",
                fusedNode->GetName().c_str(), i);
        return FAILED;
      }
      fusionDescInputName.emplace(fusedDesc->GetInputNameByIndex(i), j);
      j++;
      continue;
    }
    ge::InDataAnchorPtr fused_anchor_ptr = fusedNode->GetInDataAnchor(attr.attrIndex);
    if (fused_anchor_ptr == nullptr) {
      continue;
    }
    ge::OutDataAnchorPtr const_anchor_ptr = fused_anchor_ptr->GetPeerOutAnchor();
    if (const_anchor_ptr == nullptr) {
      continue;
    }
    ge::NodePtr constNode = const_anchor_ptr->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(constNode);
    if (type != "Constant" && type != "Const") {
      OP_LOGW(fusionOpType.c_str(), "Node %s's %dth input is not a constant.", fusedNode->GetName().c_str(),
              attr.attrIndex);
      return NOT_CHANGED;
    } else {
      if (std::find(constNodes.begin(), constNodes.end(), constNode) == constNodes.end()) {
        constNodes.push_back(constNode);
      }
    }
    if (fusedDesc->GetInputNameByIndex(attr.attrIndex).empty()) {
      OP_LOGE(fusionOpType.c_str(), "Get fuseNode:[%s]'s inputName by index [%d] failed.", fusedNode->GetName().c_str(),
              attr.attrIndex);
      return FAILED;
    }
    op.GetInputConstData(fusedDesc->GetInputNameByIndex(attr.attrIndex), constTensor);
    SetConstValueToAttrWithType(fusionDesc, constTensor, fusedDesc->GetInputDesc(attr.attrIndex).GetDataType(), attr);
  }
  // remove the inputdesc which need to be removed
  for (int i = attrIndexVec.size() - 1; i >= 0; i--) {
    unsigned int index = attrIndexVec[i];
    if (index >= fusionDesc->GetInputsSize()) {
      OP_LOGI(fusionOpType.c_str(), "Index[%d] is beyond the size[%d] of input desc", index,
              fusionDesc->GetInputsSize());
      continue;
    }
    if (!OpDescUtils::ClearInputDesc(fusionDesc, index)) {
      OP_LOGI(fusionOpType.c_str(), "Fail to clear input desc[%d]", index);
    }
  }
  fusionDesc->UpdateInputName(fusionDescInputName);

  auto realFusedOp = ge::OperatorFactory::CreateOperator("realFusedOp", fusionOpType);
  if (realFusedOp.IsEmpty()) {
    OP_LOGE(fusionOpType.c_str(), "create fusion node %s failed", fusionOpType.c_str());
    return FAILED;
  }
  auto realFusedOpDescPtr = ge::OpDescUtils::GetOpDescFromOperator(realFusedOp);
  realFusedOp.BreakConnect();
  fusionDesc->AddInferFunc(realFusedOpDescPtr->GetInferFunc());

  // add fusion node to graph
  fusionDesc->SetType(fusionOpType);
  fusionNode = graph.AddNode(fusionDesc);
  // connect in data anchor
  FUSION_PASS_CHECK(fusionNode == nullptr, OP_LOGE(fusionOpType.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   fusionNode->GetName().c_str()),
                    return PARAM_INVALID);
  for (size_t i = 0, j = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    if (std::find(attrIndexVec.begin(), attrIndexVec.end(), i) != attrIndexVec.end()) {
      continue;
    }
    if (fusedNode->GetInDataAnchor(i) != nullptr && fusedNode->GetInDataAnchor(i)->GetPeerOutAnchor() != nullptr) {
      FUSION_PASS_CHECK(
        SUCCESS !=
        ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(i)->GetPeerOutAnchor(), fusionNode->GetInDataAnchor(j)),
        OP_LOGE(fusionOpType.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                fusedNode->GetName().c_str(), i, fusionNode->GetName().c_str(), j),
        return FAILED);
    }
    OP_LOGD(fusionOpType.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
            fusedNode->GetName().c_str(), i, fusionNode->GetName().c_str(), j);
    j++;
  }
  // connect out data anchor
  std::map<int, std::vector<ge::InDataAnchorPtr>> indexInDataAnchorMap;
  for (unsigned int i = 0; i < fusedNode->GetAllOutDataAnchors().size(); i++) {
    OutDataAnchorPtr outAnchor = fusedNode->GetOutDataAnchor(i);
    std::vector<ge::InDataAnchorPtr> inDataAnchorVec;
    if (outAnchor != nullptr) {
      for (auto inAnchor : outAnchor->GetPeerInDataAnchors()) {
        if (inAnchor != nullptr) {
          inAnchor->UnlinkAll();
        }
        inDataAnchorVec.push_back(inAnchor);
      }
    }
    indexInDataAnchorMap.emplace(i, inDataAnchorVec);
  }
  for (unsigned int i = 0; i < fusedNode->GetAllOutDataAnchors().size(); i++) {
    std::vector<ge::InDataAnchorPtr> inAnchorVec = indexInDataAnchorMap[i];
    for (unsigned int j = 0; j < inAnchorVec.size(); j++) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusionNode->GetOutDataAnchor(i), inAnchorVec[j]),
          OP_LOGE(fusionOpType.c_str(), "Add edge from fusion node:%s's index[%d] to fused node:%s's index[%d] failed.",
                  fusionNode->GetName().c_str(), i, fusedNode->GetName().c_str(), j),
          return FAILED);
      OP_LOGD(fusionOpType.c_str(), "Add edge from fusion node:%s's index[%d] to fused node:%s's index[%d].",
              fusionNode->GetName().c_str(), i, fusedNode->GetName().c_str(), j);
    }
  }

  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // connect in control anchor
  if (fusedNode->GetInControlAnchor() != nullptr) {
    if (!fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().empty() &&
        fusionNode->GetInControlAnchor() != nullptr) {
      OP_LOGI(fusionOpType.c_str(), "The PeerOutControlAnchors of fused node[%s] input control anchor is empty.",
              fusedNode->GetName().c_str());
      for (OutControlAnchorPtr outCtrlAnchorPtr : fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(outCtrlAnchorPtr, fusionNode->GetInControlAnchor()),
                          OP_LOGE(fusionOpType.c_str(), "Fail to add input control edge for fusion node:%s.",
                                  fusionNode->GetName().c_str()),
                          return FAILED);
      }
    }
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  // connect out control anchor
  if (fusedNode->GetOutControlAnchor() != nullptr) {
    if (!fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().empty() &&
        fusionNode->GetOutControlAnchor() != nullptr) {
      OP_LOGI(fusionOpType.c_str(), "The PeerInControlAnchors of fused node[%s] output control anchor is empty.",
              fusedNode->GetName().c_str());
      for (InControlAnchorPtr inCtrlAnchorPtr : fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusionNode->GetOutControlAnchor(), inCtrlAnchorPtr),
                          OP_LOGE(fusionOpType.c_str(), "Fail to add output control edge for fusion node:%s.",
                                  fusionNode->GetName().c_str()),
                          return FAILED);
      }
    }
    fusedNode->GetOutControlAnchor()->UnlinkAll();
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    OP_LOGE(fusionOpType.c_str(), "Remove Node:%s failed", fusedNode->GetName().c_str()),
                    return FAILED);

  for (auto oneConstNode : constNodes) {
    if (GetOutEdgeSize(oneConstNode) == 0) {
      FUSION_PASS_CHECK(SUCCESS != LinkControlAnchorForConst(oneConstNode, fusionNode),
              OP_LOGE(fusionOpType.c_str(), "Link control anchor Node[%s] failed", oneConstNode->GetName().c_str()),
              return FAILED);
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(oneConstNode),
                        OP_LOGE(fusionOpType.c_str(), "Remove Node[%s] failed", oneConstNode->GetName().c_str()),
                        return FAILED);
      OP_LOGI(fusionOpType.c_str(), "Remove const Node:[%s].", oneConstNode->GetName().c_str());
    } else {
      OP_LOGD(fusionOpType.c_str(), "Node:[%s] have output link to other node.", oneConstNode->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status PatternFusionUtil::LinkControlAnchorForConst(ge::NodePtr oneConstNode, ge::NodePtr fusionNode) {
  //link control anchor
  FUSION_PASS_CHECK(fusionNode == nullptr || fusionNode->GetOpDesc() == nullptr,
                    OP_LOGE("LinkControlAnchorForConst", "fusionNode or OpDesc is null, fusion failed."),
                    return FAILED);
  string fusionOpType = fusionNode->GetOpDesc()->GetType();
  auto constControlAnchors = oneConstNode->GetInControlAnchor()->GetPeerOutControlAnchors();
  for (const auto &outControlAnchor : constControlAnchors) {
    auto outNode = outControlAnchor->GetOwnerNode();
    OP_LOGD(fusionOpType.c_str(), "Get outNode node : %s outEdgeSize %d, inEdgeSize %d",
            outNode->GetOpDesc()->GetName().c_str(), GetOutEdgeSize(outNode), outNode->GetAllInAnchors().size());
    if (GetOutEdgeSize(outNode) == 1 && outNode->GetAllInAnchors().size() == 0) {
      OP_LOGD(fusionOpType.c_str(), "Link outNode node : %s to fusion node %s",
              outNode->GetOpDesc()->GetName().c_str(), fusionNode->GetOpDesc()->GetName().c_str());
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(outControlAnchor, fusionNode->GetInControlAnchor()),
                        OP_LOGE(fusionOpType.c_str(), "Add out control edge failed in const2attr with node."),
                        return FAILED);
    }
  }
  return SUCCESS;
}

Status PatternFusionUtil::RecordOriginalNamesForConstToAttr(ge::NodePtr& fusedNode,
                                                            std::vector<PassAttrInfo>& attrInfos,
                                                            std::vector<ge::NodePtr>& originalNodes) {
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE("RecordOriginalNamesForConstToAttr", "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    OP_LOGE("RecordOriginalNamesForConstToAttr", "Node:%s's OpDesc is null, fusion failed.",
                            fusedNode->GetName().c_str()),
                    return PARAM_INVALID);

  string curOpType = fusedDesc->GetType();
  vector<ge::NodePtr> constNodes;
  PassAttrInfo attr;
  for (size_t i = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    if (!FindAttrInfoByIndex(attrInfos, i, attr)) {
      continue;
    }
    ge::InDataAnchorPtr fused_anchor_ptr = fusedNode->GetInDataAnchor(attr.attrIndex);
    ge::OutDataAnchorPtr const_anchor_ptr = fused_anchor_ptr->GetPeerOutAnchor();
    if (const_anchor_ptr == nullptr) {
      continue;
    }
    ge::NodePtr constNode = const_anchor_ptr->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(constNode);
    if (type != "Constant" && type != "Const") {
      OP_LOGW(curOpType.c_str(), "Node %s's %dth input is not a constant.", fusedNode->GetName().c_str(),
              attr.attrIndex);
      return NOT_CHANGED;
    } else {
      if (std::find(constNodes.begin(), constNodes.end(), constNode) == constNodes.end()) {
        constNodes.push_back(constNode);
      }
    }
  }
  // for op data dump
  originalNodes = constNodes;
  originalNodes.push_back(fusedNode);
  return SUCCESS;
}

Status PatternFusionUtil::SetOutputDescAttrForDataDump(ge::NodePtr fusedNode, ge::NodePtr fusionNode) {
  FUSION_PASS_CHECK(fusionNode == nullptr || fusionNode->GetOpDesc() == nullptr,
                    OP_LOGE("SetOutputDescAttrForDataDump", "fusedNode or OpDesc is null, fusion failed."),
                    return FAILED);
  for (unsigned int i = 0; i < fusedNode->GetAllOutDataAnchors().size(); i++) {
    ge::AttrUtils::SetStr(fusionNode->GetOpDesc()->MutableOutputDesc(i), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME,
                          fusedNode->GetName());
    ge::AttrUtils::SetInt(fusionNode->GetOpDesc()->MutableOutputDesc(i), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,
                          i);
    GraphPassUtil::SetDataDumpOriginDataType(fusedNode->GetOpDesc()->GetOutputDesc(i).GetOriginDataType(),
                                             fusionNode->GetOpDesc()->MutableOutputDesc(i));
    GraphPassUtil::SetDataDumpOriginFormat(fusedNode->GetOpDesc()->GetOutputDesc(i).GetOriginFormat(),
                                           fusionNode->GetOpDesc()->MutableOutputDesc(i));
  }
  return SUCCESS;
}

bool PatternFusionUtil::FindAttrInfoByIndex(vector<PassAttrInfo>& attrInfos, int index, PassAttrInfo& retAttrinfo) {
  for (unsigned int i = 0; i < attrInfos.size(); i++) {
    if (attrInfos[i].attrIndex == index) {
      retAttrinfo = attrInfos[i];
      return true;
    }
  }
  return false;
}

size_t PatternFusionUtil::GetOutEdgeSize(NodePtr node) {
  size_t outEdgeSize = 0;
  if (node == nullptr) {
    return outEdgeSize;
  }
  for (OutDataAnchorPtr anchor : node->GetAllOutDataAnchors()) {
    if (anchor != nullptr) {
      outEdgeSize = outEdgeSize + anchor->GetPeerAnchors().size();
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    outEdgeSize = outEdgeSize + node->GetOutControlAnchor()->GetPeerAnchors().size();
  }
  return outEdgeSize;
}

ge::OpDescPtr PatternFusionUtil::GetFusionOpDesc(ge::NodePtr fusedNodePtr, std::string fusionOpType,
                                                 std::vector<PassAttrInfo>& attrInfos) {
  FUSION_PASS_CHECK(fusedNodePtr == nullptr, OP_LOGE(fusionOpType.c_str(), "fusedNode is null, fusion failed."),
                    return nullptr);
  ge::OpDescPtr opDescPtr = fusedNodePtr->GetOpDesc();
  FUSION_PASS_CHECK(opDescPtr == nullptr, OP_LOGE(fusionOpType.c_str(), "Op desc is null."), return nullptr);

  // clone opdesc from orginal opdesc
  ge::OpDescPtr fusionDescPtr = AttrUtils::CopyOpDesc(opDescPtr);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr,
                    OP_LOGE(fusionOpType.c_str(), "Fail to clone op desc from [%s].", opDescPtr->GetName().c_str()),
                    return nullptr);
  // set op type
  fusionDescPtr->SetType(fusionOpType);

  std::vector<int> attrIndexVec;
  for (size_t i = 0; i < attrInfos.size(); i++) {
    attrIndexVec.push_back(attrInfos[i].attrIndex);
  }
  std::sort(attrIndexVec.begin(), attrIndexVec.end());

  // remove the inputdesc which need to be removed
  for (int i = attrIndexVec.size() - 1; i >= 0; i--) {
    unsigned int index = attrIndexVec[i];
    if (index >= fusionDescPtr->GetInputsSize()) {
      OP_LOGI(fusionOpType.c_str(), "Index[%d] is beyond the size[%d] of input desc", index,
              fusionDescPtr->GetInputsSize());
      continue;
    }
    if (!OpDescUtils::ClearInputDesc(fusionDescPtr, index)) {
      OP_LOGI(fusionOpType.c_str(), "Fail to clear input desc[%d]", index);
    }
  }

  // convert const input node to attribute of fusion node
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNodePtr);
  std::map<string, uint32_t> fusionDescInputName;
  uint32_t newIndex = 0;

  string fusionNodeName = fusedNodePtr->GetName();
  for (size_t i = 0; i < fusedNodePtr->GetAllInDataAnchors().size(); i++) {
    PassAttrInfo attr;
    if (!FindAttrInfoByIndex(attrInfos, i, attr)) {
      if (fusionDescPtr->GetInputNameByIndex(i).empty()) {
        OP_LOGW(fusionOpType.c_str(), "FuseNode[%s]: get the inputName by index [%d] failed.",
                fusionNodeName.c_str(), i);
        return nullptr;
      }
      fusionDescInputName.emplace(fusionDescPtr->GetInputNameByIndex(i), newIndex);
      newIndex++;
      continue;
    }
    ge::InDataAnchorPtr fused_anchor_ptr = fusedNodePtr->GetInDataAnchor(attr.attrIndex);
    ge::OutDataAnchorPtr const_anchor_ptr = fused_anchor_ptr->GetPeerOutAnchor();
    if (const_anchor_ptr == nullptr) {
      continue;
    }
    ge::NodePtr constNode = const_anchor_ptr->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(constNode);
    if (type != "Constant" && type != "Const") {
      OP_LOGW(fusionOpType.c_str(), "FuseNode[%s]: the %dth input is not a constant.", fusedNodePtr->GetName().c_str(),
              attr.attrIndex);
      return nullptr;
    }
    OP_LOGI(fusionOpType.c_str(), "FuseNode[%s]: the name of the %dth input is %s.", fusionNodeName.c_str(),
            attr.attrIndex, fusionDescPtr->GetInputNameByIndex(attr.attrIndex).c_str());
    Tensor constTensor;
    op.GetInputConstData(fusionDescPtr->GetInputNameByIndex(attr.attrIndex), constTensor);
    SetConstValueToAttrWithType(fusionDescPtr, constTensor, opDescPtr->GetInputDesc(attr.attrIndex).GetDataType(),
                                attr);
  }

  fusionDescPtr->UpdateInputName(fusionDescInputName);

  return fusionDescPtr;
}

Status PatternFusionUtil::ParseChannelIdx(ge::GeTensorDesc& tensorDesc, size_t& channelIdx) {
  ge::Format tensorGeFormat = tensorDesc.GetOriginFormat();
  if (tensorGeFormat == FORMAT_NCHW) {
    channelIdx = 1;
    return SUCCESS;
  }
  if (tensorGeFormat == FORMAT_NHWC) {
    channelIdx = 3;
    return SUCCESS;
  }
  if (tensorGeFormat == FORMAT_HWCN) {
    channelIdx = 2;
    return SUCCESS;
  }
  return FAILED;
}

Status PatternFusionUtil::ParseNChannelIdx(ge::GeTensorDesc& tensorDesc, size_t& channelIdx) {
  ge::Format tensorGeFormat = tensorDesc.GetOriginFormat();
  if (tensorGeFormat == FORMAT_NCHW) {
    channelIdx = 0;
    return SUCCESS;
  }
  if (tensorGeFormat == FORMAT_NHWC) {
    channelIdx = 0;
    return SUCCESS;
  }
  if (tensorGeFormat == FORMAT_HWCN) {
    channelIdx = 3;
    return SUCCESS;
  }
  return FAILED;
}

Status PatternFusionUtil::GenGroupPaddingTensor(ge::GeTensorDesc& inTensor, ge::GeTensorDesc& outTensor, int64_t groups,
                                                const NodePtr& weightNode) {
  OpDescPtr weightOpDesc = weightNode->GetOpDesc();
  GeTensorDesc weightOutputDesc = weightOpDesc->GetOutputDesc(0);
  GeShape weightOutputShape = weightOutputDesc.GetShape();

  outTensor.SetOriginFormat(weightOutputDesc.GetOriginFormat());
  outTensor.SetFormat(weightOutputDesc.GetFormat());
  outTensor.SetDataType(weightOutputDesc.GetDataType());
  outTensor.SetOriginDataType(weightOutputDesc.GetOriginDataType());

  inTensor = outTensor;
  inTensor.SetOriginShape(weightOutputDesc.GetOriginShape());
  inTensor.SetShape(weightOutputDesc.GetShape());

  size_t outChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != ParseChannelIdx(weightOutputDesc, outChannelIdx),
                    OP_LOGE(weightOpDesc->GetType().c_str(),
                            "The original format of node[%s, %s]'s output0 is %s, which is unsupportable.",
                            weightOpDesc->GetName().c_str(), weightOpDesc->GetType().c_str(),
                            ge::TypeUtils::FormatToSerialString(weightOutputDesc.GetOriginFormat()).c_str()),
                    return FAILED);
  int cDIm = weightOutputShape.GetDim(outChannelIdx);
  weightOutputShape.SetDim(outChannelIdx, cDIm * groups);
  outTensor.SetOriginShape(weightOutputShape);
  outTensor.SetShape(weightOutputShape);
  return SUCCESS;
}

NodePtr PatternFusionUtil::AddGroupPaddingNode(ComputeGraph& graph, ge::GeTensorDesc& inTensor,
                                               ge::GeTensorDesc& outTensor, string nodeName) {
  ge::OpDescPtr opdesc;

  // add group padding node, set shape & format & dtype
  FUSION_PASS_MAKE_SHARED(opdesc = std::make_shared<ge::OpDesc>(nodeName, "GroupPadding"), return nullptr);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != opdesc->AddInputDesc("group_padding_input", inTensor),
                    OP_LOGE(opdesc->GetType().c_str(), "AddInputDesc node %s failed", opdesc->GetName().c_str()),
                    return nullptr);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != opdesc->AddOutputDesc("group_padding_output", outTensor),
                    OP_LOGE(opdesc->GetType().c_str(), "AddOutputDesc node %s failed", opdesc->GetName().c_str()),
                    return nullptr);

  return graph.AddNode(opdesc);
}

Status PatternFusionUtil::ProcessGroupPadding(ComputeGraph& graph, const NodePtr& groupConvNode, int64_t groups) {
  FUSION_PASS_CHECK(groupConvNode == nullptr, OP_LOGE("ProcessGroupPadding", "The groupConvNode is nullptr."),
                    return FAILED);

  ge::GeTensorDesc inTensor;
  ge::GeTensorDesc outTensor;
  string curOpType = groupConvNode->GetType();
  FUSION_PASS_CHECK(
      groupConvNode->GetInAllNodes().size() < 2,
      OP_LOGE(curOpType.c_str(),
              "The number of input of the node[name=%s, type=%s] is less than 2, there is no weight input.",
              groupConvNode->GetName().c_str(), groupConvNode->GetType().c_str()),
      return FAILED);
  NodePtr weightNode = groupConvNode->GetInAllNodes().at(1);
  FUSION_PASS_CHECK(weightNode == nullptr,
                    OP_LOGE(curOpType.c_str(), "Failed to get the weight of the node[name=%s, type=%s].",
                            groupConvNode->GetName().c_str(), groupConvNode->GetType().c_str()),
                    return FAILED);
  NodePtr paddingNode;
  string nodeName = groupConvNode->GetName() + "_groupPadding";
  ge::AttrUtils::SetInt(groupConvNode->GetOpDesc(), "groups", 1);

  // reset tensor
  FUSION_PASS_CHECK(SUCCESS != GenGroupPaddingTensor(inTensor, outTensor, groups, weightNode),
                    OP_LOGE(curOpType.c_str(), "Generate group padding tensor failed."), return FAILED);
  FUSION_PASS_CHECK(
      groupConvNode->GetOpDesc()->UpdateInputDesc(1, outTensor),
      OP_LOGE(curOpType.c_str(), "Update node[%s]'s first input failed.", groupConvNode->GetName().c_str()),
      return FAILED);

  // add groupPadding node
  paddingNode = AddGroupPaddingNode(graph, inTensor, outTensor, nodeName);
  ge::AttrUtils::SetInt(paddingNode->GetOpDesc(), "groups", groups);
  FUSION_PASS_CHECK(paddingNode == nullptr, OP_LOGE(curOpType.c_str(), "Create group padding node failed."),
                    return FAILED);

  ge::GraphUtils::RemoveEdge(weightNode->GetOutDataAnchor(0), groupConvNode->GetInDataAnchor(1));

  // link weight->groupPadding->conv
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(weightNode->GetOutDataAnchor(0), paddingNode->GetInDataAnchor(0)),
      OP_LOGE(curOpType.c_str(), "Add edge from src node[%s] to dst node[%s] failed.", weightNode->GetName().c_str(),
              paddingNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(paddingNode->GetOutDataAnchor(0), groupConvNode->GetInDataAnchor(1)),
      OP_LOGE(curOpType.c_str(), "Add edge from src node[%s] to dst node[%s] failed.", paddingNode->GetName().c_str(),
              groupConvNode->GetName().c_str()),
      return FAILED);
  return SUCCESS;
}

/**
 * @ingroup fe
 * @brief add a control edge from source node to dest node
 */
Status PatternFusionUtil::LinkControlEdge(ge::NodePtr srcNode, ge::NodePtr dstNode) {
  FUSION_PASS_CHECK(srcNode == nullptr, OP_LOGE("LinkControlEdge", "srcNode is null, LinkControlEdge failed."),
                    return FAILED);
  FUSION_PASS_CHECK(dstNode == nullptr, OP_LOGE("LinkControlEdge", "dstNode is null, LinkControlEdge failed."),
                    return FAILED);
  auto destInAnchor = srcNode->GetInControlAnchor();
  FUSION_PASS_CHECK(destInAnchor == nullptr,
                    OP_LOGE("LinkControlEdge", "destInAnchor is null, LinkControlEdge failed."), return FAILED);
  for (auto peerOutControlAnchor : destInAnchor->GetPeerOutControlAnchors()) {
    if (peerOutControlAnchor->GetOwnerNode()->GetOpDesc()->GetType() == STREAMSWITCH) {
      // unlink the out anchor of STREAMSWITCH and the in anchor of dest node
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(peerOutControlAnchor, destInAnchor) != SUCCESS,
                        OP_LOGE(STREAMSWITCH, "remove edge failed, LinkControlEdge failed."), return FAILED);
      // add a control edge of the out anchor of STREAMSWITCH and the in control
      // anchor of dest node
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peerOutControlAnchor, dstNode->GetInControlAnchor()) != SUCCESS,
                        OP_LOGE(STREAMSWITCH,
                                "add anchor to in control anchor failed, LinkControlEdge "
                                "failed."),
                        return FAILED);
    }
  }
  return SUCCESS;
}

Status PatternFusionUtil::RemoveInputEdge(ge::NodePtr node) {
  // remove input data edge
  string curOpType = node->GetType();
  string curOpName = node->GetName();
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr,
                      OP_LOGE(curOpType.c_str(), "Node[%s]: indataAnchor is null", curOpName.c_str()), return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preOutDataAnchor == nullptr,
                      OP_LOGE(curOpType.c_str(), "Node[%s]: outdataAnchor is null", curOpName.c_str()), return FAILED);
    if (ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS) {
      OP_LOGW(curOpType.c_str(), "Node[%s]: remove inputdata edge error", curOpName.c_str());
      return FAILED;
    }
  }
  // remove input control edge
  ge::InControlAnchorPtr inControlAnchor = node->GetInControlAnchor();
  if (inControlAnchor != nullptr) {
    for (ge::OutControlAnchorPtr srcAnchor : inControlAnchor->GetPeerOutControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(srcAnchor, inControlAnchor) != ge::GRAPH_SUCCESS) {
        OP_LOGW(curOpType.c_str(), "Node[%s]: disconnect node input control anchor failed.", curOpName.c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status PatternFusionUtil::CopyMultiReferenceConstNode(ge::ComputeGraph &graph, ge::NodePtr nodePtr) {
  for (size_t index = 0; index < nodePtr->GetAllInDataAnchors().size(); index++) {
    if (nodePtr->GetInDataAnchor(index) == nullptr || nodePtr->GetInDataAnchor(index)->GetPeerOutAnchor() == nullptr ||
            nodePtr->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr inputNode = nodePtr->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
    // 1. find const node
    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(inputNode) == "Const") {
      // 2. if const node has only one reference, return SUCCESS
      bool isSingleReference = true;
      for (auto outAnchor : inputNode->GetAllOutDataAnchors()) {
        if (outAnchor->GetPeerInDataAnchors().size() != 1) {
          isSingleReference = false;
          break;
        }
      }
      if (isSingleReference) {
        OP_LOGD(inputNode->GetType().c_str(), "Node %s is single reference, no need to be replaced.",
                inputNode->GetName().c_str());
        return SUCCESS;
      }

      // 3. copy a new const op desc from old const node
      auto opDesc = inputNode->GetOpDesc();
      auto copyOpDesc = ge::AttrUtils::CopyOpDesc(opDesc);
      if (copyOpDesc == nullptr) {
        OP_LOGW(inputNode->GetType().c_str(), "The copyNode's opdesc is null, fusion failed.");
        return NOT_CHANGED;
      }

      // 4. modify the new const node's name
      copyOpDesc->SetName(copyOpDesc->GetName() + nodePtr->GetName());

      // 5. add new const node to graph
      ge::NodePtr newConstNode = graph.AddNode(copyOpDesc);
      if (newConstNode == nullptr) {
        OP_LOGW(inputNode->GetType().c_str(), "The newConstNode is null, add node failed.");
        return NOT_CHANGED;
      }

      // 6. delete edge between convNode and old const node
      nodePtr->GetInDataAnchor(index)->UnlinkAll();

      // 7. add edge between convNode and new const node
      if (ge::GraphUtils::AddEdge(newConstNode->GetOutDataAnchor(0), nodePtr->GetInDataAnchor(index)) != SUCCESS) {
        OP_LOGW(inputNode->GetType().c_str(), "Fail to add input[%d] edge for new Const node.", index);
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status PatternFusionUtil::SetWeightByIndex(ge::NodePtr node, ge::GeTensorPtr tensor, const uint32_t &index,
                                           ge::ComputeGraph &graph) {
  if (node == nullptr || tensor == nullptr) {
    return FAILED;
  }
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(tensor);
  FUSION_PASS_CHECK(const_opdesc == nullptr,
                    OP_LOGE(node->GetName().c_str(), "Fail to create const op desc."),
                    return FAILED);
  ge::NodePtr const_node = graph.AddNode(const_opdesc);
  FUSION_PASS_CHECK(const_node == nullptr,
                    OP_LOGE(const_opdesc->GetName().c_str(), "Fail to add const node."),
                    return FAILED);
  FUSION_PASS_CHECK(node->AddLinkFrom(index, const_node) != ge::GRAPH_SUCCESS,
                    OP_LOGE(node->GetName().c_str(), "Fail to link const node with conv node."),
                    return FAILED);
  return SUCCESS;
}

Status PatternFusionUtil::UpdateInputAndOutputName(const ge::OpDescPtr opDescPtr) {
  if (opDescPtr == nullptr) {
    return FAILED;
  }
  auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", opDescPtr->GetType());
  if (node_op.IsEmpty()) {
    OP_LOGW("Get op from OperatorFactory fail. opType: %s", opDescPtr->GetType().c_str());
  } else {
    OP_LOGD("Get op from OperatorFactory success. opType: %s", opDescPtr->GetType().c_str());
    auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
    if (temp_op_desc == nullptr) {
      OP_LOGE("temp op desc is null");
      return FAILED;
    }
    if (!opDescPtr->UpdateInputName(temp_op_desc->GetAllInputName())) {
      OP_LOGW("Verify UpdateInputName failed");
    }
    if (!opDescPtr->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
      OP_LOGW("Verify UpdateOutputName failed");
    }
  }
  node_op.BreakConnect();
  return SUCCESS;
}

// unknown shape value
const int64_t UNKNOWN_SHAPE_VALUE = -1;
const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;

bool PatternFusionUtil::IsUnknownShape(const int64_t& dims) {
  if (dims == UNKNOWN_SHAPE_VALUE || dims == SHAPE_UNKNOWN_DIM_NUM) {
    return true;
  }
  return false;
}

ge::NodePtr PatternFusionUtil::InsertInputNode(ge::ComputeGraph &graph, ge::NodePtr &src_node, const string &op_type,
                                               const int32_t &index, std::atomic<uint64_t> &name_id) {
  ge::OpDescPtr op_desc;
  ge::NodePtr single_node = nullptr;
  int32_t input_size = static_cast<int32_t>(src_node->GetAllInDataAnchorsSize());
  if (input_size <= index) {
    OP_LOGD(src_node->GetName().c_str(), "insert node anchor index[%d] is out of input's range.", index);
    return nullptr;
  }
  FUSION_PASS_MAKE_SHARED(op_desc = std::make_shared<ge::OpDesc>(src_node->GetName() + "_" +
          op_type + "_" + std::to_string(name_id), op_type), return nullptr);
  auto in_anchor = src_node->GetInAnchor(index);
  FUSION_PASS_CHECK(in_anchor == nullptr,
                    OP_LOGE(src_node->GetName().c_str(), "data anchor is nullptr."), return nullptr);
  auto peer_anchors = in_anchor->GetPeerAnchors();
  FUSION_PASS_CHECK(peer_anchors.empty() || peer_anchors.size() > 1,
                    OP_LOGE(src_node->GetName().c_str(), "out data anchor's peer in anchor is empty or more than 1."),
                    return nullptr);
  auto peer_anchor = peer_anchors.at(0);
  auto peer_anchor_idx = peer_anchor->GetIdx();
  GeTensorDesc input_desc;
  GeTensorDesc output_desc;
  if (peer_anchor_idx >= 0) {
    input_desc = peer_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peer_anchor_idx);
  }
  if (index >= 0) {
    output_desc = src_node->GetOpDesc()->GetInputDesc(index);
  }
  (void)op_desc->AddInputDesc(input_desc);
  (void)op_desc->AddOutputDesc(output_desc);
  single_node = graph.AddNode(op_desc);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(peer_anchor, in_anchor) != SUCCESS,
                    OP_LOGE(src_node->GetName().c_str(), "remove edge failed."), return nullptr);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_anchor, single_node->GetInAnchor(0)) != SUCCESS,
                    OP_LOGE(src_node->GetName().c_str(), "add edge failed."), return nullptr);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(single_node->GetOutAnchor(0), in_anchor) != SUCCESS,
                    OP_LOGE(src_node->GetName().c_str(), "add edge failed."), return nullptr);
  return single_node;
}

ge::NodePtr PatternFusionUtil::InsertOutputNode(ge::ComputeGraph &graph, ge::NodePtr &src_node, const string &op_type,
                                                const int32_t &index, std::atomic<uint64_t> &name_id) {
  ge::OpDescPtr op_desc;
  ge::NodePtr single_node = nullptr;
  int32_t output_size = static_cast<int32_t>(src_node->GetAllOutDataAnchorsSize());
  if (output_size <= index) {
    OP_LOGD(src_node->GetName().c_str(), "insert node anchor index[%d] is out of output's range.", index);
    return nullptr;
  }
  FUSION_PASS_MAKE_SHARED(op_desc = std::make_shared<ge::OpDesc>(src_node->GetName() + "_" +
          op_type + "_" + std::to_string(name_id), op_type), return nullptr);
  auto out_anchor = src_node->GetOutAnchor(index);
  FUSION_PASS_CHECK(out_anchor == nullptr,
                    OP_LOGE(src_node->GetName().c_str(), "data anchor is nullptr."), return nullptr);
  auto peer_anchors = out_anchor->GetPeerAnchors();
  for (auto peer_anchor : peer_anchors) {
    FUSION_PASS_CHECK(peer_anchor == nullptr,
                      OP_LOGE(src_node->GetName().c_str(), "data anchor is nullptr."), return nullptr);
  }
  FUSION_PASS_CHECK(peer_anchors.empty(),
                    OP_LOGE(src_node->GetName().c_str(), "out data anchor's peer in anchor is empty."), return nullptr);
  GeTensorDesc output_desc;
  GeTensorDesc input_desc;
  if (index >= 0) {
    auto peer_anchor_idx = peer_anchors.at(0)->GetIdx();
    output_desc = peer_anchors.at(0)->GetOwnerNode()->GetOpDesc()->GetInputDesc(peer_anchor_idx);
    input_desc = src_node->GetOpDesc()->GetOutputDesc(index);
  }
  (void) op_desc->AddInputDesc(input_desc);
  (void) op_desc->AddOutputDesc(output_desc);
  single_node = graph.AddNode(op_desc);
  for (auto peer_anchor : peer_anchors) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchor, peer_anchor) != SUCCESS,
                      OP_LOGE(src_node->GetName().c_str(), "remove edge failed."), return nullptr);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(single_node->GetOutAnchor(0), peer_anchor) != SUCCESS,
                      OP_LOGE(src_node->GetName().c_str(), "add edge failed."), return nullptr);
  }
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(out_anchor, single_node->GetInAnchor(0)) != SUCCESS,
                    OP_LOGE(src_node->GetName().c_str(), "add edge failed."), return nullptr);
  return single_node;
}

/*
 * insert single node before of behind src node
 * if insert input node, make sure the input anchor of src node have only one peer anchor
 * if insert output node, we will connect all peer anchors of src node's out anchor to single node
 *
 *         input node                               src node
 *              |                                       |
 *         single node                             single node
 *              |                                   /   |   \
 *          src node                            node1 node2 node3
 */
ge::NodePtr PatternFusionUtil::InsertSingleNode(ge::ComputeGraph &graph, ge::NodePtr &src_node, const string &op_type,
        const bool &is_input, const int32_t &index, vector<ge::NodePtr> &fusion_nodes) {
  ge::NodePtr single_node = nullptr;
  static std::atomic<uint64_t> name_id(0);
  if (is_input) {
    single_node = InsertInputNode(graph, src_node, op_type, index, name_id);
    FUSION_PASS_CHECK(single_node == nullptr,
            OP_LOGE(src_node->GetName().c_str(), "Insert input node failed."), return nullptr);
    name_id.fetch_add(1, std::memory_order_relaxed);
    fusion_nodes.push_back(single_node);
  } else {
    single_node = InsertOutputNode(graph, src_node, op_type, index, name_id);
    FUSION_PASS_CHECK(single_node == nullptr,
            OP_LOGE(src_node->GetName().c_str(), "Insert output node failed."), return nullptr);
    name_id.fetch_add(1, std::memory_order_relaxed);
    fusion_nodes.push_back(single_node);
  }
  return single_node;
}

Status PatternFusionUtil::InsertSliceDNodes(ComputeGraph& graph, NodePtr srcNode, unsigned int weightIdx,
                                            const vector<NodePtr>& newConvNodes, int64_t group, size_t sliceDimIdx) {
  string curOpType = "SliceD";
  OpDescPtr convDesc = srcNode->GetOpDesc();
  // InsertSliceD conv i newconv
  vector<NodePtr> newSlicedNodes;
  // Traverse every output of const node to find sliced node, if have, record sliced
  for (auto convInAnchor : srcNode->GetInAllNodes().at(weightIdx)->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (convInAnchor->GetOwnerNode()->GetType() == "SliceD") {
      newSlicedNodes.push_back(convInAnchor->GetOwnerNode());
    }
  }
  // If const node don't have sliceD output, create a sliceD node
  if (newSlicedNodes.empty()) {
    FUSION_PASS_CHECK((int64_t)(newConvNodes.size()) < group,
                      OP_LOGE(curOpType.c_str(), "Node's size less then group, fusion failed."),
                      return FAILED);
    for (unsigned int newSrcNodeIdx = 0; newSrcNodeIdx < group; newSrcNodeIdx++) {
      NodePtr slicedNode = nullptr;
      OpDescPtr sliceDesc = std::make_shared<ge::OpDesc>(
              srcNode->GetName() + to_string(weightIdx) + "_slice" + to_string(newSrcNodeIdx), "SliceD");
      FUSION_PASS_CHECK(sliceDesc == nullptr, OP_LOGE(curOpType.c_str(), "sliceNdoe's OpDesc is null, fusion failed."),
                        return FAILED);
      // Get weight of srcNode
      NodePtr constNode = srcNode->GetInDataAnchor(weightIdx)->GetPeerOutAnchor()->GetOwnerNode();
      int constOutIdx = srcNode->GetInDataAnchor(weightIdx)->GetPeerOutAnchor()->GetIdx();
      // sliceD's input desc should be the same as weight's output desc
      GeTensorDesc inputDesc = constNode->GetOpDesc()->GetOutputDesc(constOutIdx);
      GeShape inputShape = inputDesc.GetShape();
      GeShape sliceOutShape = inputShape;
      // check the dim info of sliceD node's input desc
      FUSION_PASS_CHECK(
              inputShape.GetDimNum() < 1,
              OP_LOGE(curOpType.c_str(), "sliceNdoe's dim:[%ld] less than one, fusion failed.", inputShape.GetDimNum()),
              return FAILED);
      FUSION_PASS_CHECK(group != 0 && inputShape.GetDim(sliceDimIdx) % group != 0,
                        OP_LOGE(curOpType.c_str(), "sliceNdoe's dim[%d]:[%ld] divide group(%ld) != 0, fusion failed.",
                                sliceDimIdx, inputShape.GetDim(sliceDimIdx), group),
                        return FAILED);
      // deivide the sliceDimIdx axis data into group
      sliceOutShape.SetDim(sliceDimIdx, inputShape.GetDim(sliceDimIdx) / group);
      // set sliceD's output desc info
      GeTensorDesc sliceOutDesc = inputDesc;
      sliceOutDesc.Update(sliceOutShape, constNode->GetOpDesc()->GetOutputDesc(constOutIdx).GetFormat(),
                          constNode->GetOpDesc()->GetOutputDesc(constOutIdx).GetDataType());
      sliceOutDesc.SetOriginShape(sliceOutShape);
      sliceDesc->AddInputDesc(inputDesc);
      sliceDesc->AddOutputDesc(sliceOutDesc);
      // set newConvNode's input info, it should be the same as sliceD's output desc
      newConvNodes[newSrcNodeIdx]->GetOpDesc()->UpdateInputDesc(weightIdx, sliceOutDesc);
      // set sliceD's attr: offsets & size
      vector<int64_t> vectorOffsets(inputShape.GetDims().size(), 0);
      vector<int64_t> vectorSize;
      vectorOffsets[sliceDimIdx] = newSrcNodeIdx * inputShape.GetDim(sliceDimIdx) / group;
      vectorSize = inputShape.GetDims();
      vectorSize[sliceDimIdx] /= group;
      FUSION_PASS_CHECK(!AttrUtils::SetListInt(sliceDesc, "offsets", vectorOffsets),
                        OP_LOGE(curOpType.c_str(), "Set SliceD's attr offsets failed."), return FAILED);
      FUSION_PASS_CHECK(!AttrUtils::SetListInt(sliceDesc, "size", vectorSize),
                        OP_LOGE(curOpType.c_str(), "Set SliceD's attr size failed."), return FAILED);
      FUSION_PASS_CHECK(!AttrUtils::SetInt(sliceDesc, "index", newSrcNodeIdx),
                        OP_LOGE(curOpType.c_str(), "Set SliceD's attr index failed."), return FAILED);
      slicedNode = graph.AddNode(sliceDesc);
      FUSION_PASS_CHECK(slicedNode == nullptr, OP_LOGE(curOpType.c_str(), "sliceNdoe is null, fusion failed."),
                        return FAILED);
      newSlicedNodes.push_back(slicedNode);
      // connect weight and slice node
      FUSION_PASS_CHECK(
              ge::GRAPH_SUCCESS != GraphUtils::AddEdge(srcNode->GetInAllNodes().at(weightIdx)->GetOutDataAnchor(0),
                                                       newSlicedNodes[newSrcNodeIdx]->GetInDataAnchor(0)),
              OP_LOGE(curOpType.c_str(), "add concat to output edge fail"), return FAILED);
      OP_LOGD(curOpType.c_str(), "Add edge from [%s] to [%s]",
              srcNode->GetInAllNodes().at(weightIdx)->GetName().c_str(),
              newSlicedNodes[newSrcNodeIdx]->GetName().c_str());
      // connect slice's output to each new conv node
      FUSION_PASS_CHECK(
              ge::GRAPH_SUCCESS != GraphUtils::AddEdge(newSlicedNodes[newSrcNodeIdx]->GetOutDataAnchor(0),
                                                       newConvNodes[newSrcNodeIdx]->GetInDataAnchor(weightIdx)),
              OP_LOGE(curOpType.c_str(), "add concat to output edge fail"), return FAILED);
      OP_LOGD(curOpType.c_str(), "Add edge from [%s] to [%s]", newSlicedNodes[newSrcNodeIdx]->GetName().c_str(),
              newConvNodes[newSrcNodeIdx]->GetName().c_str());
    }
  } else {
    // if there is sliceD node, just connect sliceD to new conv node
    FUSION_PASS_CHECK(newSlicedNodes.size() != (unsigned int)group,
                      OP_LOGE(curOpType.c_str(), "Node[%s] should have [%d] sliceD outputs",
                              srcNode->GetInAllNodes().at(weightIdx)->GetName().c_str(), group),
                      return FAILED);
    for (int newSrcNodeIdx = 0; newSrcNodeIdx < group; newSrcNodeIdx++) {
      for (auto sliceNode : newSlicedNodes) {
        int64_t idx = -1;
        FUSION_PASS_CHECK(!AttrUtils::GetInt(sliceNode->GetOpDesc(), "index", idx),
                          OP_LOGE(curOpType.c_str(), "Get SliceD's attr index failed."), return FAILED);
        if (idx == newSrcNodeIdx) {
          // connect slice's output to each new conv node
          FUSION_PASS_CHECK(
                  ge::GRAPH_SUCCESS != GraphUtils::AddEdge(sliceNode->GetOutDataAnchor(0),
                                                           newConvNodes[newSrcNodeIdx]->GetInDataAnchor(weightIdx)),
                  OP_LOGE(curOpType.c_str(), "add concat to output edge fail"), return FAILED);
          newConvNodes[newSrcNodeIdx]->GetOpDesc()->UpdateInputDesc(weightIdx,
                                                                    sliceNode->GetOpDesc()->GetOutputDesc(0));
          continue;
        }
      }
    }
  }
  return SUCCESS;
}

}  // namespace fe
