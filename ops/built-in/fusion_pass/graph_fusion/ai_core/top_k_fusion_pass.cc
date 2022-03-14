/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file top_k_fusion_pass.cpp
 * \brief if dim = -1. TopK --> TopKD.
 * \brief if dim != -1. TransposeD -> TopkD -> TransposeD.
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fp16_t.hpp"
#include "common/util/platform_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "top_k_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string kPatternTopK = "topk";
static const string kConstantOp = "Constant";
static const string kPatternTranspose = "TransposeD";

Status PermVecGen(int64_t dim_size, int64_t dim_aim, vector<int64_t>& perm) {
  if (dim_aim > dim_size) {
    return FAILED;
  }
  for (int64_t i = 0; i < dim_size; i++) {
    perm.push_back(i);
  }
  swap(perm[dim_aim], perm[dim_size - 1]);
  return SUCCESS;
}

Status AssitHelp(const int32_t n, uint16_t* output, bool is_segment_sort=false) {
  for (int32_t i = 0; i < n; ++i) {
    fp16_t t;
    t.val = 0;
    t = i;
    output[i] = t.val;
  }
  if (!is_segment_sort) {
    for (int32_t i = 0; i < n; ++i) {
      fp16_t t;
      t.val = 0;
      t = i;
      int32_t idx = t;
      int32_t gap = i - idx;
      fp16_t tmp;
      tmp.val = 0;
      tmp = gap;
      output[i + n] = tmp.val;
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> TopKFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // TopK->TopKD
  FusionPattern* pattern = new (nothrow) FusionPattern("TopKFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);
  // define origin graph
  pattern->AddOpDesc(kPatternTopK, {"TopK", "TopKV2"}).SetOutput(kPatternTopK);
  patterns.push_back(pattern);
  return patterns;
}

bool TopKFusionPass::CheckMultiCoreSegment(NodePtr& topk_node, SegmentCalcParams& calcParams) {
  // init soc version params
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info,
                                                                                     optional_info) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "CheckMultiCoreSegment: Get platform_info failed."),
                    return false);
  // check input_data shape
  OpDescPtr topk_desc = topk_node->GetOpDesc();
  GeTensorDesc input_data_desc = topk_desc->GetInputDesc(0);
  vector<int64_t> input_shape = input_data_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(input_shape.size() != 1,
                    OP_LOGW(kFusedOpType.c_str(), "CheckMultiCoreSegment: Input shape dims not support."),
                    return false);
  // check attr dim
  int64_t dim_aim;
  if (!AttrUtils::GetInt(topk_desc, "dim", dim_aim)) {
    dim_aim = -1;
    OP_LOGI(kFusedOpType.c_str(), "CheckMultiCoreSegment: Cannot get attr dim, use default value.");
  }
  if (dim_aim < 0) {
    dim_aim = input_shape.size() + dim_aim;
  }
  FUSION_PASS_CHECK(dim_aim != 0,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "CheckMultiCoreSegment: The attr dim is invalid."),
                    return false);

  calcParams.data_size = input_shape[dim_aim];
  if (optional_info.soc_version.find("Ascend310") != string::npos ||
      optional_info.soc_version.find("Ascend710") != string::npos ||
      optional_info.soc_version.find("Ascend910") != string::npos) {
    // check data type
    FUSION_PASS_CHECK(input_data_desc.GetDataType() != ge::DT_FLOAT16,
                      OP_LOGW(kFusedOpType.c_str(), "CheckMultiCoreSegment: Input data type not support."),
                      return false);
    // check data size
    FUSION_PASS_CHECK(calcParams.data_size <= calcParams.core_min_num,
                      OP_LOGW(kFusedOpType.c_str(), "CheckMultiCoreSegment: Input data size not support."),
                      return false);
    calcParams.ai_core_num = platform_info.soc_info.ai_core_cnt;
  } else if (optional_info.soc_version.find("Ascend920") != string::npos) {
    calcParams.soc_version = "Ascend920";
    calcParams.core_align_num = 32;
    calcParams.pro_repeat_num = 32;
    calcParams.core_min_num = 12288;
    FUSION_PASS_CHECK(calcParams.k_num <= calcParams.core_min_num / 2,
                      OP_LOGW(kFusedOpType.c_str(), "The attr k_num not support."),
                      return false);
    // check data type
    if (input_data_desc.GetDataType() == ge::DT_FLOAT16) {
      calcParams.pro_data_num = 4;
    } else if (input_data_desc.GetDataType() == ge::DT_FLOAT) {
      calcParams.pro_data_num = 2;
    } else {
      OP_LOGW(kFusedOpType.c_str(), "Input data type not support");
      return false;
    }
    // check data size
    FUSION_PASS_CHECK(calcParams.data_size <= calcParams.core_min_num,
                      OP_LOGW(kFusedOpType.c_str(), "Input data size not support."),
                      return false);
    calcParams.ai_core_num = platform_info.soc_info.vector_core_cnt;
  } else {
    OP_LOGW(kFusedOpType.c_str(), "CheckMultiCoreSegment: SocVersion not support.");
    return false;
  }
  return true;
}

Status TopKFusionPass::AddMultiMergeNode(ComputeGraph& graph, NodePtr& topk_node, NodePtr& segmentsort_node,
                                         int64_t segment_num, SegmentCalcParams& calcParams,
                                         vector<NodePtr>& fusion_nodes) {
  int64_t index = 0;
  NodePtr input_node = segmentsort_node;
  GeTensorDesc output_proposal_desc = topk_node->GetOpDesc()->GetOutputDesc(0);
  GeTensorDesc output_index_desc = topk_node->GetOpDesc()->GetOutputDesc(1);
  while (segment_num > calcParams.merge_channel) {
    // define multimerge Opdesc
    std::shared_ptr<ge::OpDesc> multimerge_desc = std::make_shared<ge::OpDesc>(topk_node->GetName() + "_multimerge_" + to_string(index), "MultiMerge");
    FUSION_PASS_CHECK(multimerge_desc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The multimerge_desc is null, Build MultiMerge Op failed."),
                      return FAILED);
    GeTensorDesc input_proposal_desc = input_node->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(multimerge_desc->AddInputDesc("input_proposal", input_proposal_desc) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "MultiMerge add input_proposal desc failed."), return FAILED);
    // update output proposal shape and dtype
    int64_t channel_num = input_proposal_desc.GetShape().GetDim(0);
    int64_t data_num = input_proposal_desc.GetShape().GetDim(1);
    int64_t result_data_num = data_num * calcParams.merge_channel;
    int64_t k_align_num = (calcParams.k_num + calcParams.pro_repeat_num - 1) / calcParams.pro_repeat_num * calcParams.pro_repeat_num;
    if (k_align_num < result_data_num) {
      result_data_num = k_align_num;
    }
    result_data_num = result_data_num + calcParams.pro_repeat_num;

    int64_t ai_core_num = channel_num / calcParams.merge_channel;
    if (ai_core_num > calcParams.merge_channel) {
      ai_core_num = (ai_core_num + calcParams.merge_channel - 1) / calcParams.merge_channel * calcParams.merge_channel;
    }
    vector<int64_t> output_proposal_shape;
    output_proposal_shape.push_back(ai_core_num);
    output_proposal_shape.push_back(result_data_num);
    output_proposal_shape.push_back(calcParams.pro_data_num);
    output_proposal_desc.SetShape(GeShape(output_proposal_shape));
    output_proposal_desc.SetOriginShape(GeShape(output_proposal_shape));
    output_proposal_desc.SetDataType(input_proposal_desc.GetDataType());
    FUSION_PASS_CHECK(multimerge_desc->AddOutputDesc("output_proposal", output_proposal_desc) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "MultiMerge add output_proposal desc failed."), return FAILED);
    vector<int64_t> output_index_shape = {1};
    output_index_desc.SetShape(GeShape(output_index_shape));
    output_index_desc.SetOriginShape(GeShape(output_index_shape));
    output_index_desc.SetDataType(DT_INT32);
    FUSION_PASS_CHECK(multimerge_desc->AddOutputDesc("output_index", output_index_desc) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "MultiMerge add output_index desc failed."), return FAILED);

    NodePtr multiMergeNode = graph.AddNode(multimerge_desc);
    FUSION_PASS_CHECK(multiMergeNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The multiMergeNode is null, Build MultiMerge Op failed."),
                      return PARAM_INVALID);
    Operator multiMerge = OpDescUtils::CreateOperatorFromNode(multiMergeNode);
    multiMerge.SetAttr("k_num", calcParams.k_num);
    fusion_nodes.push_back(multiMergeNode);
    index++;
    segment_num /= calcParams.merge_channel;
    input_node = multiMergeNode;
  }

  // define last multimerge(include_index) Opdesc
  std::shared_ptr<ge::OpDesc> last_multimerge_desc = std::make_shared<ge::OpDesc>(topk_node->GetName() + "_multimerge_" + to_string(index), "MultiMerge");
  FUSION_PASS_CHECK(last_multimerge_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The last multimerge_desc is null, Build MultiMerge Op failed."),
                    return FAILED);
  GeTensorDesc input_proposal_desc = input_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(last_multimerge_desc->AddInputDesc("input_proposal", input_proposal_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "LastMultiMerge add input_proposal desc failed."), return FAILED);
  vector<int64_t> output_shape;
  output_shape.push_back(calcParams.k_num);
  output_proposal_desc.SetShape(GeShape(output_shape));
  output_proposal_desc.SetOriginShape(GeShape(output_shape));
  output_proposal_desc.SetDataType(input_proposal_desc.GetDataType());
  FUSION_PASS_CHECK(last_multimerge_desc->AddOutputDesc("output_proposal", output_proposal_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "LastMultiMerge add output_proposal desc failed."), return FAILED);
  output_index_desc.SetShape(GeShape(output_shape));
  output_index_desc.SetOriginShape(GeShape(output_shape));
  output_index_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(last_multimerge_desc->AddOutputDesc("output_index", output_index_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "LastMultiMerge add output_index desc failed."), return FAILED);

  NodePtr lastMultiMergeNode = graph.AddNode(last_multimerge_desc);
  FUSION_PASS_CHECK(lastMultiMergeNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The last multiMergeNode is null, Build MultiMerge Op failed."),
                    return PARAM_INVALID);
  Operator lastMultiMerge = OpDescUtils::CreateOperatorFromNode(lastMultiMergeNode);
  lastMultiMerge.SetAttr("k_num", calcParams.k_num);
  lastMultiMerge.SetAttr("include_index", true);
  fusion_nodes.push_back(lastMultiMergeNode);
  return SUCCESS;
}

/*
input_data  input_index          input_data  input_index
    \           /                    \           /
         topk           --->          segment_sort
    /          \                           |
output_data output_index             multi_merge * N
                                           |
                                 multi_merge(include_index)
                                      /          \
                                output_data output_index
*/
Status TopKFusionPass::AddSegmentSortAndMergeNode(ComputeGraph& graph, NodePtr& topk_node,
                                                  SegmentCalcParams& calcParams, vector<NodePtr>& fusion_nodes) {
  OP_LOGI(kFusedOpType.c_str(), "AddSegmentSortAndMergeNode start.");
  // define segmentsort Opdesc
  std::shared_ptr<ge::OpDesc> segmentsort_desc = std::make_shared<ge::OpDesc>(topk_node->GetName() + "_segmentsort", "SegmentSort");
  FUSION_PASS_CHECK(segmentsort_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The segmentsort_desc is null, Build SegmentSort Op failed."),
                    return FAILED);
  GeTensorDesc input_data_desc = topk_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(segmentsort_desc->AddInputDesc("input_data", input_data_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SegmentSort add input_data desc failed."), return FAILED);
  GeTensorDesc output_desc = topk_node->GetOpDesc()->GetOutputDesc(0);
  // update segmentsort output shape
  int64_t ai_core_num = calcParams.ai_core_num;
  int64_t result_data_num = (calcParams.data_size + ai_core_num - 1) / ai_core_num;
  result_data_num = (result_data_num + calcParams.core_align_num - 1) / calcParams.core_align_num * calcParams.core_align_num;
  if (result_data_num < calcParams.core_min_num) {
    result_data_num = calcParams.core_min_num;
  }

  ai_core_num = (calcParams.data_size + result_data_num - 1) / result_data_num;
  if (ai_core_num > calcParams.merge_channel) {
    ai_core_num = (ai_core_num + calcParams.merge_channel - 1) / calcParams.merge_channel * calcParams.merge_channel;
  }
  result_data_num = result_data_num + calcParams.pro_repeat_num;
  vector<int64_t> output_proposal_shape;
  output_proposal_shape.push_back(ai_core_num);
  output_proposal_shape.push_back(result_data_num);
  output_proposal_shape.push_back(calcParams.pro_data_num);
  output_desc.SetShape(GeShape(output_proposal_shape));
  output_desc.SetOriginShape(GeShape(output_proposal_shape));
  output_desc.SetDataType(input_data_desc.GetDataType());
  FUSION_PASS_CHECK(segmentsort_desc->AddOutputDesc("output_proposal", output_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SegmentSort add output_proposal desc failed."),
                    return FAILED);

  NodePtr segmentSortNode = graph.AddNode(segmentsort_desc);
  FUSION_PASS_CHECK(segmentSortNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The segmentSortNode is null, Build SegmentSort Op failed."),
                    return PARAM_INVALID);

  Operator segmentSort = OpDescUtils::CreateOperatorFromNode(segmentSortNode);
  segmentSort.SetAttr("k_num", calcParams.k_num);
  fusion_nodes.push_back(segmentSortNode);

  // define multimerge Opdesc
  FUSION_PASS_CHECK(AddMultiMergeNode(graph, topk_node, segmentSortNode, ai_core_num, calcParams, fusion_nodes) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "AddMultiMergeNode failed, fusion failed."), return FAILED);

  // connect inputdata_node with segmentsort_node
  FUSION_PASS_CHECK(GraphUtils::AddEdge(topk_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                        segmentSortNode->GetInDataAnchor(0)) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add input_data node to segmentsort node edge failed."),
                    return FAILED);

  // create assist_seq and connect with segmentsort
  Status ret = SUCCESS;
  constexpr int64_t kAssistLen{2048};
  GeTensorPtr assit_ptr{nullptr};
  vector<int64_t> assist_dim_info = {kAssistLen};
  if (calcParams.soc_version == "Ascend920") {
    unique_ptr<int32_t[]> inputAssit(new (nothrow) int32_t[kAssistLen]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: InputAssit is NULL"), return FAILED);
    for (int32_t i = 0; i < kAssistLen; ++i) {
      inputAssit.get()[i] = i;
    }
    // define shape
    GeTensorDesc tensor_desc(GeShape(assist_dim_info), FORMAT_ND, DT_INT32);
    FUSION_PASS_MAKE_SHARED((assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                               kAssistLen * sizeof(int32_t))),
                            assit_ptr = nullptr;
                            return PARAM_INVALID);
  } else {
    unique_ptr<uint16_t[]> inputAssit(new (nothrow) uint16_t[kAssistLen]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: InputAssit is NULL"), return FAILED);
    ret = AssitHelp(kAssistLen, inputAssit.get(), true);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(kFusedOpType.c_str(), "Add SegmentSort: AssitHelp failed."), return NOT_CHANGED);

    // define shape
    GeTensorDesc tensor_desc(GeShape(assist_dim_info), FORMAT_ND, DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED((assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                               kAssistLen * sizeof(uint16_t))),
                            assit_ptr = nullptr;
                            return PARAM_INVALID);
  }

  vector<GeTensorPtr> weights = {assit_ptr};
  FUSION_PASS_CHECK(OpDescUtils::SetWeights(segmentSortNode, weights) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: SetWeights failed"), return FAILED);
  auto const_input_nodes = OpDescUtils::GetConstInputs(segmentSortNode);
  FUSION_PASS_CHECK(const_input_nodes.size() <= 0,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: GetConstInputs Error"),
                    return PARAM_INVALID);
  NodePtr const_input = const_input_nodes[0];
  FUSION_PASS_CHECK(const_input == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The const_input is null, Build SegmentSort Op failed."),
                    return PARAM_INVALID);
  auto const_input_desc = const_input->GetOpDesc();
  FUSION_PASS_CHECK(const_input_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The const_input_desc is null, Build SegmentSort Op failed."),
                    return PARAM_INVALID);
  const_input_desc->SetType(kConstantOp);

  // segmentsort --> multimerge * N --> last multimerge
  NodePtr input_node = segmentSortNode;
  for (int i = 1; i < fusion_nodes.size(); ++i) {
    FUSION_PASS_CHECK(GraphUtils::AddEdge(input_node->GetOutDataAnchor(0),
                                          fusion_nodes[i]->GetInDataAnchor(0)) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add segmentsort to multimerge edge failed."),
                      return FAILED);
    input_node = fusion_nodes[i];
  }

  // last multimerge --> next node
  NodePtr lastMergeNode = fusion_nodes[fusion_nodes.size() - 1];
  for (auto inDataAnchor : topk_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(topk_node->GetOutDataAnchor(0), inDataAnchor) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(lastMergeNode->GetOutDataAnchor(0), inDataAnchor) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add last multimerge output data to next node edge failed."),
                      return FAILED);
  }
  for (auto inDataAnchor : topk_node->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(topk_node->GetOutDataAnchor(1), inDataAnchor) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: Remove out index edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(lastMergeNode->GetOutDataAnchor(1), inDataAnchor) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add last multimerge output index to next node edge failed."),
                      return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(topk_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add SegmentSort: Remove topk node failed."),
                    return FAILED);
  OP_LOGI(kFusedOpType.c_str(), "AddSegmentSortAndMergeNode end.");
  return SUCCESS;
}

Status TopKFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  NodePtr topk_node = GetNodeFromMapping(kPatternTopK, mapping);
  FUSION_PASS_CHECK(topk_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The topk_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr topk_desc = topk_node->GetOpDesc();
  FUSION_PASS_CHECK(topk_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The topk_desc is null, fusion failed."),
                    return PARAM_INVALID);
  // may find TopKV2, use TopK instead
  topk_desc->SetType("TopK");
  auto input_desc_k = topk_desc->GetInputDesc(1);
  if (input_desc_k.GetDataType() == ge::DT_INT64) {
    input_desc_k.SetDataType(ge::DT_INT32);
    input_desc_k.SetOriginDataType(ge::DT_INT32);
    topk_desc->UpdateInputDesc(1, input_desc_k);
  }

  // The value of sorted cannot be false in aicore
  bool sorted = true;
  // attr sorted is optional
  AttrUtils::GetBool(topk_desc, "sorted", sorted);
  FUSION_PASS_CHECK(!sorted,
                    OP_LOGW(kFusedOpType.c_str(), "The value of sorted must be true in aicore, fusion failed."),
                    return NOT_CHANGED);

  // first input of topkv2 is non-constant, second is constant
  InDataAnchorPtr topk_anchor_ptr0 = topk_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(topk_anchor_ptr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The topk_anchor_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  OutDataAnchorPtr data_anchor_ptr = topk_anchor_ptr0->GetPeerOutAnchor();
  FUSION_PASS_CHECK(data_anchor_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The data_anchor_ptr is null, fusion failed."), return PARAM_INVALID);
  NodePtr data_node = data_anchor_ptr->GetOwnerNode();
  auto data_node_desc = data_node->GetOpDesc();
  FUSION_PASS_CHECK(data_node_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The data_node_desc is null, fusion failed."), return PARAM_INVALID);
  GeTensorDesc topk_data_tensor = data_node_desc->GetOutputDesc(0);
  GeShape topk_data_shape = topk_data_tensor.GetShape();
  vector<int64_t> dim_info = topk_data_shape.GetDims();
  FUSION_PASS_CHECK(dim_info.size() < 1, OP_LOGW(kFusedOpType.c_str(), "The dim_info size error."), return NOT_CHANGED);
  // 4096 indicates the length of index in assist matrix.
  constexpr int64_t kAssistLen{4096};

  OutDataAnchorPtr topk_anchor_out_ptr0 = topk_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(topk_anchor_out_ptr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The topk_anchor_out_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr data_node_out = topk_anchor_out_ptr0->GetOwnerNode();
  FUSION_PASS_CHECK(data_node_out == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The data_node_out is null, fusion failed."), return PARAM_INVALID);
  auto topk_data_out_tensor_desc = data_node_out->GetOpDesc();
  FUSION_PASS_CHECK(topk_data_out_tensor_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The topk_data_out_tensor_desc is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc topk_data_out_tensor = topk_data_out_tensor_desc->GetOutputDesc(0);
  GeShape topk_data_out_shape = topk_data_out_tensor.GetShape();
  vector<int64_t> dim_info_out = topk_data_out_shape.GetDims();
  FUSION_PASS_CHECK(dim_info_out.size() == 0,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The dim_info_out size is 0, fusion failed."), return PARAM_INVALID);

  // get attr k_num
  Operator op = OpDescUtils::CreateOperatorFromNode(topk_node);
  int64_t const_data_val = 0;
  Tensor const_tensor;
  bool is_topk_v2 = true;
  if (op.GetInputConstData("k", const_tensor) == GRAPH_SUCCESS) {
    // top_k_v2 use k = 0
    is_topk_v2 = false;
    auto k_tensor_desc = op.GetInputDescByName("k");
    DataType input_k_dtype = k_tensor_desc.GetDataType();
    uint8_t* const_data_ptr = const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGW(kFusedOpType.c_str(), "Get k const data failed."),
                      return NOT_CHANGED);
    if (input_k_dtype == DT_INT32) {
      const_data_val = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(const_data_ptr)));
    } else if (input_k_dtype == DT_INT64) {
      const_data_val = *(reinterpret_cast<int64_t*>(const_data_ptr));
    } else {
      OP_LOGW(kFusedOpType.c_str(), "K only support int32 and int64 in AICORE");
      return NOT_CHANGED;
    }
  }

  // topk -> segment_sort + multi_merge
  SegmentCalcParams calcParams;
  calcParams.k_num = const_data_val;
  if (!is_topk_v2 && CheckMultiCoreSegment(topk_node, calcParams)) {
    return AddSegmentSortAndMergeNode(graph, topk_node, calcParams, fusion_nodes);
  }

  vector<PassAttrInfo> topk_attr_info;
  PassAttrInfo k_attr = {1, "k", "SetInt"};
  topk_attr_info.push_back(k_attr);
  string node_name = topk_node->GetName();

  OpDescPtr fusion_desc_ptr = AttrUtils::CloneOpDesc(topk_desc);
  FUSION_PASS_CHECK(fusion_desc_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusion_desc_ptr is null, fusion failed."), return PARAM_INVALID);
  fusion_desc_ptr->SetType("TopKD");
  vector<int> attr_index_vec;
  for (size_t i = 0; i < topk_attr_info.size(); i++) {
    attr_index_vec.push_back(topk_attr_info[i].attrIndex);
  }
  sort(attr_index_vec.begin(), attr_index_vec.end());

  // remove the inputdesc which need to be removed
  for (int i = attr_index_vec.size() - 1; i >= 0; i--) {
    unsigned int index = attr_index_vec[i];
    if (index >= fusion_desc_ptr->GetInputsSize()) {
      OP_LOGI(kFusedOpType.c_str(), "Index[%u] is beyond the size[%u] of input desc", index,
              fusion_desc_ptr->GetInputsSize());
      continue;
    }
    if (!OpDescUtils::ClearInputDesc(fusion_desc_ptr, index)) {
      OP_LOGI(kFusedOpType.c_str(), "Fail to clear input desc[%u]", index);
    }
  }

  FUSION_PASS_CHECK(!AttrUtils::SetInt(fusion_desc_ptr, "k", const_data_val),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Set attr k failed"), return FAILED);
  vector<int64_t> dims = {1};
  GeShape input1_shape(dims);
  GeTensorDesc in_desc1(input1_shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  FUSION_PASS_CHECK(fusion_desc_ptr->AddInputDesc("assic_seq", in_desc1) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "AddInputDesc failed"), return FAILED);
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc_ptr), OP_LOGW(kFusedOpType.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  Status ret = SUCCESS;
  NodePtr fusion_node = topk_node;
  if (!is_topk_v2) {
    ret = PatternFusionUtil::ConstToAttrWithNode(graph, topk_node, "TopKD", topk_attr_info, fusion_node);
  }
  fusion_nodes.push_back(fusion_node);

  FUSION_PASS_CHECK(topk_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "FusionNode is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorPtr assit_ptr{nullptr};
  unique_ptr<uint16_t[]> inputAssit(new (nothrow) uint16_t[kAssistLen * 2]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "InputAssit is NULL"), return FAILED);
  ret = AssitHelp(kAssistLen, inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(kFusedOpType.c_str(), "AssitHelp failed."), return NOT_CHANGED);

  // define shape
  vector<int64_t> assit_dim_info;
  assit_dim_info.push_back(kAssistLen * 2);
  GeShape assit_shape(assit_dim_info);
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetShape(assit_shape);
  tensor_desc.SetFormat(FORMAT_ND);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  FUSION_PASS_MAKE_SHARED((assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                             kAssistLen * 2 * sizeof(uint16_t))),
                          assit_ptr = nullptr;
                          return PARAM_INVALID);

  vector<GeTensorPtr> weights = {assit_ptr};
  FUSION_PASS_CHECK(OpDescUtils::SetWeights(fusion_node, weights) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetWeights failed"), return FAILED);
  auto const_input_nodes = OpDescUtils::GetConstInputs(fusion_node);
  FUSION_PASS_CHECK(const_input_nodes.size() <= 0, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "GetConstInputs Error"),
                    return PARAM_INVALID);
  NodePtr const_input = const_input_nodes[0];
  FUSION_PASS_CHECK(const_input == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The const_input is null, fusion failed."),
                    return PARAM_INVALID);
  auto const_input_desc = const_input->GetOpDesc();
  FUSION_PASS_CHECK(const_input_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The const_input_desc is null, fusion failed."),
                    return PARAM_INVALID);
  const_input_desc->SetType(kConstantOp);
  if (is_topk_v2) {
    topk_desc->SetType("TopKV2D");
  } else {
    topk_desc->SetType("TopKD");
  }

  OpDescPtr topkd_desc = fusion_node->GetOpDesc();
  int64_t dim_size = dim_info.size();
  int64_t dim_aim;
  if (!AttrUtils::GetInt(topkd_desc, "dim", dim_aim)) {
    OP_LOGI(kFusedOpType.c_str(), "Cannot get attr dim, fusion success, no need do more");
    return SUCCESS;
  }
  if (dim_aim < 0) {
    dim_aim = dim_size + dim_aim;
  }
  if (dim_aim == dim_size - 1) {
    return SUCCESS;
  }

  NodePtr trans_input_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, true, 0, fusion_nodes);
  OpDescPtr trans_input_desc = trans_input_node->GetOpDesc();
  FUSION_PASS_CHECK(trans_input_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The trans_input_desc is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc trans_data_tensor = trans_input_desc->GetInputDesc(0);
  GeShape trans_data_shape = trans_data_tensor.GetShape();
  vector<int64_t> trans_dim_info = trans_data_shape.GetDims();
  int64_t trans_dim_info_size = trans_dim_info.size();
  FUSION_PASS_CHECK(dim_aim >= trans_dim_info_size, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dim index is out of shape range."),
                    return PARAM_INVALID);
  swap(trans_dim_info[dim_aim], trans_dim_info[dim_size - 1]);

  // get input_transpose perm
  vector<int64_t> perm;
  ret = PermVecGen(dim_size, dim_aim, perm);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "PermVecGen failed."), return ret);

  // set input_transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_input_desc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Input transporse set perm failed"), return FAILED);
  // set input_transpose output shape range
  vector<pair<int64_t, int64_t>> shape_range_after_sorted;
  if (trans_data_tensor.GetShapeRange(shape_range_after_sorted) != GRAPH_SUCCESS) {
    OP_LOGD(kFusedOpType.c_str(), "GetShapeRange failed. However the process is fine.");
  }
  if (shape_range_after_sorted.size() > 0) {
    int64_t tmp = shape_range_after_sorted[dim_aim].second;
    shape_range_after_sorted[dim_aim].second = shape_range_after_sorted[dim_size - 1].second;
    shape_range_after_sorted[dim_size - 1].second = tmp;
  }
  GeTensorDesc out_trans_data_tensor = trans_input_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(out_trans_data_tensor.SetShapeRange(shape_range_after_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);
  // set input_transpose output shape
  GeShape transpose_assit_shape(trans_dim_info);
  auto transin_mutable_output0 = trans_input_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(transin_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The transin_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  transin_mutable_output0->SetShape(GeShape(transpose_assit_shape));
  transin_mutable_output0->SetOriginShape(GeShape(transpose_assit_shape));

  // set topk dim and input desc
  FUSION_PASS_CHECK(!AttrUtils::SetInt(topkd_desc, "dim", -1), VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Set attr dim failed"),
                    return FAILED);
  auto fusion_mutable_input0 = topkd_desc->MutableInputDesc(0);
  FUSION_PASS_CHECK(fusion_mutable_input0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusion_mutable_input0 is null, fusion failed."),
                    return PARAM_INVALID);

  fusion_mutable_input0->SetShape(GeShape(transpose_assit_shape));
  fusion_mutable_input0->SetOriginShape(GeShape(transpose_assit_shape));
  // set topkd input shape range

  GeTensorDesc topkd_input_data_tensor = topkd_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(topkd_input_data_tensor.SetShapeRange(shape_range_after_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  // set topkd output desc according to transpose
  vector<int64_t> topk_out_shape;
  GeTensorDesc topkd_data_tensor = topkd_desc->GetOutputDesc(0);
  GeShape topkd_data_shape = topkd_data_tensor.GetShape();
  topk_out_shape = topkd_data_shape.GetDims();
  vector<int64_t> topkd_dim_info = topk_out_shape;
  swap(topkd_dim_info[dim_aim], topkd_dim_info[dim_size - 1]);

  // set topkd val output shape
  GeShape topk_out_ge_shape(topkd_dim_info);
  auto fusion_mutable_output0 = topkd_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(fusion_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusion_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  fusion_mutable_output0->SetShape(GeShape(topk_out_ge_shape));
  fusion_mutable_output0->SetOriginShape(GeShape(topk_out_ge_shape));
  // set topkd val output shape range
  vector<pair<int64_t, int64_t>> shape_range_val_k;
  shape_range_val_k = shape_range_after_sorted;
  if (shape_range_val_k.size() > 0) {
    shape_range_val_k[shape_range_val_k.size() - 1].second = const_data_val;
  }
  GeTensorDesc topkd_data_out_tensor = topkd_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(topkd_data_out_tensor.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  // set topkd index output shape
  auto fusion_mutable_output1 = topkd_desc->MutableOutputDesc(1);
  FUSION_PASS_CHECK(fusion_mutable_output1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusion_mutable_output1 is null, fusion failed."),
                    return PARAM_INVALID);
  fusion_mutable_output1->SetShape(GeShape(topk_out_ge_shape));
  fusion_mutable_output1->SetOriginShape(GeShape(topk_out_ge_shape));
  // set topkd index output shape range
  GeTensorDesc topkd_data_out_index_tensor = topkd_desc->GetOutputDesc(1);
  FUSION_PASS_CHECK(topkd_data_out_index_tensor.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  NodePtr trans_output_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, false, 0, fusion_nodes);
  FUSION_PASS_CHECK(trans_output_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The trans_output_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr trans_output_desc = trans_output_node->GetOpDesc();

  // set val transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_output_desc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Output val transporse set perm failed"), return FAILED);

  // set val transepose output shape
  GeShape out_transpose_output_assit_shape(topk_out_shape);
  auto transout_mutable_output0 = trans_output_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(transout_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The transout_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  transout_mutable_output0->SetShape(GeShape(out_transpose_output_assit_shape));
  transout_mutable_output0->SetOriginShape(GeShape(out_transpose_output_assit_shape));
  // set val transepose output shape input_shape_range
  GeTensorDesc trans_output_tensor_input = trans_output_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(trans_output_tensor_input.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);
  // set val transepose output shape output_shape_range
  vector<pair<int64_t, int64_t>> shape_range_val_k_sorted;
  shape_range_val_k_sorted = shape_range_val_k;

  if (shape_range_val_k_sorted.size() > 0) {
    int64_t tmp = shape_range_val_k_sorted[dim_aim].second;
    shape_range_val_k_sorted[dim_aim].second = shape_range_val_k_sorted[shape_range_val_k_sorted.size() - 1].second;
    shape_range_val_k_sorted[shape_range_val_k_sorted.size() - 1].second = tmp;
  }
  GeTensorDesc trans_output_tensor_output = trans_output_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(trans_output_tensor_output.SetShapeRange(shape_range_val_k_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  NodePtr trans_output_index_node =
      PatternFusionUtil::InsertSingleNode(graph, fusion_node, kPatternTranspose, false, 1, fusion_nodes);
  FUSION_PASS_CHECK(trans_output_index_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The trans_output_index_node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr trans_output_index_desc = trans_output_index_node->GetOpDesc();

  // set index transpose perm
  FUSION_PASS_CHECK(!AttrUtils::SetListInt(trans_output_index_desc, "perm", perm),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Output index transporse set perm failed"), return FAILED);
  // set index transepose output shape
  GeShape out_index_transpose_output_assit_shape(topk_out_shape);
  auto trans_index_mutable_output0 = trans_output_index_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(trans_index_mutable_output0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The trans_index_mutable_output0 is null, fusion failed."),
                    return PARAM_INVALID);
  trans_index_mutable_output0->SetShape(GeShape(out_index_transpose_output_assit_shape));
  trans_index_mutable_output0->SetOriginShape(GeShape(out_index_transpose_output_assit_shape));
  // set index transepose output shape input_shape_range
  GeTensorDesc trans_output_index_tensor_input = trans_output_index_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(trans_output_index_tensor_input.SetShapeRange(shape_range_val_k) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);
  // set index transepose output shape output_shape_range
  GeTensorDesc trans_output_index_tensor_output = trans_output_index_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(trans_output_index_tensor_output.SetShapeRange(shape_range_val_k_sorted) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetShapeRange failed"), return FAILED);

  return SUCCESS;
}

REGISTER_PASS("TopKFusionPass", BUILT_IN_GRAPH_PASS, TopKFusionPass);
}  // namespace fe
