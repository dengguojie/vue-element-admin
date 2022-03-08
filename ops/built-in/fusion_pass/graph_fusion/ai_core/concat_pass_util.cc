/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file concat_pass_util.cpp
 *
 * @brief util for concat pass
 *
 * @version 1.0
 *
 */
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"
#include "concat_pass_util.h"

using namespace std;
namespace fe {
int64_t GetMaxInputsNum(const ge::NodePtr& fused_node) {
  int64_t max_inputs = 63;

  // due to compile time cost
  if (HasUnKnowShape(fused_node)) {
    max_inputs = 48;

    PlatformInfo platform_info;
    OptionalInfo optional_info;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
      OP_LOGW("Concat", "Fail to get platform info.");
      optional_info.soc_version == "";
    }
    OP_LOGD("Concat", "Get soc_version is: [%s].", optional_info.soc_version.c_str());
    if (optional_info.soc_version == "Ascend310") {
      max_inputs = 21;
    }
  }

  return max_inputs;
}

bool CheckNeedChanged(const ge::OpDescPtr& fused_desc, const int64_t max_inputs) {
  int64_t inputs_num = fused_desc->GetInputsSize();
  if (inputs_num > max_inputs) {
    return true;
  }

  return false;
}

Status RemoveInvalidEdge(ge::NodePtr& fused_node, ge::OpDescPtr& fused_desc, const string op_type) {
  int64_t num_n;
  int64_t num_n_del = 0;
  int64_t num_n_new;
  vector<int64_t> bad_edges;

  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(fused_desc, "N", num_n),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to get attribte N."),
                    return FAILED);
  for (int i = 0; i < num_n; i++) {
    int64_t repeat_num = 0;
    ge::GeTensorDesc select_input_desc = fused_node->GetOpDesc()->GetInputDesc(i);
    vector<int64_t> select_input_shape = select_input_desc.GetShape().GetDims();
    for (size_t j = 0; j < select_input_shape.size(); j++) {
      if (select_input_shape[j] == 0) {
        repeat_num += 1;
      }
    }
    if (repeat_num > 0) {
      num_n_del += 1;
      bad_edges.push_back(i);
    }
  }

  if (!bad_edges.empty()) {
    for (size_t i = 0; i < bad_edges.size(); i++) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(bad_edges[i] - i)->GetPeerOutAnchor(),
                                                   fused_node->GetInDataAnchor(bad_edges[i] - i)) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Remove edge failed."), return FAILED);

      RemoveInputDesc(fused_desc, bad_edges[i] - i);
      ge::NodeUtils::ClearInDataAnchor(fused_node, fused_node->GetInDataAnchor(bad_edges[i] - i));
    }
  }

  num_n_new = num_n - num_n_del;
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(fused_desc, "N", num_n_new),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to set attribte N."),
                    return FAILED);

  return SUCCESS;
}

void UpdateInputName(const ge::OpDescPtr& input_desc_ptr) {
  auto input_count = input_desc_ptr->GetAllInputsSize();
  map<string, uint32_t> name_index_map;
  string name_val = "x";

  for (size_t idx = 0; idx < input_count; ++idx) {
    name_index_map.insert({name_val + std::to_string(idx), idx});
  }

  input_desc_ptr->UpdateInputName(name_index_map);
}

Status UnlinkUselessNodes(ge::ComputeGraph& graph, const vector<ge::NodePtr>& base_node_vec, const string& op_type) {
  if (!base_node_vec.empty()) {
    for (auto base_node : base_node_vec) {
      for (auto in_anchor : base_node->GetAllInDataAnchors()) {
        if (in_anchor != nullptr) {
          in_anchor->UnlinkAll();
        }
      }
      for (auto out_anchor : base_node->GetAllOutDataAnchors()) {
        if (out_anchor != nullptr) {
          out_anchor->UnlinkAll();
        }
      }

      FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(base_node),
      VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Remove Node [%s] failed", base_node->GetName().c_str()),
      return FAILED);
    }
  }

  return SUCCESS;
}

Status CreateConcatBaseNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                            ge::OpDescPtr& concatd_base_desc, ge::NodePtr& concatd_base_node,
                            const OriNodeInfo& ori_node_info) {
  auto op_type = ori_node_info.op_type;
  auto split_rank = ori_node_info.split_rank;
  auto nums_input = ori_node_info.nums_input;

  concatd_base_desc->SetName(concatd_base_desc->GetName() + "/" + op_type + "Base_node" + to_string(split_rank));
  if (op_type == "Pack") {
    // used for create child pack node
    auto tmp_fused_op = ge::OperatorFactory::CreateOperator("tmp_fused_op", "ConcatD");
    if (tmp_fused_op.IsEmpty()) {
      OP_LOGE(op_type, "create fusion node ConcatD failed.");
      return FAILED;
    }
    auto tmp_fused_op_desc = ge::OpDescUtils::GetOpDescFromOperator(tmp_fused_op);
    tmp_fused_op.BreakConnect();
    concatd_base_desc->AddInferFunc(tmp_fused_op_desc->GetInferFunc());

    concatd_base_desc->SetType("ConcatD");
    // op concat has no attribute axis
    if (SUCCESS == concatd_base_desc->DelAttr("axis")) {
      OP_LOGD(op_type, "delete attribute axis from pack for concat done.");
    }
  } else {
    concatd_base_desc->SetType(op_type);
  }

  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concatd_base_desc, "concat_dim", ori_node_info.concat_dim),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to set attribte concat_dim."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concatd_base_desc, "N", nums_input),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to set attribte N."),
                    return FAILED);

  for (int64_t idx = ori_node_info.base_inputs_num - 1; idx >= nums_input; idx--) {
    RemoveInputDesc(concatd_base_desc, idx);
  }

  concatd_base_node = graph.AddNode(concatd_base_desc);
  FUSION_PASS_CHECK(
      concatd_base_node == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(op_type, "concatd_base_node is null, fusion failed."),
      return PARAM_INVALID);

  // for big case, the base node will be deleted
  if (nums_input <= ori_node_info.max_inputs) {
    new_nodes.push_back(concatd_base_node);
  }

  // the output of child concatd nodes will connect to the base concatd node
  auto ori_base_out_anchor = ori_node_info.pre_base_node->GetOutDataAnchor(0);
  auto base_out_anchor = concatd_base_node->GetOutDataAnchor(0);
  for (InDataAnchorPtr in_anchor_prt : ori_base_out_anchor->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(ori_base_out_anchor, in_anchor_prt),
                      VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(base_out_anchor, in_anchor_prt),
                      VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Add out data edge failed."),
                      return FAILED);
  }

  return SUCCESS;
}

Status CreateConcatChildNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                             ge::OpDescPtr& concatd_base_desc, const ge::NodePtr& concatd_base_node,
                             const OriNodeInfo& ori_node_info) {
  auto pre_base_node = ori_node_info.pre_base_node;
  auto split_rank = ori_node_info.split_rank;
  auto node_idx = ori_node_info.node_idx;
  auto concat_dim = ori_node_info.concat_dim;
  auto max_inputs = ori_node_info.max_inputs;
  auto nums_input = ori_node_info.nums_input;
  auto base_inputs_num = ori_node_info.base_inputs_num;
  auto op_type = ori_node_info.op_type;

  ge::OpDescPtr concatd_desc = AttrUtils::CopyOpDesc(pre_base_node->GetOpDesc());

  // only top rank can be pack
  if (op_type == "Pack" && split_rank > 0) {
    concatd_desc->SetName(concatd_desc->GetName() + "/ConcatD" + to_string(node_idx) + to_string(split_rank));
    concatd_desc->SetType("ConcatD");
  } else {
    concatd_desc->SetName(concatd_desc->GetName() + "/" + op_type + to_string(node_idx) + to_string(split_rank));
    concatd_desc->SetType(op_type);
  }

  if (op_type == "Pack" && split_rank == 0) {
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concatd_desc, "axis", concat_dim),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to set attribte axis."),
                    return FAILED);
    // op pack has no attribute concat_dim
    if (SUCCESS == concatd_desc->DelAttr("concat_dim")) {
      OP_LOGD(op_type, "delete attribute concat_dim from concat for pack done.");
    }
  } else {
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concatd_desc, "concat_dim", concat_dim),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to set attribte concat_dim."),
                    return FAILED);
  }
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concatd_desc, "N", nums_input),
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "Failed to set attribte N."),
                    return FAILED);

  // the beginning input index will be reset to zero after delete
  if (node_idx == 0) {
    for (int64_t idx = base_inputs_num - 1; idx >= max_inputs; idx--) {
      RemoveInputDesc(concatd_desc, idx);
    }
  } else {
    for (int64_t idx_1 = node_idx * max_inputs - 1; idx_1 >= 0; idx_1--) {
      RemoveInputDesc(concatd_desc, idx_1);
    }

    // will not run this line for last node
    int64_t left_num = base_inputs_num - (max_inputs * node_idx + 1);
    if (left_num >= max_inputs) {
      for (int64_t idx_2 = left_num; idx_2 >= max_inputs; idx_2--) {
        RemoveInputDesc(concatd_desc, idx_2);
      }
    }
  }

  ge::NodePtr concatd_node = graph.AddNode(concatd_desc);
  FUSION_PASS_CHECK(concatd_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "concatd_node is null, fusion failed."),
                    return PARAM_INVALID);

  new_nodes.push_back(concatd_node);

  // infershape begin
  UpdateInputName(concatd_desc);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(concatd_node);
  auto infer_shape_ret = op.InferShapeAndType();
  OP_LOGE_IF(infer_shape_ret != GRAPH_SUCCESS, FAILED, op_type, "InferShapeAndType failed.");
  concatd_base_desc->UpdateInputDesc(node_idx, concatd_desc->GetOutputDesc(0));
  // infershape end

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_node->GetOutDataAnchor(0),
                                                       concatd_base_node->GetInDataAnchor(node_idx)),
                    VECTOR_FUSION_INNER_ERR_REPORT(
                        op_type,
                        "Add edge from fused node:%s's index[%ld] to fusion node:%s's index[%ld] failed.",
                        concatd_base_node->GetName().c_str(), node_idx, concatd_node->GetName().c_str(), node_idx),
                    return FAILED);

  auto cur_idx = node_idx * max_inputs;
  for (int64_t idx_3 = 0; idx_3 < nums_input; idx_3++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(pre_base_node->GetInDataAnchor(idx_3 + cur_idx)->GetPeerOutAnchor(),
                                           concatd_node->GetInDataAnchor(idx_3)),
        VECTOR_FUSION_INNER_ERR_REPORT(
            op_type,
            "Add edge from fused node:%s's index[%ld] to fusion node:%s's index[%ld] failed.",
            pre_base_node->GetName().c_str(), (idx_3 + cur_idx), concatd_node->GetName().c_str(), idx_3),
        return FAILED);
  }

  return SUCCESS;
}

Status SplitConcatNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                       ge::NodePtr& fused_node, const int64_t max_inputs, const string op_type) {
  auto fused_desc = fused_node->GetOpDesc();
  ge::OpDescPtr concatd_base_desc = AttrUtils::CopyOpDesc(fused_desc);
  ge::NodePtr concatd_base_node = nullptr;
  int64_t nodes_num = 0;
  int64_t last_node_inputs_num = 0;
  bool is_need_split = true;

  int64_t concat_dim = 0;
  if (op_type == "Pack") {
    if (!ge::AttrUtils::GetInt(fused_desc, "axis", concat_dim)) {
      OP_LOGW(op_type, "get attribute axis failed, use default value!");
    }
  } else {
    FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(fused_desc, "concat_dim", concat_dim),
                      VECTOR_FUSION_INNER_ERR_REPORT(op_type, "get attribute concat_dim failed!"),
                      return FAILED);
  }

  // save nodes which will be deleted at the end
  vector<ge::NodePtr> base_node_vec;
  base_node_vec.push_back(fused_node);

  OriNodeInfo ori_node_info;
  ori_node_info.concat_dim = concat_dim;
  ori_node_info.max_inputs = max_inputs;
  ori_node_info.split_rank = 0;
  ori_node_info.op_type = op_type;

  while (is_need_split) {
    ori_node_info.pre_base_node = base_node_vec[ori_node_info.split_rank];
    ori_node_info.base_inputs_num = concatd_base_desc->GetInputsSize();
    nodes_num = (ori_node_info.base_inputs_num + max_inputs - 1) / max_inputs;
    last_node_inputs_num = ori_node_info.base_inputs_num % max_inputs;
    if (last_node_inputs_num == 0) {
      last_node_inputs_num = max_inputs;
    }

    ori_node_info.nums_input = nodes_num;
    FUSION_PASS_CHECK(CreateConcatBaseNode(graph, new_nodes, concatd_base_desc,
                                           concatd_base_node, ori_node_info) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(op_type, "create new base concatd node failed."),
                      return FAILED);

    ori_node_info.nums_input = max_inputs;
    for (int64_t node_idx = 0; node_idx < nodes_num; node_idx++) {
      if (node_idx == nodes_num - 1) {
        ori_node_info.nums_input = last_node_inputs_num;
      }
      ori_node_info.node_idx = node_idx;
      FUSION_PASS_CHECK(CreateConcatChildNode(graph, new_nodes, concatd_base_desc,
                                              concatd_base_node, ori_node_info) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(op_type, "create new child concatd node failed."),
                          return FAILED);
    }
    UpdateInputName(concatd_base_desc);

    if (nodes_num > max_inputs) {
      base_node_vec.push_back(concatd_base_node);
      concatd_base_desc = AttrUtils::CopyOpDesc(concatd_base_desc);
      ori_node_info.split_rank++;
    } else {
      is_need_split = false;
    }
  }

  // remove unlinked nodes from graph
  FUSION_PASS_CHECK(UnlinkUselessNodes(graph, base_node_vec, op_type) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(op_type, "delete unlinked concat nodes failed."),
                    return FAILED);

  return SUCCESS;
}
}  // namespace fe
