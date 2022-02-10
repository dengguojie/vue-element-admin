/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicGRUV2 fusion pass(DynamicGRUV2 --> BatchMatmulV2 & DynamicGRUV2Hidden)
 *
 */

#include "dynamic_gru_v2_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>
#include "external/graph/operator_factory.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const int BLOCKSIZE = 16;
static const char* GRUV2_NODE = "DynamicGRUV2";
static const std::string PATTERN_GRUV2_NODE = "DynamicGRUV2";
static const int GM_IO_SPEED = 32;
static const int L2_IO_SPEED = 96;
static const int INDEX_2 = 2;
static const int INDEX_3 = 3;
static const int INDEX_4 = 4;
static const int INDEX_5 = 5;
static const int INDEX_6 = 6;
static const int CONST_NUM_2 = 2;
static const int HIDDEN_NUM = 3;
static const int CACHE_NUM = 4;
static const int ALIGN_16 = 16;
static const float THREE_QUARTERS = 0.75;

namespace gruv2 {
struct GruCalcParam {
  int t_size;
  float m_size;
  float x_size;
  float h_size;
  uint32_t core_num;
};

using CalcOp = float (*)(int, int, GruCalcParam&);

float FindBestCalc(CalcOp func, GruCalcParam& param) {
  float best = -1.0;
  for (int core_num = 1; static_cast<uint32_t>(core_num) <= param.core_num; core_num++) {
    int m_cut = 1;
    while (m_cut <= core_num) {
      if (core_num % m_cut != 0) {
        m_cut++;
        continue;
      }
      int n_cut = core_num / m_cut;
      auto ret = func(m_cut, n_cut, param);
      if (best == -1.0 || ret < best) {
        best = ret;
      }
      m_cut++;
    }
  }
  return best;
}


float CalcSolution1Op(int m_cut, int n_cut, GruCalcParam& param) {
  if (m_cut == 0 || n_cut == 0) {
    VECTOR_FUSION_INNER_ERR_REPORT("DynamicGRUV2FusionPass", "divied by zero error.");
    return 0.0;
  }
  float op_m = ceil(ceil(param.m_size / ALIGN_16) / m_cut) * ALIGN_16;
  float op_k = param.x_size + param.h_size;
  float op_h = ceil(ceil(param.h_size / ALIGN_16) / n_cut) * ALIGN_16;
  float op_n = HIDDEN_NUM * op_h;

  // t1
  float left_matrix_in_cost = op_k * op_m * CONST_NUM_2 / GM_IO_SPEED;
  float right_matrix_in_cost = op_k * op_n * CONST_NUM_2 / GM_IO_SPEED;
  float op_t1_total = left_matrix_in_cost + right_matrix_in_cost;

  // t2
  float left_x_matrix_in_cost = op_m * param.x_size * CONST_NUM_2 / GM_IO_SPEED;
  float left_h_matrix_in_cost = op_m * param.h_size * CONST_NUM_2 / L2_IO_SPEED;
  // no right matrix
  float op_t2_total = left_x_matrix_in_cost + left_h_matrix_in_cost;

  return op_t1_total + op_t2_total * (param.t_size - 1);
}


// solution1.2, weight reused
float CalcSolution1(GruCalcParam& param) {
  return FindBestCalc(CalcSolution1Op, param);
}


float CalcSolution2Op1(int m_cut, int n_cut, GruCalcParam& param) {
  if (m_cut == 0 || n_cut == 0) {
    VECTOR_FUSION_INNER_ERR_REPORT("DynamicGRUV2FusionPass", "divied by zero error.");
    return 0.0;
  }
  float op1_m = ceil(param.t_size * ceil(param.m_size / ALIGN_16) / m_cut) * ALIGN_16;
  float op1_k = param.x_size;
  float op1_n = ceil(HIDDEN_NUM * ceil(param.h_size / ALIGN_16) / n_cut) * ALIGN_16;

  float left_matrix_in_cost = op1_m * op1_k * CONST_NUM_2 / GM_IO_SPEED;
  float right_matrix_in_cost = op1_k * op1_n * CONST_NUM_2 / GM_IO_SPEED;

  return left_matrix_in_cost + right_matrix_in_cost;
}


float CalcSolution2Op2(int m_cut, int n_cut, GruCalcParam& param) {
  if (m_cut == 0 || n_cut == 0) {
    VECTOR_FUSION_INNER_ERR_REPORT("DynamicGRUV2FusionPass", "divied by zero error.");
    return 0.0;
  }
  float op2_m = ceil(ceil(param.m_size / ALIGN_16) / m_cut) * ALIGN_16;
  float op2_k = param.h_size;
  float op2_h = ceil(ceil(param.h_size / ALIGN_16) / n_cut) * ALIGN_16;
  float op2_n = HIDDEN_NUM * op2_h;

  // t1
  float left_matrix_in_cost = op2_m * op2_k * CONST_NUM_2 / GM_IO_SPEED;
  float right_matrix_in_cost = op2_k * op2_n * CONST_NUM_2 / GM_IO_SPEED;
  float cache_in_cost = op2_m * op2_n * CACHE_NUM / L2_IO_SPEED;
  float op2_t1_total = left_matrix_in_cost + right_matrix_in_cost + cache_in_cost;

  // t2
  // last h
  left_matrix_in_cost = op2_m * op2_k * CONST_NUM_2 / L2_IO_SPEED;
  // no right matrix
  cache_in_cost = op2_m * op2_n * CACHE_NUM / L2_IO_SPEED;
  float op2_t2_total = left_matrix_in_cost + cache_in_cost;

  return op2_t1_total + op2_t2_total * (param.t_size - 1);
}


// solution2, 2 ops, weight reused in op2
float CalcSolution2(GruCalcParam& param) {
  return FindBestCalc(CalcSolution2Op1, param) + FindBestCalc(CalcSolution2Op2, param);
}
}  // namespace gruv2

vector<FusionPattern*> DynamicGRUV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicGRUV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_GRUV2_NODE, {GRUV2_NODE}).SetOutput(PATTERN_GRUV2_NODE);
  patterns.push_back(pattern);
  return patterns;
}

void DynamicGRUV2FusionPass::SetAttr(ge::OpDescPtr gru_desc, ge::OpDescPtr gru_split_desc) const {
  std::string direction("UNIDIRECTIONAL");
  int cell_depth = 1;
  float keep_prob = 1.0;
  float cell_clip = -1.0;
  int num_proj = 0;
  bool time_major = true;
  std::string activation("tanh");
  std::string gate_order("zrh");
  bool reset_after = true;
  bool is_training = true;
  if (ge::AttrUtils::GetStr(gru_desc, "direction", direction)) {
    ge::AttrUtils::SetStr(gru_split_desc, "direction", direction);
  }
  if (ge::AttrUtils::GetInt(gru_desc, "cell_depth", cell_depth)) {
    ge::AttrUtils::SetInt(gru_split_desc, "cell_depth", cell_depth);
  }
  if (ge::AttrUtils::GetFloat(gru_desc, "keep_prob", keep_prob)) {
    ge::AttrUtils::SetFloat(gru_split_desc, "keep_prob", keep_prob);
  }
  if (ge::AttrUtils::GetFloat(gru_desc, "cell_clip", cell_clip)) {
    ge::AttrUtils::SetFloat(gru_split_desc, "cell_clip", cell_clip);
  }
  if (ge::AttrUtils::GetInt(gru_desc, "num_proj", num_proj)) {
    ge::AttrUtils::SetInt(gru_split_desc, "num_proj", num_proj);
  }
  if (ge::AttrUtils::GetBool(gru_desc, "time_major", time_major)) {
    ge::AttrUtils::SetBool(gru_split_desc, "time_major", time_major);
  }
  if (ge::AttrUtils::GetStr(gru_desc, "activation", activation)) {
    ge::AttrUtils::SetStr(gru_split_desc, "activation", activation);
  }
  if (ge::AttrUtils::GetStr(gru_desc, "gate_order", gate_order)) {
    ge::AttrUtils::SetStr(gru_split_desc, "gate_order", gate_order);
  }
  if (ge::AttrUtils::GetBool(gru_desc, "reset_after", reset_after)) {
    ge::AttrUtils::SetBool(gru_split_desc, "reset_after", reset_after);
  }
  if (ge::AttrUtils::GetBool(gru_desc, "is_training", is_training)) {
    ge::AttrUtils::SetBool(gru_split_desc, "is_training", is_training);
  }
}

ge::NodePtr DynamicGRUV2FusionPass::AddSplitNode(ge::NodePtr gru_node, ge::NodePtr matmul_node,
                                                 ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes) {
  std::string gru_op_name = gru_node->GetName() + "/DynamicGRUV2Hidden";
  auto split_op = ge::OperatorFactory::CreateOperator(gru_op_name.c_str(),
                                                      "DynamicGRUV2Hidden");
  FUSION_PASS_CHECK(split_op.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create DynamicGRUV2Hidden operator error"),
                    return nullptr);
  auto gru_split_desc = ge::OpDescUtils::GetOpDescFromOperator(split_op);
  split_op.BreakConnect();

  // Input
  ge::OpDescPtr gru_desc = gru_node->GetOpDesc();
  ge::GeTensorDesc x_weight_input_desc = matmul_node->GetOpDesc()->GetOutputDesc(0).Clone();
  gru_split_desc->UpdateInputDesc("x_weight_input", x_weight_input_desc);
  gru_split_desc->UpdateInputDesc("weight_hidden", gru_desc->GetInputDesc(INDEX_2));
  bool has_bias = gru_desc->MutableInputDesc("bias_hidden") != nullptr;
  if (has_bias) {
    gru_split_desc->UpdateInputDesc("bias_hidden", *gru_desc->MutableInputDesc("bias_hidden"));
  }
  bool has_seq_length = gru_desc->MutableInputDesc("seq_length") != nullptr;
  if (has_seq_length) {
    gru_split_desc->UpdateInputDesc("seq_length", *gru_desc->MutableInputDesc("seq_length"));
  }
  bool has_init_h = gru_desc->MutableInputDesc("init_h") != nullptr;
  if (has_init_h) {
    gru_split_desc->UpdateInputDesc("init_h", *gru_desc->MutableInputDesc("init_h"));
  }

  // output
  gru_split_desc->UpdateOutputDesc("y", gru_desc->GetOutputDesc(0));
  gru_split_desc->UpdateOutputDesc("output_h", gru_desc->GetOutputDesc(1));
  gru_split_desc->UpdateOutputDesc("update", gru_desc->GetOutputDesc(INDEX_2));
  gru_split_desc->UpdateOutputDesc("reset", gru_desc->GetOutputDesc(INDEX_3));
  gru_split_desc->UpdateOutputDesc("new", gru_desc->GetOutputDesc(INDEX_4));
  gru_split_desc->UpdateOutputDesc("hidden_new", gru_desc->GetOutputDesc(INDEX_5));

  SetAttr(gru_desc, gru_split_desc);

  // create node
  ge::NodePtr split_node = graph.AddNode(gru_split_desc);
  FUSION_PASS_CHECK(split_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add split node failed."),
                    return nullptr);
  new_nodes.push_back(split_node);

  // Edge
  // cache
  ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0),
                          split_node->GetInDataAnchor(0));
  // weight_hidden
  ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(INDEX_2)->GetPeerOutAnchor(),
                          split_node->GetInDataAnchor(1));
  if (has_bias) {
    ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(INDEX_4)->GetPeerOutAnchor(),
                            split_node->GetInDataAnchor(INDEX_2));
  }
  if (has_seq_length) {
    ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(INDEX_5)->GetPeerOutAnchor(),
                            split_node->GetInDataAnchor(INDEX_3));
  }
  if (has_init_h) {
    ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(INDEX_6)->GetPeerOutAnchor(),
                            split_node->GetInDataAnchor(INDEX_4));
  }
  // output edges
  for (unsigned int i = 0; i < INDEX_6; i++) {
    if (gru_node->GetOutDataAnchor(i)->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr in_anchor_ptr : gru_node->GetOutDataAnchor(i)->GetPeerInDataAnchors()) {
        in_anchor_ptr->UnlinkAll();
        ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(i), in_anchor_ptr);
      }
    }
  }

  return split_node;
}


ge::NodePtr DynamicGRUV2FusionPass::AddMatmulNode(ge::NodePtr gru_node, ge::ComputeGraph& graph,
                                                  vector<ge::NodePtr>& new_nodes) {
  std::string batchmatmul_op_name = gru_node->GetName() + "/BatchMatMulV2";
  auto matmul_op = ge::OperatorFactory::CreateOperator(batchmatmul_op_name.c_str(), "BatchMatMulV2");
  FUSION_PASS_CHECK(matmul_op.IsEmpty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create matmul operator error"),
                    return nullptr);
  auto matmul_desc = ge::OpDescUtils::GetOpDescFromOperator(matmul_op);
  matmul_op.BreakConnect();

  // Input
  ge::OpDescPtr gru_desc = gru_node->GetOpDesc();
  ge::GeTensorDesc x_desc = gru_desc->GetInputDesc(0).Clone();
  int64_t t_size = x_desc.GetOriginShape().GetDim(0);
  int64_t m_size = x_desc.GetOriginShape().GetDim(1);
  matmul_desc->UpdateInputDesc("x1", x_desc);
  matmul_desc->UpdateInputDesc("x2", gru_desc->GetInputDesc(1));
  bool has_bias = gru_desc->MutableInputDesc("bias_input") != nullptr;
  ge::NodePtr cast_node = nullptr;
  if (has_bias) {
    ge::GeTensorDescPtr bias_desc_ptr = gru_desc->MutableInputDesc("bias_input");
    if (bias_desc_ptr->GetDataType() == ge::DT_FLOAT16) {
      std::string cast_op_name = gru_node->GetName() + "/BatchMatMulV2/Cast";
      Operator cast_op = ge::OperatorFactory::CreateOperator(cast_op_name.c_str(), "Cast");
      FUSION_PASS_CHECK(cast_op.IsEmpty(),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create cast operator error"),
                        return nullptr);
      auto cast_desc = ge::OpDescUtils::GetOpDescFromOperator(cast_op);
      cast_op.BreakConnect();

      cast_desc->UpdateInputDesc("x", bias_desc_ptr->Clone());
      bias_desc_ptr->SetDataType(DT_FLOAT);
      cast_desc->UpdateOutputDesc("y", *bias_desc_ptr);
      ge::AttrUtils::SetInt(cast_desc, "dst_type", 0);

      // create cast node
      cast_node = graph.AddNode(cast_desc);
      FUSION_PASS_CHECK(cast_node == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add cast node failed."),
                        return nullptr);
      new_nodes.push_back(cast_node);
    }
    matmul_desc->UpdateInputDesc("bias", *bias_desc_ptr);
  }

  // output
  int64_t h_size = gru_node->GetOpDesc()->GetInputDesc(INDEX_2).GetOriginShape().GetDim(0);
  vector<int64_t> output_ori_dims;
  output_ori_dims.push_back(t_size);
  output_ori_dims.push_back(m_size);
  output_ori_dims.push_back(HIDDEN_NUM * h_size);
  ge::GeShape output_ori_shape(output_ori_dims);
  vector<int64_t> output_dims;
  output_dims.push_back(t_size);
  output_dims.push_back(HIDDEN_NUM * ((h_size + ALIGN_16 - 1) / ALIGN_16));
  output_dims.push_back((m_size + ALIGN_16 - 1) / ALIGN_16);
  output_dims.push_back(ALIGN_16);
  output_dims.push_back(ALIGN_16);
  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc output_tensor_desc = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  output_tensor_desc.SetOriginShape(output_ori_shape);
  output_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  matmul_desc->UpdateOutputDesc("y", output_tensor_desc);

  // attr
  ge::AttrUtils::SetBool(matmul_desc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmul_desc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmul_node = graph.AddNode(matmul_desc);
  FUSION_PASS_CHECK(matmul_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add matmul node failed."),
                    return nullptr);
  new_nodes.push_back(matmul_node);

  // Edge
  ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(0)->GetPeerOutAnchor(), matmul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(1)->GetPeerOutAnchor(), matmul_node->GetInDataAnchor(1));
  if (has_bias) {
    if (cast_node != nullptr) {
      ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(INDEX_3)->GetPeerOutAnchor(), cast_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(INDEX_2));
    } else {
      ge::GraphUtils::AddEdge(gru_node->GetInDataAnchor(INDEX_3)->GetPeerOutAnchor(),
                              matmul_node->GetInDataAnchor(INDEX_2));
    }
  }

  return matmul_node;
}


bool DynamicGRUV2FusionPass::JudgeSplit(ge::NodePtr gru_node, bool& result) const {
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info,
                                                                                     optional_info) != fe::SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get platform_info failed."),
                    return false);
  uint64_t l1_size = platform_info.ai_core_spec.l1_size;
  uint32_t core_num = platform_info.soc_info.ai_core_cnt;

  int t_size = gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(0);
  float m_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(1));
  float x_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(INDEX_2));
  float h_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(INDEX_2).GetOriginShape().GetDim(0));
  m_size = ceil(m_size / BLOCKSIZE) * BLOCKSIZE;
  x_size = ceil(x_size / BLOCKSIZE) * BLOCKSIZE;
  h_size = ceil(h_size / BLOCKSIZE) * BLOCKSIZE;
  float weight_size = (x_size + h_size) * HIDDEN_NUM * h_size * CONST_NUM_2;

  // weight can be entirely loaded in L1
  if (ceil(m_size / BLOCKSIZE) >= core_num && weight_size < l1_size * THREE_QUARTERS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "weight can be entirely loaded in L1.");
    result = false;
    return true;
  }

  // after cores enabled, weight can be loaded in L1
  if (weight_size / (h_size / BLOCKSIZE) * ceil((h_size / BLOCKSIZE) / core_num) < l1_size * THREE_QUARTERS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "after cores enabled, weight can be loaded in L1.");
    gruv2::GruCalcParam param({t_size, m_size, x_size, h_size, core_num});
    float solution1_cost = gruv2::CalcSolution1(param);
    float solution2_cost = gruv2::CalcSolution2(param);
    if (solution1_cost < solution2_cost) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "keeping single op is better.");
      result = false;
      return true;
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "splitting ops is better.");
  result = true;
  return true;
}

Status DynamicGRUV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  // get gru_node
  ge::NodePtr gru_node = GetNodeFromMapping(PATTERN_GRUV2_NODE, mapping);

  float t_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(0));
  float m_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(1));
  float x_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(INDEX_2));
  float h_size = static_cast<float>(gru_node->GetOpDesc()->GetInputDesc(INDEX_2).GetOriginShape().GetDim(0));
  if (PatternFusionUtil::IsUnknownShape(t_size) ||
      PatternFusionUtil::IsUnknownShape(m_size) ||
      PatternFusionUtil::IsUnknownShape(x_size) ||
      PatternFusionUtil::IsUnknownShape(h_size)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DynamicGRUV2FusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  int64_t input_x = gru_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDim(INDEX_2);
  int64_t hidden_size = gru_node->GetOpDesc()->GetInputDesc(INDEX_2).GetOriginShape().GetDim(0);
  if ((input_x % ALIGN_16 != 0) || (hidden_size % ALIGN_16 != 0)) {
    return NOT_CHANGED;
  }

  bool split = false;
  if (!JudgeSplit(gru_node, split)) {
    return FAILED;
  }

  if (!split) {
    return NOT_CHANGED;
  }

  // add matmul
  ge::NodePtr matmul_node = AddMatmulNode(gru_node, graph, new_nodes);
  FUSION_PASS_CHECK(matmul_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddMatmulNode failed, fusion failed."),
                    return FAILED);

  // add split node
  ge::NodePtr split_node = AddSplitNode(gru_node, matmul_node, graph, new_nodes);
  FUSION_PASS_CHECK(split_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddSplitNode failed, fusion failed."),
                    return FAILED);

  // unlink all
  NodeUtils::UnlinkAll(*gru_node);
  // remove gru_node from graph
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(gru_node),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", gru_node->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

REGISTER_PASS("DynamicGRUV2FusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, DynamicGRUV2FusionPass);
}  // namespace fe
