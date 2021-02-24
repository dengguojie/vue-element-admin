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

#include "adaptive_avg_pool_grad_fusion_pass.h"

#include <map>
#include <memory>

#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace {
const string PATTERN_FUSEDNODE = "AdaptiveAvgPool2dGrad";
const string FUSED_NODE = "AdaptiveAvgPool2dGrad";
const int INDEX_BATCH = 0;
const int INDEX_CHANNEL = 1;
const int INDEX_H = 2;
const int INDEX_W = 3;
}  // namespace

namespace fe {
template <class T>
void UpdataAvgPoolGradAssistValue(T &assist, T value, int batch, int channel,
                                  int mat_hw_size, int position) {
  T *matrix = &assist;
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      matrix[b * channel * mat_hw_size + c * mat_hw_size + position] = value;
    }
  }
}

vector<FusionPattern *> AdaptiveAvgPoolGradFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("AdaptiveAvgPool2dGradFusion");
  if (!pattern) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed.");
    return patterns;
  }
  // FUSED_NODE should be the OpType
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void AdaptiveAvgPoolGradFusionPass::GetNodeInfo(ge::NodePtr node) {
  ge::OpDescPtr node_desc = node->GetOpDesc();
  ge::GeTensorDesc input_tensor_desc = node_desc->GetInputDesc(0);
  this->input_dims = input_tensor_desc.GetShape().GetDims();
  ge::AttrUtils::GetListInt(node_desc, "orig_input_shape", this->output_dims);
  this->GenKernelIndex(this->h_kernel_index, this->input_dims[INDEX_H],
                       this->output_dims[INDEX_H]);
  this->GenKernelIndex(this->w_kernel_index, this->input_dims[INDEX_W],
                       this->output_dims[INDEX_W]);
  return;
}

// calculate the start index of kernel
int AdaptiveAvgPoolGradFusionPass::StartIndex(int a, int b, int c) const {
  if (b == 0) {
    OP_LOGE("divied by zero error", "get start index failed.");
    return 0;
  }
  return static_cast<int>(std::floor(static_cast<float>(a * c) / b));
}

// calculate the end index of kernel
int AdaptiveAvgPoolGradFusionPass::EndIndex(int a, int b, int c) const {
  if (b == 0) {
    OP_LOGE("divied by zero error", "get end index failed.");
    return 0;
  }
  return static_cast<int>(std::ceil(static_cast<float>((a + 1) * c) / b));
}

// calculate the range of kernel index
void AdaptiveAvgPoolGradFusionPass::GenKernelIndex(vector<vector<int>> &kernel,
                                                   int in_size,
                                                   int out_size) const {
  kernel.resize(in_size);
  for (int i = 0; i < in_size; i++) {
    kernel[i].push_back(this->StartIndex(i, in_size, out_size));
    kernel[i].push_back(this->EndIndex(i, in_size, out_size));
  }
  return;
}

void AdaptiveAvgPoolGradFusionPass::GenBatchMatMulAssistMatrix(
    uint16_t &matrix, bool is_left_mat) const {
  int data_index = INDEX_H;  // get data from input tensor NCHW.
  vector<vector<int>> kernel_index = this->h_kernel_index;
  if (!is_left_mat) {
    data_index = INDEX_W;
    kernel_index = this->w_kernel_index;
  }

  fp16_t tmp;
  tmp = 1.0;
  int batch = this->input_dims[INDEX_BATCH];
  int channel = this->input_dims[INDEX_CHANNEL];
  int grad_in_size = this->input_dims[data_index];
  int grad_out_size = this->output_dims[data_index];
  int mat_hw_size = grad_in_size * grad_out_size;
  for (int i = 0; i < grad_in_size; i++) {
    int position = 0;
    for (int j = kernel_index[i][0]; j < kernel_index[i][1]; j++) {
      if (is_left_mat) {
        position = j * grad_in_size + i;
      } else {
        position = i * grad_out_size + j;
      }
      UpdataAvgPoolGradAssistValue(matrix, tmp.val, batch, channel, mat_hw_size,
                                   position);
    }
  }
  return;
}

bool AdaptiveAvgPoolGradFusionPass::GenVectorMulAssistMatrix(
    float &matrix) const {
  int batch = this->input_dims[INDEX_BATCH];
  int channel = this->input_dims[INDEX_CHANNEL];
  int mat_hw_size = this->input_dims[INDEX_H] * this->input_dims[INDEX_W];
  for (int i = 0; i < this->input_dims[INDEX_H]; i++) {
    int h_param = this->h_kernel_index[i][1] - this->h_kernel_index[i][0];
    for (int j = 0; j < this->input_dims[INDEX_W]; j++) {
      int coefficient =
          (this->w_kernel_index[j][1] - this->w_kernel_index[j][0]) * h_param;
      if (coefficient == 0) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "coefficient is invalid.");
        return false;
      }
      float value = 1.0 / coefficient;
      int position = i * input_dims[INDEX_W] + j;
      UpdataAvgPoolGradAssistValue(matrix, value, batch, channel, mat_hw_size,
                                   position);
    }
  }
  return true;
}

void AdaptiveAvgPoolGradFusionPass::SetTensorDesc(
    ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims,
    const ge::DataType &dtype, const ge::Format &format) const {
  ge::GeShape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(dtype);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginDataType(dtype);
  tensorDesc.SetOriginFormat(format);
  return;
}

Status AdaptiveAvgPoolGradFusionPass::CreateVectorMulWeightNode(
    ge::NodePtr &vmul_node) const {
  int64_t size = this->input_dims[INDEX_BATCH] *
                 this->input_dims[INDEX_CHANNEL] * this->input_dims[INDEX_H] *
                 this->input_dims[INDEX_W];
  vector<int64_t> weight_dims = {size};
  OP_LOGI(FUSED_OP_TYPE.c_str(), "calculate weight size. %d", size);

  unique_ptr<float[]> data(new (std::nothrow) float[size]());
  const float init_value = 1.0;
  if (NnSet(size, init_value, *reinterpret_cast<float *>(data.get())) !=
      SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet data failed.");
    return FAILED;
  }

  if (!(this->GenVectorMulAssistMatrix(*data.get()))) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "gen assist matrix fail.");
    return PARAM_INVALID;
  }

  ge::GeTensorDesc desc;
  this->SetTensorDesc(desc, weight_dims, ge::DT_FLOAT, ge::FORMAT_ND);
  ge::GeTensorPtr weight_ptr = std::make_shared<ge::GeTensor>(
      desc, reinterpret_cast<uint8_t *>(data.get()), size * sizeof(float));
  if (!weight_ptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create weight failed.");
    return FAILED;
  }

  vector<ge::GeTensorPtr> weights = {weight_ptr};
  ge::OpDescUtils::SetWeights(vmul_node, weights);
  auto const_input_nodes = OpDescUtils::GetConstInputs(vmul_node);
  if (const_input_nodes.empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "ConstInputNodes is empty, fusion failed.");
    return PARAM_INVALID;
  }

  NodePtr const_input = const_input_nodes[0];
  const_input->GetOpDesc()->SetType("Const");
  return SUCCESS;
}

Status AdaptiveAvgPoolGradFusionPass::CreateBatchMatMulWeightNode(
    ge::NodePtr &matmul_node, vector<int64_t> &weight_dims,
    bool is_left_mat) const {
  int64_t size = 1;
  for (vector<int64_t>::const_iterator it = weight_dims.begin();
       it != weight_dims.end(); it++) {
    size *= *it;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "calculate weight size. %d", size);

  unique_ptr<uint16_t[]> data(new (std::nothrow) uint16_t[size]());
  const uint16_t init_value = 0;
  if (NnSet(size, init_value, *reinterpret_cast<uint16_t *>(data.get())) !=
      SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet data failed.");
    return FAILED;
  }
  this->GenBatchMatMulAssistMatrix(*data.get(), is_left_mat);

  ge::GeTensorDesc desc;
  this->SetTensorDesc(desc, weight_dims, ge::DT_FLOAT16, ge::FORMAT_ND);
  ge::GeTensorPtr weight_ptr = std::make_shared<ge::GeTensor>(
      desc, reinterpret_cast<uint8_t *>(data.get()), size * sizeof(uint16_t));
  if (!weight_ptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "create weight failed.");
    return FAILED;
  }

  vector<ge::GeTensorPtr> weights = {weight_ptr};
  ge::OpDescUtils::SetWeights(matmul_node, weights);
  auto const_input_nodes = OpDescUtils::GetConstInputs(matmul_node);
  if (const_input_nodes.empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "ConstInputNodes is empty, fusion failed.");
    return PARAM_INVALID;
  }
  NodePtr const_input = const_input_nodes[0];
  const_input->GetOpDesc()->SetType("Const");
  return SUCCESS;
}

ge::NodePtr AdaptiveAvgPoolGradFusionPass::AddNewNode(
    ge::ComputeGraph &graph, ge::OpDescPtr &op_desc,
    vector<ge::NodePtr> &new_nodes, bool &fail_status) const {
  ge::NodePtr node = graph.AddNode(op_desc);
  if (!node) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
            node->GetName().c_str());
    fail_status = true;
  }
  new_nodes.push_back(node);
  return node;
}

//  Operator: VectorMul(input_data, assist)
ge::NodePtr AdaptiveAvgPoolGradFusionPass::AddVectorMulNode(
    ge::NodePtr avgpool_grad_node, ge::ComputeGraph &graph,
    vector<ge::NodePtr> &new_nodes, bool &fail_status) const {
  // create matmul desc
  ge::OpDescPtr mul_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (mul_op_desc = std::make_shared<ge::OpDesc>(avgpool_grad_node->GetName() + "Mul", "Mul")), return nullptr);

  // add input desc, input tensor of adaptive avgpool grad
  ge::GeTensorDesc x_desc =
      avgpool_grad_node->GetOpDesc()->GetInputDesc(0).Clone();
  vector<int64_t> shape_vec = {
      this->input_dims[INDEX_BATCH] * this->input_dims[INDEX_CHANNEL] *
      this->input_dims[INDEX_H] * this->input_dims[INDEX_W]};
  ge::GeShape inputShape(shape_vec);
  x_desc.SetShape(inputShape);
  x_desc.SetDataType(ge::DT_FLOAT);
  mul_op_desc->AddInputDesc("x", x_desc);

  ge::GeTensorDesc out_tensor_desc;
  vector<int64_t> output_shape_vec = {
      this->input_dims[INDEX_BATCH], this->input_dims[INDEX_CHANNEL],
      this->input_dims[INDEX_H], this->input_dims[INDEX_W]};
  this->SetTensorDesc(out_tensor_desc, output_shape_vec, ge::DT_FLOAT,
                      ge::FORMAT_ND);
  mul_op_desc->AddOutputDesc("VMul", out_tensor_desc);

  // create matmul node
  ge::NodePtr mul_node =
      this->AddNewNode(graph, mul_op_desc, new_nodes, fail_status);
  this->CreateVectorMulWeightNode(mul_node);

  // input Edge, assist * input_tensor
  ge::GraphUtils::AddEdge(
      avgpool_grad_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      mul_node->GetInDataAnchor(0));
  return mul_node;
}

//  Operator: BatchMatMul(input_data.T, assist.T)
ge::NodePtr AdaptiveAvgPoolGradFusionPass::AddLeftMatmulNode(
    ge::NodePtr avgpool_grad_node, ge::NodePtr mul_node,
    ge::ComputeGraph &graph, vector<ge::NodePtr> &new_nodes,
    bool &fail_status) const {
  // create matmul desc
  ge::OpDescPtr matmul_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmul_op_desc = std::make_shared<ge::OpDesc>(avgpool_grad_node->GetName() + "LeftBatchMatmul", "BatchMatMul")),
      return nullptr);

  // add input desc
  ge::GeTensorDesc x_desc = mul_node->GetOpDesc()->GetOutputDesc(0).Clone();
  vector<int64_t> input_shape_vec = {
      this->input_dims[INDEX_BATCH] * this->input_dims[INDEX_CHANNEL],
      this->input_dims[INDEX_H], this->input_dims[INDEX_W]};
  ge::GeShape input_shape(input_shape_vec);
  x_desc.SetShape(input_shape);
  x_desc.SetDataType(ge::DT_FLOAT16);
  matmul_op_desc->AddInputDesc("x", x_desc);

  // Add output desc. For example, GradNode inShape:[2,3,5,6]
  // outShape:[2,3,7,9], format:NCHW. Input of left bmm are input_data[2,3,5,6]
  // and left assist([2,3,7,5], shape:{N*C,grad_input_h, grad_output_h}). Result
  // of left bmm is [2,3,7,6]. Because of res = input.T x assist.T. the shape of
  // res is [2,3,6,7]
  ge::GeTensorDesc out_tensor_desc;
  vector<int64_t> output_dim = {
      this->input_dims[INDEX_BATCH] * this->input_dims[INDEX_CHANNEL],
      this->input_dims[INDEX_W], this->output_dims[INDEX_H]};
  this->SetTensorDesc(out_tensor_desc, output_dim, ge::DT_FLOAT16,
                      ge::FORMAT_ND);
  matmul_op_desc->AddOutputDesc("LeftBMM", out_tensor_desc);

  // attr
  ge::AttrUtils::SetBool(matmul_op_desc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmul_op_desc, "adj_x2", true);

  // create matmul node
  ge::NodePtr matmul_node =
      this->AddNewNode(graph, matmul_op_desc, new_nodes, fail_status);
  vector<int64_t> weight_dim = {
      this->input_dims[INDEX_BATCH] * this->input_dims[INDEX_CHANNEL],
      this->output_dims[INDEX_H], this->input_dims[INDEX_H]};
  this->CreateBatchMatMulWeightNode(matmul_node, weight_dim, true);

  // input Edge
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0),
                          matmul_node->GetInDataAnchor(0));
  return matmul_node;
}

//  Operator: BatchMatMul(input_data.T, assist)
void AdaptiveAvgPoolGradFusionPass::AddRightMatmulNode(
    ge::NodePtr avgpool_grad_node, ge::NodePtr left_matmul_node,
    ge::ComputeGraph &graph, vector<ge::NodePtr> &new_nodes,
    bool &fail_status) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add right matmul fusion begin.");
  // create matmul desc
  ge::OpDescPtr matmul_op_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (matmul_op_desc = std::make_shared<ge::OpDesc>(avgpool_grad_node->GetName() + "RightBatchMatmul", "BatchMatMul")),
      return);

  // add input desc, output tensor of left batch matmul
  ge::GeTensorDesc yDesc =
      left_matmul_node->GetOpDesc()->GetOutputDesc(0).Clone();
  matmul_op_desc->AddInputDesc("y", yDesc);

  // Add output desc. For example, grad node inShape:[2,3,5,6]
  // outShape:[2,3,7,9], format:NCHW. Input of right bmm are leftBMM
  // result([2,3,7,6]) and right assist([2,3,6,9], shape:{N*C, grad_input_w,
  // grad_output_w}). Shape of result of right bmm is [2,3,7,9]. shape:{N*C,
  // grad_output_h, grad_output_w}
  ge::GeTensorDesc out_tensor_desc;
  vector<int64_t> output_dim = {
      this->input_dims[INDEX_BATCH] * this->input_dims[INDEX_CHANNEL],
      this->output_dims[INDEX_H], this->output_dims[INDEX_W]};
  this->SetTensorDesc(out_tensor_desc, output_dim, ge::DT_FLOAT16,
                      ge::FORMAT_ND);
  matmul_op_desc->AddOutputDesc("RightBMM", out_tensor_desc);

  // attr
  ge::AttrUtils::SetBool(matmul_op_desc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmul_op_desc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmul_node =
      this->AddNewNode(graph, matmul_op_desc, new_nodes, fail_status);
  vector<int64_t> weight_dim = {
      this->input_dims[INDEX_BATCH] * this->input_dims[INDEX_CHANNEL],
      this->input_dims[INDEX_W], this->output_dims[INDEX_W]};
  this->CreateBatchMatMulWeightNode(matmul_node, weight_dim, false);

  // input Edge
  ge::GraphUtils::AddEdge(left_matmul_node->GetOutDataAnchor(0),
                          matmul_node->GetInDataAnchor(0));

  // output Edge
  for (InDataAnchorPtr in_anchor_ptr :
       avgpool_grad_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    in_anchor_ptr->UnlinkAll();
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), in_anchor_ptr);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add right matmul fusion end.");
  return;
}

Status AdaptiveAvgPoolGradFusionPass::Fusion(ge::ComputeGraph &graph,
                                             Mapping &mapping,
                                             vector<ge::NodePtr> &new_node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "Define AdaptiveAvgPoolGradFusionPass fusion begin.");
  ge::NodePtr grad_node = this->GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  this->GetNodeInfo(grad_node);

  bool is_failure = false;
  // add vector_mul
  ge::NodePtr vector_mul_node =
      this->AddVectorMulNode(grad_node, graph, new_node, is_failure);
  if (is_failure) {
    OP_LOGE(FUSED_OP_TYPE.c_str(),
            "VectorMulNode:check failed, fusion failed.");
    return FAILED;
  }

  // add left batch_mat_mul
  ge::NodePtr left_matmul_node = this->AddLeftMatmulNode(
      grad_node, vector_mul_node, graph, new_node, is_failure);
  if (is_failure) {
    OP_LOGE(FUSED_OP_TYPE.c_str(),
            "LeftMatMulNode:check failed, fusion failed.");
    return FAILED;
  }

  this->AddRightMatmulNode(grad_node, left_matmul_node, graph, new_node,
                           is_failure);
  if (is_failure) {
    OP_LOGE(FUSED_OP_TYPE.c_str(),
            "RightMatMulNode:check failed, fusion failed.");
    return FAILED;
  }

  // unlink all input of grad_node
  for (auto inAnchor : grad_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // remove grad_node from graph
  if (graph.RemoveNode(grad_node) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
            grad_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}
// register pass rule
REGISTER_PASS("AdaptiveAvgPoolGradFusionPass", BUILT_IN_GRAPH_PASS,
              AdaptiveAvgPoolGradFusionPass);
}  // namespace fe
