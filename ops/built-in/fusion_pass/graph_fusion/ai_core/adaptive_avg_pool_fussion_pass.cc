/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file adaptive_avg_pool_fussion_pass.cc
 * \brief adaptive avgPool fusion pass
 */
#include "adaptive_avg_pool_fussion_pass.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "common/debug/log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

using namespace std;
using namespace ge;

namespace {
const uint16_t UINT_NUM_ZERO = 0;
const string PATTERN_FUSEDNODE = "AdaptiveAvgPool2d";
const string FUSED_NODE = "AdaptiveAvgPool2d";
}  // namespace

namespace fe {
// 计算索引矩阵
int StartIndex(int a, int b, int c) {
  if (b == 0) {
    OP_LOGE("divied by zero error", "get start index failed.");
    return 0;
  }
  return static_cast<int>(std::floor(static_cast<float>(a * c) / b));
}

int EndIndex(int a, int b, int c) {
  if (b == 0) {
    OP_LOGE("divied by zero error", "get end index failed.");
    return 0;
  }
  return static_cast<int>(std::ceil(static_cast<float>((a + 1) * c) / b));
}

// 获取切分方案
Status KernelSegment(vector<int64_t> in_size, vector<int64_t> out_size,
                     vector<int64_t> &h_list, vector<int64_t> &w_list) {
  int64_t i_size_h = in_size[0];
  int64_t i_size_w = in_size[1];
  int64_t o_size_h = out_size[0];
  int64_t o_size_w = out_size[1];
  for (int64_t oh = 0; oh < o_size_h; oh++) {
    int istart_h = StartIndex(oh, o_size_h, i_size_h);
    int iend_h = EndIndex(oh, o_size_h, i_size_h);
    int k_h = iend_h - istart_h;
    h_list.push_back(k_h);
  }
  for (int64_t ow = 0; ow < o_size_w; ow++) {
    int istart_w = StartIndex(ow, o_size_w, i_size_w);
    int iend_w = EndIndex(ow, o_size_w, i_size_w);
    int k_w = iend_w - istart_w;
    w_list.push_back(k_w);
  }
  return SUCCESS;
}

// 处理左辅助矩阵
Status ProcessMatrixLeft(vector<vector<float>> &arr, vector<int64_t> size_h) {
  int64_t size_arr = arr.size();
  int64_t i_size_h = arr[0].size();
  int64_t start = 0;

  for (int64_t i = 0; i < size_arr; i++) {
    int64_t lens = size_h[i];
    start = StartIndex(i, size_arr, i_size_h);
    for (int64_t j = start; j < (start + lens) && j < i_size_h; j++) {
      arr[i][j] = 1.0;
    }
  }
  return SUCCESS;
}

//  处理右辅助矩阵
Status ProcessMatrixRight(vector<vector<float>> &arr, vector<int64_t> size_w) {
  int64_t size_arr = arr[0].size();
  int64_t i_size_w = arr.size();
  int64_t start = 0;
  for (int64_t i = 0; i < size_arr; i++) {
    int64_t lens = size_w[i];
    start = StartIndex(i, size_arr, i_size_w);
    for (int64_t j = start; j < (start + lens) && j < i_size_w; j++) {
      arr[j][i] = 1.0;
    }
  }
  return SUCCESS;
}

//  处理乘法辅助矩阵
Status ProcessMatrixMul(vector<vector<float>> &arr, vector<int64_t> size_h,
                        vector<int64_t> size_w) {
  int64_t h_size = size_h.size();
  int64_t w_size = size_w.size();
  for (int64_t i = 0; i < h_size; i++) {
    for (int64_t j = 0; j < w_size; j++) {
      if(size_h[i] == 0 || size_w[j] == 0) {
        OP_LOGE("ProcessMatrixMul", "ProcessMatrixMul divied by zero error.");
        return FAILED;
      }
      arr[i][j] = 1.0 / (float)(size_h[i] * size_w[j]);
    }
  }
  return SUCCESS;
}

//  将辅助矩阵转化成tensor
Status ArrToTensor(vector<float> &arr_tensor, vector<vector<float>> arr) {
  auto tensor_size = arr_tensor.size();
  auto size_n = arr.size();
  auto size_m = arr[0].size();
  if(size_m == 0 || size_n == 0) {
    OP_LOGE("AdaptiveAvgPool2d", "ArrToTensor divied by zero error.");
    return FAILED;
  }
  auto arr_size = size_m * size_n;
  for (int64_t i = 0; i < (static_cast<int64_t>(tensor_size)); i++) {
    int64_t temp_index = i % arr_size;
    int64_t m = temp_index / size_m;
    int64_t n = temp_index % size_m;
    arr_tensor[i] = arr[m][n];
  }
  return SUCCESS;
}

// 根据输入shape输出两个具体辅助矩阵tensor
Status AdaptiveAvgPool2dPass::AdaptiveValueGen(
    vector<int64_t> &input_shape, vector<int64_t> &output_shape,
    vector<float> &left_tensor, vector<float> &right_tensor,
    vector<float> &mul_tensor) const {
  int64_t input_lens = input_shape.size();
  int64_t output_lens = output_shape.size();
  int64_t lens_one = 1;
  int64_t lens_two = 2;
  vector<int64_t> input_size = {input_shape[input_lens - lens_two],
                                input_shape[input_lens - lens_one]};
  vector<int64_t> output_size = {output_shape[output_lens - lens_two],
                                 output_shape[output_lens - lens_one]};
  // 定义切分矩阵
  vector<int64_t> h_list;
  vector<int64_t> w_list;
  // 获取kernel切分矩阵
  FUSION_PASS_CHECK(KernelSegment(input_size, output_size, h_list, w_list) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get KernelSegment failed."),
                    return FAILED);
  // 定义两个辅助矩阵
  int n_left = output_size[0];
  int m_left = input_size[0];
  int n_right = input_size[1];
  int m_right = output_size[1];
  vector<vector<float>> arr_left(n_left, vector<float>(m_left));
  vector<vector<float>> arr_right(n_right, vector<float>(m_right));
  // 构造乘法矩阵
  vector<vector<float>> arr_mul(n_left, vector<float>(m_right));
  // 处理左右辅助矩阵得到二维的矩阵
  FUSION_PASS_CHECK(ProcessMatrixLeft(arr_left, h_list) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get ProcessMatrixLeft failed."),
                    return FAILED);
  FUSION_PASS_CHECK(ProcessMatrixRight(arr_right, w_list) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get ProcessMatrixRight failed."),
                    return FAILED);
  FUSION_PASS_CHECK(ProcessMatrixMul(arr_mul, h_list, w_list) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get ProcessMatrixMul failed."),
                    return FAILED);
  // 将辅助矩阵转成tensor
  FUSION_PASS_CHECK(ArrToTensor(left_tensor, arr_left) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get left_tensor failed."),
                    return FAILED);
  FUSION_PASS_CHECK(ArrToTensor(right_tensor, arr_right) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get right_tensor failed."),
                    return FAILED);
  FUSION_PASS_CHECK(ArrToTensor(mul_tensor, arr_mul) != SUCCESS,
                    OP_LOGE("AdaptiveAvgPool2d", "get mul_tensor failed."),
                    return FAILED);
  return SUCCESS;
}

//  获取输出的shape
Status GetOutputShape(vector<int64_t> &input_shape,
                      vector<int64_t> &output_size,
                      vector<int64_t> &output_shape) {
  int64_t input_size = input_shape.size();
  int64_t lens_two = 2;
  for (int64_t i = 0; i < input_size - lens_two; i++) {
    output_shape.push_back(input_shape[i]);
  }
  output_shape.push_back(output_size[0]);
  output_shape.push_back(output_size[1]);
  return SUCCESS;
}

//  获取输出的shape
Status GetBatOneShape(vector<int64_t> &input_shape,
                      vector<int64_t> &output_shape,
                      vector<int64_t> &bat_shape) {
  int64_t input_size = input_shape.size();
  int64_t lens_one = 1;
  int64_t lens_two = 2;

  for (auto i = 0; i < input_size - lens_two; i++) {
    bat_shape.push_back(input_shape[i]);
  }
  bat_shape.push_back(output_shape[input_size - lens_two]);
  bat_shape.push_back(input_shape[input_size - lens_one]);
  return SUCCESS;
}

// 获取辅助矩阵的tensor shape
Status GetTensorShape(vector<int64_t> &input_shape,
                      vector<int64_t> &output_shape,
                      vector<int64_t> &left_tensor_shape,
                      vector<int64_t> &right_tensor_shape) {
  int64_t input_size = input_shape.size();
  int64_t lens_one = 1;
  int64_t lens_two = 2;
  for (int64_t i = 0; i < input_size - lens_two; i++) {
    left_tensor_shape.push_back(input_shape[i]);
    right_tensor_shape.push_back(input_shape[i]);
  }
  left_tensor_shape.push_back(output_shape[input_size - lens_two]);
  left_tensor_shape.push_back(input_shape[input_size - lens_two]);
  right_tensor_shape.push_back(input_shape[input_size - lens_one]);
  right_tensor_shape.push_back(output_shape[input_size - lens_one]);
  return SUCCESS;
}

// shape dim
int64_t GetDimNum(const vector<int64_t> &shapes) {
  auto shape_lens = shapes.size();
  int64_t dim_num = 1;
  for (size_t i = 0; i < shape_lens; i++) {
    dim_num = dim_num * shapes[i];
  }
  return dim_num;
}

Status AssistDataGen(vector<float> data, uint16_t *output) {
  if (output == nullptr) {
    OP_LOGE("batchMatmul", "output pointer is null!");
    return FAILED;
  }
  auto size_data = data.size();
  for (size_t i = 0; i < size_data; i++) {
    fp16_t tmp;
    tmp = data[i];
    output[i] = tmp.val;
  }
  return SUCCESS;
}

// set desc for node
Status AdaptiveAvgPool2dPass::SetConstDesc(vector<int64_t> &tensor_shape,
                                           ge::GeTensorDesc &tensor_desc,
                                           ge::GeTensorDesc &des_desc) const {
  // 定义辅助矩阵输入left shape
  ge::GeShape ten_shapes(tensor_shape);
  tensor_desc.SetOriginFormat(des_desc.GetOriginFormat());
  tensor_desc.SetFormat(des_desc.GetFormat());
  tensor_desc.SetOriginDataType(des_desc.GetOriginDataType());
  tensor_desc.SetDataType(des_desc.GetDataType());
  tensor_desc.SetOriginShape(ten_shapes);
  tensor_desc.SetShape(ten_shapes);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::RemoveNodes(ge::NodePtr &data_node,
                                          ge::ComputeGraph &graph) const {
  for (auto in_anchor : data_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }
  for (auto out_anchor : data_node->GetAllOutDataAnchors()) {
    if (out_anchor != nullptr) {
      out_anchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(data_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove data_node failed."),
                    return FAILED);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::Bridge(ge::NodePtr &fuse_node,
                                     ge::NodePtr &one_node,
                                     ge::NodePtr &two_node,
                                     ge::NodePtr &mul_node) const {
  // 将输入连到batmmOne
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(fuse_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                              one_node->GetInDataAnchor(1)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."),
      return FAILED);
  // 将batchOne的输出连到batchTwo的输入
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(one_node->GetOutDataAnchor(0),
                                         two_node->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s to fusion node:%s failed.",
              one_node->GetName().c_str(), two_node->GetName().c_str()),
      return FAILED);

  // 将batchTwo的输出连到mulNode的输入
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(two_node->GetOutDataAnchor(0),
                                         mul_node->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s to fusion node:%s failed.",
              two_node->GetName().c_str(), mul_node->GetName().c_str()),
      return FAILED);
  // 将batmmTwo的输出连到输出
  for (auto in_data_anchor :
       fuse_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::RemoveEdge(fuse_node->GetOutDataAnchor(0),
                                   in_data_anchor) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove prod and outnode edge failed."),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0),
                                in_data_anchor) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."),
        return FAILED);
  }
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::CreatOneNode(
    ge::NodePtr &one_node, ge::NodePtr &fuse_node, ge::ComputeGraph &graph,
    vector<ge::NodePtr> &new_nodes, ge::GeShape &bat_one_outshape) const {
  ge::OpDescPtr batmm_one_desc_ptr;
  FUSION_PASS_MAKE_SHARED((batmm_one_desc_ptr = std::make_shared<ge::OpDesc>(
                               "BatchMatMul", "BatchMatMul")),
                          return INTERNAL_ERROR);
  ge::GeTensorDesc output_descs = fuse_node->GetOpDesc()->GetOutputDesc(0);
  batmm_one_desc_ptr->AddOutputDesc("y", output_descs);
  batmm_one_desc_ptr->SetType("BatchMatMul");
  ge::OpDescPtr ada_desc_ptr = fuse_node->GetOpDesc();
  batmm_one_desc_ptr->SetName(ada_desc_ptr->GetName() + "_BatchMatMul_1");
  // set adj_x1  attr for batmm node
  FUSION_PASS_CHECK(
      true != ge::AttrUtils::SetBool(batmm_one_desc_ptr, "adj_x1", false),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set adj_x1 attr for batmm node."),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      true != ge::AttrUtils::SetBool(batmm_one_desc_ptr, "adj_x2", false),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set adj_x2 attr for batmm node."),
      return PARAM_INVALID);
  one_node = graph.AddNode(batmm_one_desc_ptr);
  new_nodes.push_back(one_node);
  // 设置batone输出的shape
  ge::GeTensorDesc batone_input_desc = one_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc batone_output_desc = one_node->GetOpDesc()->GetOutputDesc(0);
  batone_output_desc.SetOriginShape(bat_one_outshape);
  batone_output_desc.SetShape(bat_one_outshape);
  batone_input_desc.SetDataType(ge::DT_FLOAT16);
  batone_output_desc.SetDataType(ge::DT_FLOAT16);
  one_node->GetOpDesc()->UpdateInputDesc(0, batone_input_desc);
  one_node->GetOpDesc()->UpdateOutputDesc(0, batone_output_desc);

  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::CreatTwoNode(
    ge::NodePtr &two_node, ge::NodePtr &fuse_node, ge::ComputeGraph &graph,
    vector<ge::NodePtr> &new_nodes, ge::GeShape &bat_one_outshape) const {
  ge::OpDescPtr batmm_two_desc_ptr;
  FUSION_PASS_MAKE_SHARED((batmm_two_desc_ptr = std::make_shared<ge::OpDesc>(
                               "BatchMatMul", "BatchMatMul")),
                          return INTERNAL_ERROR);
  ge::GeTensorDesc input_descs = fuse_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc output_descs = fuse_node->GetOpDesc()->GetOutputDesc(0);
  batmm_two_desc_ptr->AddInputDesc("x1", input_descs);
  batmm_two_desc_ptr->AddOutputDesc("y", output_descs);
  batmm_two_desc_ptr->SetType("BatchMatMul");
  ge::OpDescPtr ada_desc_ptr = fuse_node->GetOpDesc();
  batmm_two_desc_ptr->SetName(ada_desc_ptr->GetName() + "_BatchMatMul_2");
  // set adj_x1  attr for batmm node
  FUSION_PASS_CHECK(
      true != ge::AttrUtils::SetBool(batmm_two_desc_ptr, "adj_x1", false),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set adj_x1 attr for batmm node."),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      true != ge::AttrUtils::SetBool(batmm_two_desc_ptr, "adj_x2", false),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set adj_x2 attr for batmm node."),
      return PARAM_INVALID);
  two_node = graph.AddNode(batmm_two_desc_ptr);
  new_nodes.push_back(two_node);
  // 设置battwo输出的shapes
  ge::GeTensorDesc battwo_input_desc = two_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc bat_two_output_desc =
      two_node->GetOpDesc()->GetOutputDesc(0);
  battwo_input_desc.SetOriginShape(bat_one_outshape);
  battwo_input_desc.SetShape(bat_one_outshape);
  battwo_input_desc.SetDataType(ge::DT_FLOAT16);
  bat_two_output_desc.SetDataType(ge::DT_FLOAT16);
  two_node->GetOpDesc()->UpdateInputDesc(0, battwo_input_desc);
  two_node->GetOpDesc()->UpdateOutputDesc(0, bat_two_output_desc);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::CreatMulNode(
    ge::NodePtr &mul_node, ge::NodePtr &fuse_node, ge::ComputeGraph &graph,
    vector<ge::NodePtr> &new_nodes) const {
  ge::OpDescPtr mul_desc_ptr;
  FUSION_PASS_MAKE_SHARED(
      (mul_desc_ptr = std::make_shared<ge::OpDesc>("Mul", "Mul")),
      return INTERNAL_ERROR);
  ge::GeTensorDesc input_descs = fuse_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc output_descs = fuse_node->GetOpDesc()->GetOutputDesc(0);
  mul_desc_ptr->AddInputDesc("x", input_descs);
  mul_desc_ptr->AddOutputDesc("y", output_descs);
  mul_desc_ptr->SetType("Mul");
  ge::OpDescPtr ada_desc_ptr = fuse_node->GetOpDesc();
  mul_desc_ptr->SetName(ada_desc_ptr->GetName() + "_Mul");
  mul_node = graph.AddNode(mul_desc_ptr);
  new_nodes.push_back(mul_node);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::CreatFuseNode(
    ge::NodePtr &fuse_node, vector<int64_t> &input_shape,
    vector<int64_t> &output_shape, vector<int64_t> &bat_one_shape) const {
  FUSION_PASS_CHECK(
      fuse_node == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
      return PARAM_INVALID);
  ge::OpDescPtr adaptive_desc_ptr = fuse_node->GetOpDesc();
  FUSION_PASS_CHECK(adaptive_desc_ptr == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // step2: get adaptive description
  ge::GeTensorDesc adaptive_input_tensor =
      fuse_node->GetOpDesc()->GetInputDesc(0);
  ge::GeShape adaptive_input_shape = adaptive_input_tensor.GetShape();
  // step3: get input_shape and output_size
  input_shape = adaptive_input_shape.GetDims();
  vector<int64_t> output_size;
  ge::AttrUtils::GetListInt(fuse_node->GetOpDesc(), "output_size", output_size);
  // get output_shape
  FUSION_PASS_CHECK(GetOutputShape(input_shape, output_size, output_shape) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get OutputShape failed."),
                    return FAILED);
  // 计算batone输出的shape
  FUSION_PASS_CHECK(GetBatOneShape(input_shape, output_shape, bat_one_shape) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get BatOneShape failed."),
                    return FAILED);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::LeftConstNode(
    vector<int64_t> &left_tensor_shape, ge::GeTensorDesc &input_desc1,
    ge::GeTensorPtr &assit_left_ptr, vector<float> &left_tensor,
    ge::GeTensorDesc &left_tensor_desc) const {
  int64_t left_dim_num = GetDimNum(left_tensor_shape);
  Status ret = SetConstDesc(left_tensor_shape, left_tensor_desc, input_desc1);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "LeftConstNode fusion failed"),
                    return FAILED);
  unique_ptr<uint16_t[]> left_assit(new (std::nothrow)
                                        uint16_t[left_dim_num]());
  FUSION_PASS_CHECK(left_assit.get() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "left_assit is NULL"),
                    return PARAM_INVALID);
  // 初始化辅助矩阵
  ret = NnSet(left_dim_num, UINT_NUM_ZERO,
              *reinterpret_cast<uint16_t *>(left_assit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                    return ret);
  ret = AssistDataGen(left_tensor, left_assit.get());
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get AssistDataGen failed"),
                    return FAILED);
  left_tensor_desc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED(
      (assit_left_ptr = std::make_shared<ge::GeTensor>(
           left_tensor_desc, reinterpret_cast<uint8_t *>(left_assit.get()),
           left_dim_num * sizeof(uint16_t))),
      assit_left_ptr = nullptr;
      return PARAM_INVALID);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::MidConstNode(
    vector<int64_t> &input_shape, ge::GeTensorDesc &input_desc1,
    ge::GeTensorPtr &assit_mid_ptr, ge::GeTensorDesc &mid_tensor_desc) const {
  int64_t mid_dim_num = GetDimNum(input_shape);
  // 定义辅助矩阵输入mid shape
  Status ret = SetConstDesc(input_shape, mid_tensor_desc, input_desc1);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get MidConstNode failed"),
                    return FAILED);
  unique_ptr<uint16_t[]> mid_assit(new (std::nothrow) uint16_t[mid_dim_num]());
  FUSION_PASS_CHECK(mid_assit.get() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "mid_assit is NULL"),
                    return PARAM_INVALID);
  ret = NnSet(mid_dim_num, UINT_NUM_ZERO,
              *reinterpret_cast<uint16_t *>(mid_assit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet mid_assit failed."),
                    return ret);
  // 绑定中间矩阵
  mid_tensor_desc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED(
      (assit_mid_ptr = std::make_shared<ge::GeTensor>(
           mid_tensor_desc, reinterpret_cast<uint8_t *>(mid_assit.get()),
           mid_dim_num * sizeof(uint16_t))),
      assit_mid_ptr = nullptr;
      return PARAM_INVALID);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::RightConstNode(
    vector<int64_t> &right_tensor_shape, ge::GeTensorDesc &input_desc1,
    ge::GeTensorPtr &assit_right_ptr, vector<float> &right_tensor,
    ge::GeTensorDesc &right_tensor_desc) const {
  int64_t right_dim_num = GetDimNum(right_tensor_shape);
  // 定义辅助矩阵输入right shape
  Status ret = SetConstDesc(right_tensor_shape, right_tensor_desc, input_desc1);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get RightConstNode failed"),
                    return FAILED);
  unique_ptr<uint16_t[]> right_assit(new (std::nothrow)
                                         uint16_t[right_dim_num]());
  FUSION_PASS_CHECK(right_assit.get() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "right_assit is NULL"),
                    return PARAM_INVALID);
  ret = NnSet(right_dim_num, UINT_NUM_ZERO,
              *reinterpret_cast<uint16_t *>(right_assit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                    return ret);
  // 给辅助矩阵赋值
  ret = AssistDataGen(right_tensor, right_assit.get());
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get RightConstNode failed"),
                    return FAILED);
  right_tensor_desc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED(
      (assit_right_ptr = std::make_shared<ge::GeTensor>(
           right_tensor_desc, reinterpret_cast<uint8_t *>(right_assit.get()),
           right_dim_num * sizeof(uint16_t))),
      assit_right_ptr = nullptr;
      return PARAM_INVALID);
  return SUCCESS;
}

Status AdaptiveAvgPool2dPass::MulConstNode(
    vector<int64_t> &output_shape, ge::GeTensorDesc &input_desc1,
    ge::GeTensorPtr &assit_mul_ptr, vector<float> &mul_tensor,
    ge::GeTensorDesc &mul_tensor_desc) const {
  int64_t mul_dim_num = GetDimNum(output_shape);
  // 定义辅助矩阵输入mul shape
  Status ret = SetConstDesc(output_shape, mul_tensor_desc, input_desc1);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get RightConstNode failed"),
                    return FAILED);
  unique_ptr<uint16_t[]> mul_assit(new (std::nothrow) uint16_t[mul_dim_num]());
  FUSION_PASS_CHECK(mul_assit.get() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_assit is NULL"),
                    return PARAM_INVALID);
  ret = NnSet(mul_dim_num, UINT_NUM_ZERO,
              *reinterpret_cast<uint16_t *>(mul_assit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."),
                    return ret);
  // 给辅助矩阵赋值
  ret = AssistDataGen(mul_tensor, mul_assit.get());
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get RightConstNode failed"),
                    return FAILED);
  mul_tensor_desc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED(
      (assit_mul_ptr = std::make_shared<ge::GeTensor>(
           mul_tensor_desc, reinterpret_cast<uint8_t *>(mul_assit.get()),
           mul_dim_num * sizeof(uint16_t))),
      assit_mul_ptr = nullptr;
      return PARAM_INVALID);
  return SUCCESS;
}

vector<FusionPattern *> AdaptiveAvgPool2dPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  // 类名 AdaptiveAvgPool2dPass 用户可以自行定义，不能和别的融合规则重名
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("AdaptiveAvgPool2dPass");
  FUSION_PASS_CHECK(
      pattern == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
      return patterns);
  // 这里的第二个参数 FUSED_NODE 即算子的类型定义 OpType
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status AdaptiveAvgPool2dPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                     vector<ge::NodePtr> &new_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion begin.");
  // step1: get fused Node
  ge::NodePtr adaptive_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(adaptive_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "adaptive_node is null."),
                    return PARAM_INVALID);
  // 定义输入输出的shape
  vector<int64_t> input_shape;
  vector<int64_t> output_shape;
  vector<int64_t> bat_one_shape;
  Status ret =
      CreatFuseNode(adaptive_node, input_shape, output_shape, bat_one_shape);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  ge::GeShape bat_one_outshape(bat_one_shape);
  //  step3: 定义新的node节点
  ge::GeTensorDesc input_desc1 = adaptive_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc outputDesc = adaptive_node->GetOpDesc()->GetOutputDesc(0);
  ge::NodePtr batmm_one_node;
  ret = CreatOneNode(batmm_one_node, adaptive_node, graph, new_nodes,
                     bat_one_outshape);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  FUSION_PASS_CHECK(batmm_one_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "batmm_one_node is null."),
                    return PARAM_INVALID);
  // 定义一个新的节点 batmmTwo
  ge::NodePtr batmm_two_node;
  ret = CreatTwoNode(batmm_two_node, adaptive_node, graph, new_nodes,
                     bat_one_outshape);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  FUSION_PASS_CHECK(batmm_two_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "batmm_two_node is null."),
                    return PARAM_INVALID);
  // 定义一个新的节点 mul_node
  ge::NodePtr mul_node;
  ret = CreatMulNode(mul_node, adaptive_node, graph, new_nodes);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node is null."),
                    return PARAM_INVALID);
  // 设置shape
  vector<int64_t> left_tensor_shape;
  vector<int64_t> right_tensor_shape;
  // 获取辅助矩阵的shape信息
  ret = GetTensorShape(input_shape, output_shape, left_tensor_shape,
                       right_tensor_shape);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // get assist lens
  int64_t left_dim_num = GetDimNum(left_tensor_shape);
  int64_t right_dim_num = GetDimNum(right_tensor_shape);
  int64_t mul_dim_num = GetDimNum(output_shape);
  vector<float> left_tensor(left_dim_num);
  vector<float> right_tensor(right_dim_num);
  vector<float> mul_tensor(mul_dim_num);
  //  gen value to assist matirx
  ret = AdaptiveValueGen(input_shape, output_shape, left_tensor, right_tensor,
                         mul_tensor);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // 设置左矩阵
  ge::GeTensorPtr assit_left_ptr = nullptr;
  ge::GeTensorDesc left_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ret = LeftConstNode(left_tensor_shape, input_desc1, assit_left_ptr,
                      left_tensor, left_tensor_desc);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // 设置中矩阵
  ge::GeTensorPtr assit_mid_ptr = nullptr;
  ge::GeTensorDesc mid_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ret = MidConstNode(input_shape, input_desc1, assit_mid_ptr, mid_tensor_desc);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // 构造右assit
  ge::GeTensorPtr assit_right_ptr = nullptr;
  ge::GeTensorDesc right_tensor_desc(GeShape(), ge::FORMAT_NCHW,
                                     ge::DT_FLOAT16);
  ret = RightConstNode(right_tensor_shape, input_desc1, assit_right_ptr,
                       right_tensor, right_tensor_desc);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // 构造mul 矩阵
  ge::GeTensorPtr assit_mul_ptr = nullptr;
  ge::GeTensorDesc mul_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ret = MulConstNode(output_shape, input_desc1, assit_mul_ptr, mul_tensor,
                     mul_tensor_desc);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // 将赋值矩阵连接batchone并设置类型
  vector<ge::GeTensorPtr> left_weights = {assit_left_ptr, assit_mid_ptr};
  ge::OpDescUtils::SetWeights(batmm_one_node, left_weights);
  auto const_left_input_node = OpDescUtils::GetConstInputs(batmm_one_node);

  FUSION_PASS_CHECK(const_left_input_node.size() < 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), " left const nodes size less than 2."),
                    return FAILED);
  NodePtr const_left_input = const_left_input_node[0];
  FUSION_PASS_CHECK(const_left_input == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const_left_input is null."),
                    return PARAM_INVALID);
  const_left_input->GetOpDesc()->SetType("Const");
  NodePtr const_mid_input = const_left_input_node[1];
  FUSION_PASS_CHECK(const_mid_input == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const_mid_input is null."),
                    return PARAM_INVALID);
  const_mid_input->GetOpDesc()->SetType("Const");
  // remove mid node
  ret = RemoveNodes(const_mid_input, graph);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // 将赋值矩阵连接batchTwo
  vector<ge::GeTensorPtr> right_weights = {assit_right_ptr};
  ge::OpDescUtils::SetWeights(batmm_two_node, right_weights);
  auto const_right_input_node = OpDescUtils::GetConstInputs(batmm_two_node);
  FUSION_PASS_CHECK(const_right_input_node.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), " right const nodes size less than 1."),
                    return FAILED);
  NodePtr const_right_input = const_right_input_node[0];
  FUSION_PASS_CHECK(const_right_input == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const_right_input is null."),
                    return PARAM_INVALID);
  const_right_input->GetOpDesc()->SetType("Const");
  // add edge to batmmOne and two
  // 将赋值矩阵连接Mul Node
  vector<ge::GeTensorPtr> mul_weights = {assit_mul_ptr};
  ge::OpDescUtils::SetWeights(mul_node, mul_weights);
  auto const_mul_input_node = OpDescUtils::GetConstInputs(mul_node);
  FUSION_PASS_CHECK(const_mul_input_node.size() < 1,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), " const mul node size less than 1."),
                    return FAILED);
  NodePtr const_mul_input = const_mul_input_node[0];
  FUSION_PASS_CHECK(const_mul_input == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const_mul_input is null."),
                    return PARAM_INVALID);
  const_mul_input->GetOpDesc()->SetType("Const");
  // add edge to batmmOne and two
  ret = Bridge(adaptive_node, batmm_one_node, batmm_two_node, mul_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  // remove adaptive_node
  ret = RemoveNodes(adaptive_node, graph);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion failed"),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AdaptiveAvgPool2dPass fusion end");
  return SUCCESS;
}
// register pass rule
REGISTER_PASS("AdaptiveAvgPool2dPass", BUILT_IN_GRAPH_PASS,
              AdaptiveAvgPool2dPass);
}  // namespace fe