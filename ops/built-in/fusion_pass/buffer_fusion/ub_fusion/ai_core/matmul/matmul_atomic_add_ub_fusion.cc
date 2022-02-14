/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file matmul_atmoic_add_ub_fusion.cpp
 * \brief
 * support dtype: float16
 * unsupported scene: cube_vector_split
 * trans MatMul(out:FRACTAL_NZ/float16) to MatMul(out:FRACTAL_NZ/float32) -> Cast(out:FRACTAL_NZ/float16)
 * trans MatMul(out:ND/float16) to MatMul(out:FRACTAL_NZ/float32) -> Cast(out:ND/float16)
 * -> TransData(out:FRACTAL_NZ/float16)
 */
#include "matmul_atomic_add_ub_fusion.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "aicore_util_attr_define.h"
#include "anchor_util.h"
#include "common/util/platform_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char katternMatmul[] = "matmul";

static const std::string kOpTypeFullyConnection = "FullyConnection";
static const std::string kOpTypeTransdata = "TransData";
static const std::string kOpTypeCast = "Cast";

static const unsigned int kIndexM = 0;
static const unsigned int kIndexK = 1;
static const unsigned int kIndexN = 2;
static const int kFloat16Size = 2;
static const float kAtomicAddBwLoseRadio = 0.5;
static const int kBlockSize = 16;
static const int kScheduleTime = 10;
static const int kLimitCoreNumber = 8;

vector<BufferFusionPattern *> MatmulAtomicAddUbFusion::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "MatmulAtomicAddUbFusion";

  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK((pattern == nullptr), CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "new an object failed."),
                    return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());

  // define pattern rules
  pattern
      ->AddOpDesc(katternMatmul, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({katternMatmul});
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}

bool MatmulAtomicAddUbFusion::EnableAtomicAdd(const ge::NodePtr &matmul_node) {
  bool enable_atomic_add = true;
  auto matmul_name = matmul_node->GetName();

  if (!NeedSplitK(matmul_node)) {
    enable_atomic_add = false;
    OP_LOGD(kFusedOpType.c_str(), "the Matmul node %s not support atomic add.", matmul_name.c_str());
  } else {
    OP_LOGD(kFusedOpType.c_str(), "the Matmul node %s support atomic add.", matmul_name.c_str());
  }

  size_t matmul_base_input_num = 2;
  if (matmul_node->GetInDataNodes().size() > matmul_base_input_num) {
    enable_atomic_add = false;
    OP_LOGD(kFusedOpType.c_str(),
            "the Matmul node %s's input node number is %i, may have biasadd graph fusion, not support atomic add.",
            matmul_name.c_str(), matmul_node->GetInDataNodes().size());
  }
  if (matmul_node->GetType() == kOpTypeFullyConnection) {
    OP_LOGD(kFusedOpType.c_str(), "current node's op_type is fully_connection, not split k.");
    enable_atomic_add = false;
  }
  return enable_atomic_add;
}

bool MatmulAtomicAddUbFusion::NeedSplitK(const ge::NodePtr &matmul_node) {
  vector<int64_t> shapes = GetMatMulDims(matmul_node);
  auto src_dtype = matmul_node->GetOpDesc()->GetInputDesc(0).GetDataType();

  FUSION_PASS_CHECK(src_dtype != ge::DT_FLOAT16,
                    OP_LOGD(kFusedOpType.c_str(), "only support float16 input, will not split k."),
                    return false);

  if (core_num < kLimitCoreNumber) {
    OP_LOGD(kFusedOpType.c_str(), "core number is less than 8 will not enable split k.");
    return false;
  }
  ge::DataType out_dtype = matmul_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::Format out_format = matmul_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
  bool need_split_k = false;
  auto m_shape = shapes[kIndexM];
  auto k_shape = shapes[kIndexK];
  auto n_shape = shapes[kIndexN];
  int64_t need_l2_size_value = (m_shape * k_shape + k_shape * n_shape + m_shape * n_shape) * kFloat16Size;
  int64_t hbm_bandwidth;
  int64_t l2_bandwidth;
  GetBandWidth(hbm_bandwidth, l2_bandwidth);
  int64_t cur_bandwidth = hbm_bandwidth;
  if (need_l2_size_value < l2_size) {
    cur_bandwidth = l2_bandwidth;
  }

  float min_cost =
      (static_cast<float>(core_num * (m_shape * n_shape + m_shape * k_shape + n_shape * k_shape) * kFloat16Size) /
       hbm_bandwidth);

  int64_t m_axis_outer = (m_shape + block_in - 1) / block_in;
  int64_t n_axis_outer = (n_shape + block_out - 1) / block_out;
  int64_t k_axis_outer = (k_shape + block_reduce - 1) / block_reduce;

  int m_max_dim = (m_axis_outer > core_num) ? core_num : m_axis_outer;
  int n_max_dim = (n_axis_outer > core_num) ? core_num : n_axis_outer;
  int k_max_dim = (k_axis_outer > core_num) ? core_num : k_axis_outer;
  int k_factor = 1;
  int total_max_dim = m_max_dim * k_max_dim * n_max_dim;
  for (int i = 0; i < total_max_dim; ++i) {
    int k_dim = static_cast<int>(i / (m_max_dim * n_max_dim)) + 1;
    int n_dim = static_cast<int>(i / m_max_dim) % n_max_dim + 1;
    int m_dim = i % m_max_dim + 1;
    if (m_dim * k_dim * n_dim > core_num) {
      continue;
    }
    if ((m_dim > m_axis_outer) || (n_dim > n_axis_outer) || (k_dim > k_axis_outer)) {
      continue;
    }
    vector<int> block_dims = {m_dim, k_dim, n_dim};
    float cur_cost;
    bool ret = computePerf(shapes, block_dims, cur_bandwidth, hbm_bandwidth, out_dtype, out_format, cur_cost);
    if (!ret) {
      OP_LOGW(kFusedOpType.c_str(), "Failed compute best perf corenum, will not split k.");
      return false;
    }
    if (cur_cost < min_cost) {
      min_cost = cur_cost;
      k_factor = k_dim;
    }
  }
  if (k_factor != 1) {
    need_split_k = true;
  }
  return need_split_k;
}

Status MatmulAtomicAddUbFusion::GetBandWidth(int64_t &hbm_bandwidth, int64_t &l2_bandwidth) {
  hbm_bandwidth = 0;
  l2_bandwidth = 0;
  auto iter1 = soc_hbm_bandwidth_info.find(core_num);
  if (iter1 != soc_hbm_bandwidth_info.end()) {
    hbm_bandwidth = iter1->second;
  }
  auto iter2 = soc_l2_bandwidth_info.find(core_num);
  if (iter2 != soc_l2_bandwidth_info.end()) {
    l2_bandwidth = iter2->second;
  }
  if (hbm_bandwidth == 0 || l2_bandwidth == 0) {
    OP_LOGD(kFusedOpType.c_str(), "Get bandwidth fail, use other bandwidth info");
    int distant = std::abs(core_num - 8);
    int core_num_best = 8;
    for (const auto &soc_hbm_bandwidth : soc_hbm_bandwidth_info) {
      int inner_core_num = soc_hbm_bandwidth.first;
      if (std::abs(core_num - inner_core_num) < distant) {
        distant = std::abs(core_num - inner_core_num);
        core_num_best = inner_core_num;
      }
    }
    hbm_bandwidth = soc_hbm_bandwidth_info.find(core_num_best)->second;
    l2_bandwidth = soc_l2_bandwidth_info.find(core_num_best)->second;
  }
  return true;
}

bool MatmulAtomicAddUbFusion::computePerf(vector<int64_t> shapes, vector<int> block_dims, int64_t cur_bandwidth,
                                          int64_t hbm_bandwidth, ge::DataType out_dtype, ge::Format out_format,
                                          float &cur_cost) {
  bool ret = true;
  auto m_shape = shapes[kIndexM];
  auto k_shape = shapes[kIndexK];
  auto n_shape = shapes[kIndexN];
  auto m_dim = block_dims[kIndexM];
  auto k_dim = block_dims[kIndexK];
  auto n_dim = block_dims[kIndexN];
  int64_t m_shape_inner = (m_shape + m_dim - 1) / m_dim;
  int64_t k_shape_inner = (k_shape + k_dim - 1) / k_dim;
  int64_t n_shape_inner = (n_shape + n_dim - 1) / n_dim;

  int out_data_size_fp32 = 1;
  ret = ret & getValueByKey(bytes_dtype, ge::DT_FLOAT, out_data_size_fp32);
  int out_data_size_fp16 = 1;
  ret = ret & getValueByKey(bytes_dtype, ge::DT_FLOAT16, out_data_size_fp16);
  int ori_out_data_size = 1;
  ret = ret & getValueByKey(bytes_dtype, out_dtype, ori_out_data_size);
  int in_data_size = 1;
  ret = ret & getValueByKey(bytes_dtype, ge::DT_FLOAT16, in_data_size);
  if (!ret) {
    return false;
  }

  float cast_node_cost = 0;
  float transdata_node_cost = 0;
  float mte3_cost = 0;
  cur_cost = 0;
  if (k_dim != 1) {
    mte3_cost = (static_cast<float>(k_dim * (m_shape_inner * n_shape_inner * out_data_size_fp32)) /
                 (kAtomicAddBwLoseRadio * cur_bandwidth));
    if (out_dtype == ge::DT_FLOAT16) {
      cast_node_cost = (static_cast<float>(m_shape * n_shape * out_data_size_fp32) / cur_bandwidth +
                        static_cast<float>(m_shape * n_shape * in_data_size) / hbm_bandwidth);
      // the schedule time of cast
      cur_cost += kScheduleTime;
    }
    if (out_format == ge::FORMAT_ND) {
      transdata_node_cost = (static_cast<float>(m_shape * n_shape * in_data_size) / cur_bandwidth +
                             static_cast<float>(m_shape * n_shape * out_data_size_fp16) / hbm_bandwidth);
      // the schedule time of transdata
      cur_cost += kScheduleTime;
    }
  } else {
    mte3_cost = static_cast<float>((m_shape_inner * n_shape_inner * ori_out_data_size)) / cur_bandwidth;
  }

  float base_load_cost =
      ((static_cast<float>((m_shape_inner * k_shape_inner + k_shape_inner * n_shape_inner) * in_data_size)) /
       cur_bandwidth);
  float b_repeat_load_cost =
      (static_cast<float>((m_dim - 1) * k_shape_inner * n_shape_inner * in_data_size)) / cur_bandwidth;
  float a_repeat_load_cost =
      (static_cast<float>((n_dim - 1) * k_shape_inner * m_shape_inner * in_data_size)) / cur_bandwidth;
  cur_cost +=
      base_load_cost + mte3_cost + a_repeat_load_cost + b_repeat_load_cost + cast_node_cost + transdata_node_cost;
  return true;
}

bool MatmulAtomicAddUbFusion::getValueByKey(std::unordered_map<ge::DataType, int> ori_map, ge::DataType target_key,
                                            int &target_value) {
  auto iter = ori_map.find(target_key);
  if (iter != ori_map.end()) {
    target_value = iter->second;
  } else {
    OP_LOGD(kFusedOpType.c_str(), "can't find value by key %s, need add info.", target_key);
    return false;
  }

  return true;
}

vector<int64_t> MatmulAtomicAddUbFusion::GetMatMulDims(const ge::NodePtr &matmul_node) {
  auto input0_desc = GetCurrNodeInputDesc(matmul_node, 0);
  auto input1_desc = GetCurrNodeInputDesc(matmul_node, 1);

  int64_t m_dim = 0;
  int64_t k_dim = 0;
  int64_t n_dim = 0;
  vector<int64_t> input0_dims = input0_desc->GetOriginShape().GetDims();
  vector<int64_t> input1_dims = input1_desc->GetOriginShape().GetDims();
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  auto format_x1 = matmul_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  auto format_x2 = matmul_node->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  ge::AttrUtils::GetBool(matmul_node->GetOpDesc(), "transpose_x1", transpose_x1);
  ge::AttrUtils::GetBool(matmul_node->GetOpDesc(), "transpose_x2", transpose_x2);
  int m_index = 0;
  int k_index = 1;
  int64_t cur_block_in = 1;
  int64_t cur_block_reduce = 1;
  if (transpose_x1) {
    m_index ^= 1;
    k_index ^= 1;
  }
  if (format_x1 == ge::FORMAT_FRACTAL_NZ) {
    m_index ^= 1;
    k_index ^= 1;
    cur_block_in = block_in;
    cur_block_reduce = block_reduce;
  }
  m_dim = input0_dims[m_index];
  k_dim = input0_dims[k_index];
  vector<pair<int64_t, int64_t>> input0_range;
  if (m_dim == -1) {
    input0_desc->GetShapeRange(input0_range);
    m_dim = input0_range[m_index].second;
  }
  if (k_dim == -1) {
    input0_desc->GetShapeRange(input0_range);
    k_dim = input0_range[k_index].first;
  }

  m_dim *= cur_block_in;
  k_dim *= cur_block_reduce;

  int n_index = 1;
  int64_t cur_block_out = 1;
  if (transpose_x2) {
    n_index ^= 1;
  }
  if (format_x2 == ge::FORMAT_FRACTAL_NZ) {
    n_index ^= 1;
    cur_block_out = block_out;
  }
  n_dim = input1_dims[n_index];
  vector<pair<int64_t, int64_t>> input1_range;
  if (n_dim == -1) {
    input1_desc->GetShapeRange(input1_range);
    n_dim = input1_range[n_index].second;
  }
  n_dim *= cur_block_out;

  vector<int64_t> dims = {m_dim, k_dim, n_dim};
  auto matmul_name = matmul_node->GetName();
  OP_LOGD(kFusedOpType.c_str(), "the Matmul node %s's shape is {%lld, %lld, %lld}.", matmul_name.c_str(), m_dim, k_dim,
          n_dim);
  return dims;
}

int MatmulAtomicAddUbFusion::AtomicAddType(const ge::NodePtr &matmul_node) {
  int atomic_add_type = ATOMIC_ADD_DISABLE;
  if (is_no_range || !EnableAtomicAdd(matmul_node)) {
    return atomic_add_type;
  }
  auto out_dtype = matmul_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  auto out_format = matmul_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
  atomic_add_type = ATOMIC_ADD_ENABLE;
  if (out_dtype == ge::DT_FLOAT16) {
    atomic_add_type = atomic_add_type | ATOMIC_ADD_NEED_CAST;
  }
  if (out_format == ge::FORMAT_ND) {
    atomic_add_type = atomic_add_type | ATOMIC_ADD_NEED_TRANSDATA;
  }
  return atomic_add_type;
}

Status MatmulAtomicAddUbFusion::IsDynamic(const ge::NodePtr &matmul_node, bool &is_dynamic, bool &is_no_range) {
  is_dynamic = false;
  auto input0_desc = GetCurrNodeInputDesc(matmul_node, 0);
  auto input1_desc = GetCurrNodeInputDesc(matmul_node, 1);
  auto outputDesc = GetCurrNodeOutputDesc(matmul_node, 0);
  vector<int64_t> input0_dims = input0_desc->GetOriginShape().GetDims();
  vector<int64_t> input1_dims = input1_desc->GetOriginShape().GetDims();
  vector<int64_t> all_dims;
  all_dims.resize(input0_dims.size() + input1_dims.size());
  merge(input0_dims.begin(), input0_dims.end(), input1_dims.begin(), input1_dims.end(), all_dims.begin());
  for (auto single_dim : all_dims) {
    if (single_dim < 0) {
      is_dynamic = true;
      break;
    }
  }
  if (!is_dynamic) {
    return SUCCESS;
  }
  vector<pair<int64_t, int64_t>> range_data_0;
  vector<pair<int64_t, int64_t>> range_data_1;
  FUSION_PASS_CHECK(matmul_node->GetOpDesc()->MutableInputDesc(0)->GetShapeRange(range_data_0) == ge::GRAPH_FAILED,
                    OP_LOGE(kFusedOpType, "Failed to set first shape range of MatMul."),
                    return FAILED);
  FUSION_PASS_CHECK(matmul_node->GetOpDesc()->MutableInputDesc(1)->GetShapeRange(range_data_1) == ge::GRAPH_FAILED,
                    OP_LOGE(kFusedOpType, "Failed to set second shape range of MatMul."),
                    return FAILED);

  if (IsTheRangeOfNoRange(range_data_0) || IsTheRangeOfNoRange(range_data_1)) {
    is_no_range = true;
  }
  return SUCCESS;
}

bool MatmulAtomicAddUbFusion::IsTheRangeOfNoRange(const vector<pair<int64_t, int64_t>> &range_data) {
  if (range_data.empty()) {
    OP_LOGD(kFusedOpType, "the range of MatMul is empty, is no range.");
    return true;
  }
  for (auto dim_range : range_data) {
    if ((dim_range.first == 1) && (dim_range.second == -1)) {
      OP_LOGD(kFusedOpType, "range is (1, -1), is no range.");
      return true;
    }
  }
  return false;
}

Status MatmulAtomicAddUbFusion::GenerateTransDataNode(ge::NodePtr &matmul_node, ge::NodePtr &transdata_node) {
  auto base_name = matmul_node->GetName();
  auto op_name = base_name + "_transdata";
  ge::OpDescPtr new_desc_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc_ptr = std::make_shared<ge::OpDesc>(op_name, kOpTypeTransdata)),
                          OP_LOGE(kFusedOpType.c_str(), "create %s_desc_ptr failed.", op_name.c_str());
                          new_desc_ptr = nullptr;
                          return FAILED);
  ge::GeTensorDesc matmul_out_desc = matmul_node->GetOpDesc()->GetOutputDesc(0);
  auto matmul_output_shape = matmul_out_desc.GetOriginShape().GetDims();
  ge::ComputeGraphPtr graph_ptr = matmul_node->GetOwnerComputeGraph();
  int64_t n_shape = matmul_output_shape[1];
  int64_t m_shape = matmul_output_shape[0];
  if (n_shape != -1) {
    n_shape /= block_out;
  }
  if (m_shape != -1) {
    m_shape /= block_in;
  }

  vector<int64_t> matmul_output_nz_shape = {n_shape, m_shape, block_in, block_out};

  ge::GeTensorDesc transdata_in_desc;
  ge::GeTensorDesc transdata_out_desc;

  int transdata_in_dim_cnt = 4;
  int transdata_out_dim_cnt = 2;
  transdata_in_desc.SetDataType(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType());
  transdata_in_desc.SetOriginDataType(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginDataType());
  transdata_in_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  transdata_in_desc.SetOriginFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginFormat());
  transdata_in_desc.SetShape(ge::GeShape(matmul_output_nz_shape));
  transdata_in_desc.SetOriginShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape());
  ge::TensorUtils::SetRealDimCnt(transdata_in_desc, transdata_in_dim_cnt);

  transdata_out_desc.SetDataType(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType());
  transdata_out_desc.SetOriginDataType(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginDataType());
  transdata_out_desc.SetFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat());
  transdata_out_desc.SetOriginFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginFormat());
  transdata_out_desc.SetShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetShape());
  transdata_out_desc.SetOriginShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape());
  ge::TensorUtils::SetRealDimCnt(transdata_out_desc, transdata_out_dim_cnt);

  if (is_dynamic_flag) {
    std::vector<std::pair<int64_t, int64_t>> out_ori_range_matmul;
    std::vector<std::pair<int64_t, int64_t>> out_nz_range_matmul;
    FUSION_PASS_CHECK(
        matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(out_ori_range_matmul) == ge::GRAPH_FAILED,
        OP_LOGE(kFusedOpType.c_str(), "Failed to get output shape range of matmul."), return fe::FAILED);

    pair<int64_t, int64_t> n_range;
    if (n_shape == -1) {
      n_range = std::make_pair((out_ori_range_matmul[1].first + kBlockSize - 1) / kBlockSize,
                               (out_ori_range_matmul[1].second + kBlockSize - 1) / kBlockSize);
    } else {
      n_range = std::make_pair(n_shape, n_shape);
    }
    pair<int64_t, int64_t> m_range;
    if (m_shape == -1) {
      m_range = std::make_pair((out_ori_range_matmul[0].first + kBlockSize - 1) / kBlockSize,
                               (out_ori_range_matmul[0].second + kBlockSize - 1) / kBlockSize);
    } else {
      m_range = std::make_pair(m_shape, m_shape);
    }

    pair<int64_t, int64_t> block_range = std::make_pair(kBlockSize, kBlockSize);
    out_nz_range_matmul.push_back(n_range);
    out_nz_range_matmul.push_back(m_range);
    out_nz_range_matmul.push_back(block_range);
    out_nz_range_matmul.push_back(block_range);
    transdata_in_desc.SetShapeRange(out_nz_range_matmul);
    transdata_in_desc.SetOriginShapeRange(out_nz_range_matmul);
    transdata_out_desc.SetShapeRange(out_nz_range_matmul);
    transdata_out_desc.SetOriginShapeRange(out_nz_range_matmul);
    FUSION_PASS_CHECK(
        matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(out_nz_range_matmul) == ge::GRAPH_FAILED,
        OP_LOGE(kFusedOpType.c_str(), "Failed to set output shape range of matmul."), return fe::FAILED);
  }

  new_desc_ptr->AddInputDesc(transdata_in_desc);
  new_desc_ptr->AddOutputDesc(transdata_out_desc);
  int fe_imply_type = 6;
  ge::AttrUtils::SetInt(new_desc_ptr, "_fe_imply_type", fe_imply_type);
  ge::AttrUtils::SetInt(new_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 0);
  transdata_node = graph_ptr->AddNode(new_desc_ptr);

  matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_FRACTAL_NZ);
  matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape(matmul_output_nz_shape));
  matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
  matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(ge::GeShape(matmul_output_nz_shape));

  ge::AttrUtils::SetBool(transdata_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetBool(transdata_node->GetOpDesc(), "_node_need_compile", true);
  ge::AttrUtils::SetBool(matmul_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  Operator op_transdata = ge::OpDescUtils::CreateOperatorFromNode(transdata_node);

  op_transdata.SetAttr("src_format", "FRACTAL_NZ");
  op_transdata.SetAttr("dst_format", "ND");

  (void)ge::AttrUtils::SetBool(graph_ptr, NEED_RE_PRECOMPILE, true);
  return SUCCESS;
}

Status MatmulAtomicAddUbFusion::GenerateCastNode(ge::NodePtr &matmul_node, ge::NodePtr &cast_node) {
  auto base_name = matmul_node->GetName();
  auto matmul_out_shape_size = GetCurrNodeOutputDesc(matmul_node, 0)->GetShape().GetDims().size();
  auto op_name = base_name + "_cast";
  ge::OpDescPtr new_desc_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc_ptr = std::make_shared<ge::OpDesc>(op_name, kOpTypeCast)),
                          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "create %s_desc_ptr failed.", op_name.c_str());
                          new_desc_ptr = nullptr;
                          return FAILED);
  ge::ComputeGraphPtr graph_ptr = matmul_node->GetOwnerComputeGraph();

  ge::GeTensorDesc cast_in_desc;
  ge::GeTensorDesc cast_out_desc;

  cast_in_desc.SetDataType(ge::DT_FLOAT);
  cast_in_desc.SetOriginDataType(ge::DT_FLOAT);
  cast_in_desc.SetFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat());
  cast_in_desc.SetOriginFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginFormat());
  cast_in_desc.SetShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetShape());
  cast_in_desc.SetOriginShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape());
  ge::TensorUtils::SetRealDimCnt(cast_in_desc, matmul_out_shape_size);

  cast_out_desc.SetDataType(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetDataType());
  cast_out_desc.SetOriginDataType(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginDataType());
  cast_out_desc.SetFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetFormat());
  cast_out_desc.SetOriginFormat(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginFormat());
  cast_out_desc.SetShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetShape());
  cast_out_desc.SetOriginShape(matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape());
  ge::TensorUtils::SetRealDimCnt(cast_out_desc, matmul_out_shape_size);

  if (is_dynamic_flag) {
    std::vector<std::pair<int64_t, int64_t>> out_ori_range_matmul;
    FUSION_PASS_CHECK(
        matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(out_ori_range_matmul) == ge::GRAPH_FAILED,
        OP_LOGE(kFusedOpType.c_str(), "Failed to get output shape range of matmul."), return fe::FAILED);
    cast_in_desc.SetShapeRange(out_ori_range_matmul);
    cast_in_desc.SetOriginShapeRange(out_ori_range_matmul);
    cast_out_desc.SetShapeRange(out_ori_range_matmul);
    cast_out_desc.SetOriginShapeRange(out_ori_range_matmul);
  }

  new_desc_ptr->AddInputDesc(cast_in_desc);
  new_desc_ptr->AddOutputDesc(cast_out_desc);

  ge::AttrUtils::SetInt(new_desc_ptr, ge::ATTR_NAME_CONTINUOUS_INPUT, 0);
  ge::AttrUtils::SetInt(new_desc_ptr, ge::ATTR_NAME_CONTINUOUS_OUTPUT, 0);
  ge::AttrUtils::SetInt(new_desc_ptr, ge::CAST_ATTR_TRUNCATE, 0);
  ge::AttrUtils::SetInt(new_desc_ptr, ge::CAST_ATTR_DSTT, 1);
  ge::AttrUtils::SetInt(new_desc_ptr, ge::CAST_ATTR_SRCT, 0);
  ge::AttrUtils::SetInt(new_desc_ptr, "dst_type", 1);

  std::map<string, uint32_t> name_index_map = {{"x", 0}};
  new_desc_ptr->UpdateInputName(name_index_map);
  new_desc_ptr->SetInputName({base_name});

  cast_node = graph_ptr->AddNode(new_desc_ptr);
  ge::AttrUtils::SetBool(cast_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetBool(cast_node->GetOpDesc(), "_node_need_compile", true);
  matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
  matmul_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(ge::DT_FLOAT);
  ge::AttrUtils::SetBool(matmul_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  (void)ge::AttrUtils::SetBool(graph_ptr, NEED_RE_PRECOMPILE, true);

  return SUCCESS;
}

Status MatmulAtomicAddUbFusion::AddCustomNode(int cur_add_node_type, ge::NodePtr &matmul_node,
                                              vector<ge::NodePtr> &fusion_nodes) {
  auto base_name = matmul_node->GetName();
  auto next_node_name = base_name;
  ge::NodePtr next_node;
  FUSION_PASS_CHECK(matmul_node->GetOutDataAnchor(0) == nullptr,
                    OP_LOGW(kFusedOpType.c_str(), "Node %s get output failed.", matmul_node->GetName().c_str()),
                    return SUCCESS);
  FUSION_PASS_CHECK(matmul_node->GetAllOutDataAnchors().size() > 1,
                    OP_LOGW(kFusedOpType.c_str(), "Node %s have multi out data anchor, not handle this now.",
                            matmul_node->GetName().c_str()),
                    return SUCCESS);

  if (cur_add_node_type == ATOMIC_ADD_NEED_TRANSDATA) {
    next_node_name = next_node_name + "_Transdata";
    FUSION_PASS_CHECK(
      SUCCESS != GenerateTransDataNode(matmul_node, next_node),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add transdata node for %s fail.", matmul_node->GetName().c_str()),
      return FAILED);
  } else if (cur_add_node_type == ATOMIC_ADD_NEED_CAST) {
    next_node_name = next_node_name + "_Cast";
    FUSION_PASS_CHECK(
      SUCCESS != GenerateCastNode(matmul_node, next_node),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add cast node for %s fail.", matmul_node->GetName().c_str()),
      return FAILED);
  }

  if (matmul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (ge::InDataAnchorPtr &in_anchor_ptr : matmul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(matmul_node->GetOutDataAnchor(0), in_anchor_ptr),
          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Remove edge between node %s and %s fail.",
                                matmul_node->GetName().c_str(), in_anchor_ptr->GetOwnerNode()->GetName().c_str()),
          return FAILED);
      OP_LOGD(kFusedOpType.c_str(), "Remove edge between node %s and %s.", matmul_node->GetName().c_str(),
              in_anchor_ptr->GetOwnerNode()->GetName().c_str());
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(next_node->GetOutDataAnchor(0), in_anchor_ptr),
          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge between node %s and %s fail.",
                                next_node->GetName().c_str(), in_anchor_ptr->GetOwnerNode()->GetName().c_str()),
          return FAILED);
      OP_LOGD(kFusedOpType.c_str(), "Add edge between node %s and %s.", next_node->GetName().c_str(),
              in_anchor_ptr->GetOwnerNode()->GetName().c_str());
    }
  } else {
    OP_LOGD(kFusedOpType.c_str(), "matmul's out anchor is not connected to other node!");
  }

  FUSION_PASS_CHECK(SUCCESS != MatMulLinkControlEdge(matmul_node, next_node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add control edge between node fail."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), next_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge between node %s and %s fail.",
                                          matmul_node->GetName().c_str(), next_node->GetName().c_str()),
                    return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge between node %s and %s.", matmul_node->GetName().c_str(),
          next_node->GetName().c_str());
  OP_LOGD(kFusedOpType.c_str(), "Add the node %s success!", next_node_name.c_str());
  return SUCCESS;
}

Status MatmulAtomicAddUbFusion::MatMulLinkControlEdge(ge::NodePtr &matmul_node, ge::NodePtr &next_node) {
  if (matmul_node->GetOutControlAnchor() != nullptr) {
    if (matmul_node->GetOutControlAnchor()->GetPeerInControlAnchors().size() > 0) {
      for (auto in_control_anchor : matmul_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(matmul_node->GetOutControlAnchor(), in_control_anchor),
          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Remove control edge between node %s and %s fail.",
                                matmul_node->GetName().c_str(), in_control_anchor->GetOwnerNode()->GetName().c_str()),
          return FAILED);
        OP_LOGD(kFusedOpType.c_str(), "Remove control edge between node %s and %s.", matmul_node->GetName().c_str(),
                in_control_anchor->GetOwnerNode()->GetName().c_str());
        FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(next_node->GetOutControlAnchor(), in_control_anchor),
          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add control edge between node %s and %s fail.",
                                next_node->GetName().c_str(), in_control_anchor->GetOwnerNode()->GetName().c_str()),
          return FAILED);
        OP_LOGD(kFusedOpType.c_str(), "Add control edge between node %s and %s.", next_node->GetName().c_str(),
                in_control_anchor->GetOwnerNode()->GetName().c_str());
      }
    } else {
      OP_LOGD(kFusedOpType.c_str(), "The node of %s's out control anchor is not connected to other node!",
              matmul_node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status MatmulAtomicAddUbFusion::GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  OP_LOGD(kFusedOpType.c_str(), "Begin to do MatmulAtomicAddUbFusion!");
  FUSION_PASS_CHECK(platform_info.ai_core_spec.cube_vector_split == 1,
                    OP_LOGI(kFusedOpType.c_str(), "MatMul is not support atomic_add k when cube vector split"),
                    return SUCCESS);
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != fe::SUCCESS,
      OP_LOGE(kFusedOpType.c_str(), "Get platform_info failed."), return FAILED);
  core_num = platform_info.soc_info.ai_core_cnt;
  l2_size = platform_info.soc_info.l2_size;

  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(katternMatmul, mapping);
  if (matmul_nodes.empty()) {
    OP_LOGD(kFusedOpType.c_str(), "matmul node not matched");
    return SUCCESS;
  }

  if (matmul_nodes[0] == nullptr) {
    OP_LOGD(kFusedOpType.c_str(), "matmul node invalid.");
    return SUCCESS;
  }
  auto matmul_node = matmul_nodes[0];
  bool ret = IsDynamic(matmul_node, is_dynamic_flag, is_no_range);
  if (ret != SUCCESS) {
    OP_LOGW(kFusedOpType.c_str(), "Get dynamic flag or no range flag fail, end.", matmul_node->GetName().c_str());
    return SUCCESS;
  }
  OP_LOGD(kFusedOpType.c_str(), "Current matmul node name is %s.", matmul_node->GetName().c_str());

  auto atomic_add_type = AtomicAddType(matmul_node);
  OP_LOGD(kFusedOpType.c_str(), "In matmul atomic add fusion pass, mode is %d", atomic_add_type);
  if (atomic_add_type != ATOMIC_ADD_DISABLE) {
    fusion_nodes.push_back(matmul_node);
    if ((atomic_add_type & ATOMIC_ADD_NEED_TRANSDATA) == ATOMIC_ADD_NEED_TRANSDATA) {
      // add transdata node
      OP_LOGD(kFusedOpType.c_str(), "Current matmul node %s will connect with Transdata.",
              matmul_node->GetName().c_str());
      FUSION_PASS_CHECK(AddCustomNode(ATOMIC_ADD_NEED_TRANSDATA, matmul_node, fusion_nodes) != SUCCESS,
                        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add Transdata node fail."), return FAILED);
    }
    if ((atomic_add_type & ATOMIC_ADD_NEED_CAST) == ATOMIC_ADD_NEED_CAST) {
      // add cast node
      OP_LOGD(kFusedOpType.c_str(), "Current matmul node %s will connect with Cast.", matmul_node->GetName().c_str());
      FUSION_PASS_CHECK(AddCustomNode(ATOMIC_ADD_NEED_CAST, matmul_node, fusion_nodes) != SUCCESS,
                        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add Cast node fail."), return FAILED);
    }
  } else {
    OP_LOGD(kFusedOpType.c_str(), "Current matmul node %s is disabled atomic add.", matmul_node->GetName().c_str());
  }

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulAtomicAddUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, MatmulAtomicAddUbFusion);
}  // namespace fe