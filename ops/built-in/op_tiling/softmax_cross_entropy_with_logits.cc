/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file softmax_cross_entropy_with_logits.cpp
 * \brief
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../op_proto/util/op_common_util.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"

namespace optiling {
static const size_t MAX_DIM_LEN = 8;
static const size_t INPUT_NUM = 2;
static const size_t BTYPE_PER_BLOCK = 32;
static const size_t MAX_COEXIST_NUM = 10;
static const size_t ND_SHAPE_LEN = 2;
static const size_t MAX_DTYPE_SIZE = 4;
static const size_t FP16_BLOCK_SIZE = 16;
static const size_t MAX_WORKSPACE_NUMS = 3;

// tiling info
struct TilingInfo {
  int32_t key;
  int32_t block_nparts;
  int32_t block_axis;
  int32_t ub_factor;
  int32_t ub_axis;
};

const std::unordered_map<ge::DataType, int32_t> kDtypeSizeMap{{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}};
struct ScewlOpInfo {
  int32_t ub_size;
  int32_t temp_ub_size;
  string max_type;
  string min_type;
  int32_t core_num;
  int32_t align_base_key;
  bool is_const;
  // norm vars
  std::unordered_map<int32_t, std::vector<string>> normal_vars;
  std::vector<int32_t> dimension_align_ward;
  int32_t max_dim_len;
  string reduce_mean_cof_dtype;
  ge::DataType reduce_mean_cof_ge_dtype;
  std::vector<int32_t> ori_reduce_axis;
};

bool SoftmaxCrossEntropyWithLogitsParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                            ScewlOpInfo& op_info) {
  using namespace nlohmann;

  if (compile_info.count("reduce_mean_cof_dtype") > 0) {
    GetCompileValue(compile_info, "reduce_mean_cof_dtype", op_info.reduce_mean_cof_dtype);
    op_info.reduce_mean_cof_ge_dtype = GetGeTypeFromStr(op_info.reduce_mean_cof_dtype);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_reduce_axis", op_info.ori_reduce_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc get _ori_reduce_axis error"),
                    return false);
  }
  const auto& local_normal_vars = compile_info.at("_normal_vars").get<std::unordered_map<string,
                                                                                         std::vector<string>>>();
  for (const auto& single_item : local_normal_vars) {
    op_info.normal_vars[std::stoi(single_item.first)] = single_item.second;
  }
  OP_TILING_CHECK(!GetCompileValue(compile_info, "_dimension_align_ward", op_info.dimension_align_ward),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get dimension_align_ward error"),
                  return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_max_type", op_info.max_type),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _max_type error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_min_type", op_info.min_type),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _min_type error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_core_num", op_info.core_num),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _core_num error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_available_size", op_info.temp_ub_size),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _available_size error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_align_base_key", op_info.align_base_key),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _align_base_key error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_is_const", op_info.is_const),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _is_const error"),
      return false);
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "_max_dim_len", op_info.max_dim_len),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get _max_dim_len error"),
      return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

void CalNdKeyTemplate(const std::string& op_type, TilingInfo& tiling_info,
                      const std::vector<std::vector<int64_t>>& input_shapes, int32_t bound_size, bool is_workspace) {
  constexpr int key_10 = 10;
  constexpr int key_1 = 1;
  constexpr int key_4 = 4;
  constexpr int key_2 = 2;
  constexpr int key_8 = 8;
  constexpr int key_9 = 9;
  constexpr int key_6 = 6;
  int64_t reduce_mode = 100;
  int64_t key = 0;
  std::vector<int64_t> features_shape = input_shapes[0];
  std::vector<int64_t> labels_shape = input_shapes[1];
  int64_t dim00 = features_shape[0];
  int64_t dim01 = features_shape[1];
  int64_t dim10 = labels_shape[0];
  int64_t dim11 = labels_shape[1];
  bool broadcast_dim00_to_dim10 = (dim00 == 1 && dim10 > 1);
  bool broadcast_dim10_to_dim00 = (dim10 == 1 && dim00 > 1);
  bool broadcast_dim01_to_dim11 = (dim01 == 1 && dim11 > 1);
  bool broadcast_dim11_to_dim01 = (dim11 == 1 && dim01 > 1);
  if (dim00 == dim10 && dim01 == dim11) {
    // copy: [a, b] [a, b] choose no_stride=0, but [a, 1] [a, 1] choose no_stride=1
    key = key_10;
  } else if (broadcast_dim00_to_dim10 && dim01 == dim11) {
    // 1 common, large common
    key = key_1;
  } else if (broadcast_dim10_to_dim00 && dim01 == dim11) {
    // large common, 1 common
    key = key_4;
  } else if (broadcast_dim01_to_dim11 && dim00 == dim10) {
    // common 1, common large
    key = key_2;
  } else if (broadcast_dim11_to_dim01 && dim00 == dim10) {
    // common large, common 1
    key = key_8;
  } else if (broadcast_dim00_to_dim10 && broadcast_dim11_to_dim01) {
    // 1 large, large, 1
    key = key_9;
  } else if (broadcast_dim01_to_dim11 && broadcast_dim10_to_dim00) {
    // large 1, 1, large
    key = key_6;
  }
  if (max(dim11, dim01) > bound_size || is_workspace) {
    key += reduce_mode;
  }
  tiling_info.key = key;
  GELOGD("op [%s] key[%d].", op_type.c_str(), key);
}


bool DoNdTilingWorkspace(const std::string& op_type, const ScewlOpInfo& op_info,
                         TilingInfo& tiling_info, const ge::DataType& out_type,
                         const std::vector<std::vector<int64_t>>& input_shapes,
                         const std::vector<int64_t>& output_shape) {
  GELOGI("op [%s]: DoNdTilingWorkspace func running", op_type.c_str());
  int32_t n_h_w = max(input_shapes[0][0], input_shapes[1][0]);
  int32_t c_size = max(input_shapes[0][1], input_shapes[1][1]);
  int32_t ub_axis = 0;
  int32_t ub_factor = output_shape[0];
  int32_t block_axis = 0;
  int32_t block_nparts = 1;
  int32_t dtype_size = kDtypeSizeMap.at(out_type);
  GELOGD("op [%s] dtype_size = %d.", op_type.c_str(), dtype_size);
  OP_TILING_CHECK(dtype_size == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dtype_size cannot be zero"), return false);
  int32_t bound_size = op_info.temp_ub_size;
  int32_t num_per_block = BTYPE_PER_BLOCK / dtype_size;
  int32_t c_size_align = (c_size + FP16_BLOCK_SIZE - 1) / FP16_BLOCK_SIZE * FP16_BLOCK_SIZE;
  GELOGD("op [%s] bound_size = %d.", op_type.c_str(), bound_size);
  if (c_size_align > bound_size) {
    block_nparts = (n_h_w * dtype_size) >= (op_info.core_num * static_cast<int32_t>(BTYPE_PER_BLOCK))
                       ? op_info.core_num
                       : max(n_h_w * dtype_size / static_cast<int32_t>(BTYPE_PER_BLOCK), 1);
    ub_factor = (c_size - bound_size) >= num_per_block ? bound_size : bound_size - num_per_block;
  } else {
    // for open multi-core
    block_nparts = (n_h_w * dtype_size) >= (op_info.core_num * static_cast<int32_t>(BTYPE_PER_BLOCK))
                       ? op_info.core_num
                       : max(n_h_w * dtype_size / static_cast<int32_t>(BTYPE_PER_BLOCK), 1);
    ub_factor = c_size;
  }
  int32_t block_inner = (n_h_w + block_nparts - 1) / block_nparts;
  if (block_inner < num_per_block) {
    block_nparts = 1;
  }

  block_nparts = max(block_nparts, 1);
  GELOGD("op [%s] block_nparts = %d.", op_type.c_str(), block_nparts);
  GELOGD("op [%s] ub_factor = %d.", op_type.c_str(), ub_factor);

  tiling_info.block_nparts = block_nparts;
  tiling_info.block_axis = block_axis;
  tiling_info.ub_factor = ub_factor;
  tiling_info.ub_axis = ub_axis;
  bool is_workspace = true;
  CalNdKeyTemplate(op_type, tiling_info, input_shapes, bound_size, is_workspace);

  return true;
}

bool GetInput(std::vector<std::vector<int64_t>>& input_shapes, std::vector<int64_t>& max_input_shape,
              const int32_t input_num, const int32_t dim_len, const ge::Operator& op_paras,
              const ScewlOpInfo& op_info) {
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  std::vector<int32_t> dimension_align_ward = op_info.dimension_align_ward;
  for (int32_t i = 0; i < input_num; i++) {
    int32_t cur_dim_len = operator_info->MutableInputDesc(i)->MutableShape().GetDimNum();
    int32_t start_index = 0;
    if (dimension_align_ward[i] != 1) {
      start_index = dim_len - cur_dim_len;
    }
    for (int32_t j = 0; j < cur_dim_len; j++) {
      input_shapes[i][start_index] = operator_info->MutableInputDesc(i)->MutableShape().GetDim(j);
      start_index++;
    }
  }
  for (int32_t i = 0; i < dim_len; i++) {
    for (int32_t j = 0; j < input_num; j++) {
      max_input_shape[i] = max(max_input_shape[i], input_shapes[j][i]);
    }
  }
  return true;
}

bool WriteTilingDataTemplate(const std::string& op_type, const ScewlOpInfo& op_info, TilingInfo& tiling_info,
                             utils::OpRunInfo& run_info, std::vector<std::vector<int64_t>>& input_shapes,
                             const ge::Operator& op_paras, std::vector<int64_t>& max_input_shape,
                             const ge::DataType& out_type) {
  GELOGD("op [%s] tiling ub_size:%lld", op_type.c_str(), op_info.ub_size);
  GELOGD("op [%s] tiling core_num:%lld", op_type.c_str(), op_info.core_num);
  GELOGD("op [%s] tiling key:%lld", op_type.c_str(), tiling_info.key);
  GELOGD("op [%s] tiling block_nparts:%lld", op_type.c_str(), tiling_info.block_nparts);
  GELOGD("op [%s] tiling ub_factor:%lld", op_type.c_str(), tiling_info.ub_factor);
  GELOGD("op [%s] tiling block_axis:%lld", op_type.c_str(), tiling_info.block_axis);
  GELOGD("op [%s] tiling ub_axis:%lld", op_type.c_str(), tiling_info.ub_axis);

  int32_t dtype_size = 4;  // workspace dtype is float32
  int64_t workspace_size = dtype_size;
  for (const auto& dim_value : max_input_shape) {
    workspace_size *= dim_value;
  }
  std::array<int64_t, MAX_WORKSPACE_NUMS> workspaces{workspace_size, workspace_size, workspace_size};
  for (int64_t ws : workspaces) {
    run_info.AddWorkspace(ws);
  }

  run_info.SetBlockDim(tiling_info.block_nparts);
  int32_t tiling_key = static_cast<int32_t>(tiling_info.key);
  run_info.SetTilingKey(tiling_key);

  // template define normal vars
  std::unordered_map<int32_t, std::vector<string>> normal_vars = op_info.normal_vars;
  std::vector<string> var_list = normal_vars.at(tiling_key);
  std::vector<int32_t> var_value;
  string dim_prefix = "_dim_";
  for (const auto& var : var_list) {
    if (var == "_ub_factor") {
      var_value.push_back(static_cast<int32_t>(tiling_info.ub_factor));
    } else if (var == "_block_nparts") {
      var_value.push_back(static_cast<int32_t>(tiling_info.block_nparts));
    } else if (var.rfind(dim_prefix, 0) == 0) {
      int32_t row = var[5] - '0';
      int32_t col = var[7] - '0';
      var_value.push_back(static_cast<int32_t>(input_shapes[row][col]));
    } else {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get var value error. Error message: %s", var.c_str());
      return false;
    }
  }

  for (const auto& varv : var_value) {
    run_info.AddTilingData(static_cast<int32_t>(varv));
  }

  if (op_type == "LayerNorm") {
    OP_TILING_CHECK(op_info.reduce_mean_cof_dtype.empty(),
                    OP_LOGI(op_type.c_str(), "need not do AddReduceMeanCof, return true"), return true);
    const auto& input_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(0)->GetShape();
    OP_TILING_CHECK(!AddReducMeanCof(input_shape, op_info.reduce_mean_cof_ge_dtype, op_info.ori_reduce_axis, run_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormTiling, do AddReduceMeanCof failed"),
                    return false);
  }
  return true;
}

bool TemplateTiling(const std::string& op_type, const ge::Operator& op_paras, const ScewlOpInfo& op_info,
                    utils::OpRunInfo& run_info) {
  const int32_t input_num = op_paras.GetInputsSize();
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  const ge::DataType input0_type = op_paras.GetInputDesc(0).GetDataType();
  const ge::DataType out_type = op_paras.GetOutputDesc(0).GetDataType();

  int32_t dim_len = op_info.max_dim_len;
  std::vector<std::vector<int64_t>> input_shapes(input_num, std::vector<int64_t>(dim_len, 1));
  std::vector<int64_t> max_input_shape(dim_len, 1);
  bool GetInputRet = GetInput(input_shapes, max_input_shape, input_num, dim_len, op_paras, op_info);
  OP_TILING_CHECK(!GetInputRet, OP_LOGI(op_type.c_str(), "getinput func failed, return false"), return false);
  int32_t n_dim = 1;
  // merge axis:AR, default last axis is reduce axis
  for (size_t i = 0; i < max_input_shape.size() - 1; i++) {
    n_dim *= max_input_shape[i];
  }
  // reduce dims
  int32_t d_dim = max_input_shape[1];
  int32_t ub_axis = 0;
  int32_t ub_factor = 1;
  int32_t block_axis = 0;
  int32_t ub_size = op_info.temp_ub_size;
  int32_t core_num = op_info.core_num;
  int32_t block_nparts = core_num;
  int32_t per_block_num = BTYPE_PER_BLOCK / kDtypeSizeMap.at(input0_type);
  bool is_align_case = (d_dim % per_block_num == 0);
  int32_t aligned_d_dim = (d_dim + per_block_num - 1) / per_block_num * per_block_num;

  // workspace case use old strategy
  if (aligned_d_dim > ub_size) {
    OP_LOGI(op_type.c_str(), "template tiling strategy failed, back old tiling strategy, return false");
    TilingInfo tiling_info_workspace;
    bool ret = DoNdTilingWorkspace(op_type, op_info, tiling_info_workspace, out_type, input_shapes, max_input_shape);
    OP_TILING_CHECK(!ret, OP_LOGI(op_type.c_str(), "DoNdTilingWorkspace func failed, return false"),
                    return false);
    ret = ret && WriteTilingDataTemplate(op_type, op_info, tiling_info_workspace, run_info, input_shapes, op_paras,
                                         max_input_shape, out_type);
    OP_TILING_CHECK(!ret, OP_LOGI(op_type.c_str(), "WriteTilingDataTemplate func failed, return false"),
                    return false);
    OP_LOGI(op_type.c_str(), "template tiling strategy success !!!!");
    return ret;
  }

  ub_factor = ub_size / aligned_d_dim;
  // if ub_factor larger than n_dim update ub_factor
  ub_factor = (ub_factor > n_dim) ? n_dim : ub_factor;
  // n_dim is divided with ub_factor no remainderd, and ub_factor is larger than per_block_num
  int32_t old_ub_factor = ub_factor;
  while ((n_dim % old_ub_factor != 0) && (old_ub_factor > per_block_num)) {
    old_ub_factor--;
  }
  ub_factor = (n_dim % old_ub_factor == 0) ? old_ub_factor : ub_factor;
  int32_t ub_outer = (n_dim + ub_factor - 1) / ub_factor;

  if (ub_outer < core_num) {
    // Calculate the optimal block nparts
    for (int32_t i = ub_factor; i >= 1; i--) {
      if ((ub_factor % i == 0) && (ub_factor / i >= per_block_num) && (ub_outer * i <= core_num)) {
        ub_factor = ub_factor / i;
        ub_outer = ub_outer * i;
        break;
      }
    }
    block_nparts = ub_outer;
  }

  if (ub_factor < per_block_num) {
    block_nparts = 1;
  }

  TilingInfo tiling_info;
  bool is_workspace = false;
  CalNdKeyTemplate(op_type, tiling_info, input_shapes, ub_size, is_workspace);
  int32_t key = is_align_case ? tiling_info.key : tiling_info.key + op_info.align_base_key;
  tiling_info.key = key;
  tiling_info.block_nparts = block_nparts;
  tiling_info.block_axis = block_axis;
  tiling_info.ub_factor = ub_factor;
  tiling_info.ub_axis = ub_axis;
  TilingInfo tiling_info_workspace;
  tiling_info_workspace.block_nparts = 1;
  // 160 for cut reduce can bind at least 10 thread, while not cut can only bind 1 thread
  if (block_nparts == 1 && n_dim > 160) {
    bool ret = DoNdTilingWorkspace(op_type, op_info, tiling_info_workspace, out_type, input_shapes, max_input_shape);
    OP_TILING_CHECK(!ret, OP_LOGI(op_type.c_str(), "DoNdTilingWorkspace func failed, return false"),
                    return false);
  }
  TilingInfo tiling_info_final;
  // if cut reduce workspace strategy can bind less than 10 thread, choose not cut strategy
  if (tiling_info_workspace.block_nparts < 10) {
    tiling_info_final.key = tiling_info.key;
    tiling_info_final.block_nparts = tiling_info.block_nparts;
    tiling_info_final.block_axis = tiling_info.block_axis;
    tiling_info_final.ub_factor = tiling_info.ub_factor;
    tiling_info_final.ub_axis = tiling_info.ub_axis;
  } else {
    tiling_info_final.key = tiling_info_workspace.key;
    tiling_info_final.block_nparts = tiling_info_workspace.block_nparts;
    tiling_info_final.block_axis = tiling_info_workspace.block_axis;
    tiling_info_final.ub_factor = tiling_info_workspace.ub_factor;
    tiling_info_final.ub_axis = tiling_info_workspace.ub_axis;
  }

  bool writeRet = WriteTilingDataTemplate(op_type, op_info, tiling_info_final, run_info, input_shapes, op_paras,
                                          max_input_shape, out_type);
  OP_TILING_CHECK(!writeRet, OP_LOGI(op_type.c_str(), "WriteTilingDataTemplate func failed, return false"),
                  return false);
  OP_LOGI(op_type.c_str(), "template tiling strategy success !!!!");
  return true;
}

bool SoftmaxCrossEntropyWithLogitsTiling(const std::string& op_type, const ge::Operator& op_paras,
                                         const ScewlOpInfo& op_info, utils::OpRunInfo& run_info) {
  GELOGI("op [%s]: tiling running", op_type.c_str());
  bool ret = TemplateTiling(op_type, op_paras, op_info, run_info);
  return ret;
}
REGISTER_OP_TILING_V3_CUSTOM(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsTiling,
                             SoftmaxCrossEntropyWithLogitsParseFunc, ScewlOpInfo);
}  // namespace optiling
