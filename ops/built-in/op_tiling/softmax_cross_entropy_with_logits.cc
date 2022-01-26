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

// compile info
struct CompileInfo {
  int32_t ub_size;
  int32_t core_num;
};

// tiling info
struct TilingInfo {
  int32_t key;
  int32_t block_nparts;
  int32_t block_axis;
  int32_t ub_factor;
  int32_t ub_axis;
};

const std::unordered_map<ge::DataType, int32_t> kDtypeSizeMap{{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}};
struct opInfo {
  std::vector<bool> flag_info;
  int32_t ub_size;
  int32_t core_num;
};

bool SoftmaxCrossEntropyWithLogitsParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                            opInfo& compile_value) {
  using namespace nlohmann;
  OP_TILING_CHECK(
      !GetCompileValue(compile_info, "flag_info", compile_value.flag_info),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get flag_info error"),
      return false);
  auto common_info_iter = compile_info.find("common_info");
  OP_TILING_CHECK(
      common_info_iter == compile_info.end(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get common_info error"),
      return false);
  const auto& sub_common_info = compile_info["common_info"];
  OP_TILING_CHECK(!GetCompileValue(sub_common_info, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get core_num error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(sub_common_info, "ub_size", compile_value.ub_size),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SoftmaxCrossEntropyWithLogitsParseFunc get ub_size error"),
                  return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}
int64_t GetDtypeSize(const ge::DataType& dtype) {
  // element nums in one block
  int32_t dtype_size = kDtypeSizeMap.at(dtype);
  return dtype_size;
}

bool WriteTilingData(const std::string& op_type, const CompileInfo& compile_info,
                     TilingInfo& tiling_info, utils::OpRunInfo& run_info, std::vector<int64_t> input_features_shape,
                     std::vector<int64_t> input_labels_shape) {
  GELOGD("op [%s] tiling ub_size:%lld", op_type.c_str(), compile_info.ub_size);
  GELOGD("op [%s] tiling core_num:%lld", op_type.c_str(), compile_info.core_num);

  GELOGD("op [%s] tiling key:%lld", op_type.c_str(), tiling_info.key);
  GELOGD("op [%s] tiling ub_factor:%lld", op_type.c_str(), tiling_info.ub_factor);
  GELOGD("op [%s] tiling ub_axis:%lld", op_type.c_str(), tiling_info.ub_axis);
  GELOGD("op [%s] tiling block_nparts:%lld", op_type.c_str(), tiling_info.block_nparts);
  GELOGD("op [%s] tiling block_axis:%lld", op_type.c_str(), tiling_info.block_axis);

  run_info.SetBlockDim(tiling_info.block_nparts);

  int32_t tiling_key = static_cast<int32_t>(tiling_info.key);
  run_info.SetTilingKey(tiling_key);

  run_info.AddTilingData(static_cast<int32_t>(input_features_shape[0]));
  run_info.AddTilingData(static_cast<int32_t>(input_labels_shape[0]));
  run_info.AddTilingData(static_cast<int32_t>(input_features_shape[1]));
  run_info.AddTilingData(static_cast<int32_t>(input_labels_shape[1]));

  run_info.AddTilingData(static_cast<int32_t>(tiling_info.block_nparts));
  run_info.AddTilingData(static_cast<int32_t>(tiling_info.ub_factor));

  return true;
}

bool CompletedShapes(std::array<std::array<int64_t, MAX_DIM_LEN>, INPUT_NUM>& input_shapes,
                     std::array<int64_t, MAX_DIM_LEN>& output_shape, const size_t& dim_len, const std::string& op_type,
                     const ge::Operator& op_paras) {
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  for (size_t i = 0; i < INPUT_NUM; i++) {
    OP_TILING_CHECK(operator_info->MutableInputDesc(i)->MutableShape().GetDims().empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, " input tensor cannot be empty"), return false);
    // init all dim to 1
    input_shapes[i].fill(1ll);
  }
  output_shape.fill(1ll);
  for (size_t i = 0; i < INPUT_NUM; i++) {
    size_t cur_dim_len = operator_info->MutableInputDesc(i)->MutableShape().GetDimNum();
    size_t start_index = dim_len - cur_dim_len;
    for (size_t j = 0; j < cur_dim_len; j++) {
      input_shapes[i][start_index] = operator_info->MutableInputDesc(i)->MutableShape().GetDim(j);
      start_index++;
    }
  }
  for (size_t i = 0; i < dim_len; i++) {
    int64_t max_output = input_shapes[0][i];
    output_shape[i] = input_shapes[0][i];
    for (size_t j = 1; j < INPUT_NUM; j++) {
      bool verify_broadcast = input_shapes[j][i] != 1 && (input_shapes[j][i] != max_output && max_output != 1);
      if (input_shapes[j][i] > max_output) {
        max_output = input_shapes[j][i];
        output_shape[i] = input_shapes[j][i];
      }
      OP_TILING_CHECK(verify_broadcast,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shapes [%s] cannot broadcast to shape [%s]",
                                                      std::to_string(input_shapes[j][i]).c_str(),
                                                      std::to_string(max_output).c_str()),
                      return false);
    }
  }
  return true;
}

void CalNdKey(TilingInfo& tiling_info, const std::array<std::array<int64_t, MAX_DIM_LEN>, INPUT_NUM>& input_shapes) {
  constexpr int key_10 = 10;
  constexpr int key_1 = 1;
  constexpr int key_4 = 4;
  constexpr int key_2 = 2;
  constexpr int key_8 = 8;
  constexpr int key_9 = 9;
  constexpr int key_6 = 6;
  int64_t key = 0;
  std::array<int64_t, MAX_DIM_LEN> features_shape = input_shapes[0];
  std::array<int64_t, MAX_DIM_LEN> labels_shape = input_shapes[1];
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
  tiling_info.key = key;
}

bool DoNdTiling(const std::string& op_type, CompileInfo& compile_info, TilingInfo& tiling_info,
                std::array<std::array<int64_t, MAX_DIM_LEN>, INPUT_NUM>& input_shapes, ge::DataType& out_type,
                const std::array<int64_t, MAX_DIM_LEN>& output_shape) {
  GELOGI("op [%s]: DoTiling func running", op_type.c_str());
  int32_t n_h_w = max(input_shapes[0][0], input_shapes[1][0]);
  int32_t c_size = max(input_shapes[0][1], input_shapes[1][1]);
  int32_t ub_axis = 0;
  int32_t ub_factor = output_shape[0];
  int32_t block_axis = 0;
  int32_t block_nparts = 1;
  int32_t dtype_size = GetDtypeSize(out_type);
  OP_TILING_CHECK(dtype_size == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dtype_size cannot be zero"), return false);
  int32_t bound_size = compile_info.ub_size / dtype_size / MAX_COEXIST_NUM;
  int32_t num_per_block = BTYPE_PER_BLOCK / dtype_size;
  int32_t c_size_align = (c_size + num_per_block - 1) / num_per_block * num_per_block;
  if (c_size > bound_size) {
    VECTOR_INNER_ERR_REPORT_TILIING("SoftmaxCrossEntropyWithLogitsTiling", "not supported shape");
    return false;
  } else if ((c_size * num_per_block < bound_size) && (n_h_w >= num_per_block)) {
    // for open multi-core
    block_nparts = (n_h_w * dtype_size) >= (static_cast<int32_t>(compile_info.core_num) * BTYPE_PER_BLOCK)
                       ? compile_info.core_num
                       : n_h_w * dtype_size / BTYPE_PER_BLOCK;
    int32_t block_tiling_inner_loop = n_h_w / block_nparts;
    ub_factor = min(bound_size / c_size_align, block_tiling_inner_loop);
  } else {
    // for cannot open multi-core scene
    block_nparts = 1;
    ub_factor = min(bound_size / c_size_align, n_h_w);
  }

  tiling_info.block_nparts = block_nparts;
  tiling_info.block_axis = block_axis;
  tiling_info.ub_factor = ub_factor;
  tiling_info.ub_axis = ub_axis;

  CalNdKey(tiling_info, input_shapes);

  return true;
}

bool GetPreCompileParams(const std::string& op_type, const opInfo& op_info, CompileInfo& compile_info) {
  // core num
  compile_info.core_num = op_info.core_num;
  // ub size
  compile_info.ub_size = op_info.ub_size;

  GELOGD("op [%s] GetPreCompileParams, core_num[%d].", op_type.c_str(), compile_info.core_num);
  GELOGD("op [%s] GetPreCompileParams, ub_size[%d].", op_type.c_str(), compile_info.ub_size);
  return true;
}

bool SoftmaxCrossEntropyWithLogitsTiling(const std::string& op_type, const ge::Operator& op_paras,
                                         const opInfo& op_info, utils::OpRunInfo& run_info) {
  GELOGI("op [%s]: tiling running", op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input_features = operator_info->MutableInputDesc(0)->MutableShape();
  auto input_labels = operator_info->MutableInputDesc(1)->MutableShape();
  std::vector<int64_t> input_features_shape = input_features.GetDims();
  std::vector<int64_t> input_labels_shape = input_labels.GetDims();
  const ge::DataType input0_type = op_paras.GetInputDesc(0).GetDataType();
  ge::DataType out_type = op_paras.GetOutputDesc(0).GetDataType();

  size_t dim_len =
      input_features_shape.size() > input_labels_shape.size() ? input_features_shape.size() : input_labels_shape.size();
  std::array<std::array<int64_t, MAX_DIM_LEN>, INPUT_NUM> input_shapes{};

  if (kDtypeSizeMap.count(input0_type) == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SoftmaxCrossEntropyWithLogitsTiling", "Invalid input dtype");
    return false;
  }
  std::array<int64_t, MAX_DIM_LEN> output_shape;
  bool ret = CompletedShapes(input_shapes, output_shape, dim_len, op_type, op_paras);
  // not use pure template
  CompileInfo compile_info = {1, 1};
  TilingInfo tiling_info = {1, 1, 1, 1, 1};

  ret = ret && GetPreCompileParams(op_type, op_info, compile_info);

  int64_t key = 110000000;
  tiling_info.key = key;

  if (dim_len == ND_SHAPE_LEN) {
    ret = ret && DoNdTiling(op_type, compile_info, tiling_info, input_shapes, out_type, output_shape);
  }
  ret = ret && WriteTilingData(op_type, compile_info, tiling_info, run_info, input_features_shape,
                               input_labels_shape);
  return ret;
}
REGISTER_OP_TILING_V3_CUSTOM(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsTiling,
                             SoftmaxCrossEntropyWithLogitsParseFunc, opInfo);
}  // namespace optiling
