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

/*!
 * \file softmax_cross_entropy_with_logits.cpp
 * \brief
 */
#include <algorithm>
#include <unordered_map>

#include "graph/debug/ge_log.h"
#include "error_log.h"
#include "vector_tiling.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace optiling {

static const size_t MAX_DIM_LEN = 8;
static const size_t input_num = 2;

// compile info
struct CompileInfo {
  int32_t ub_size;
  int32_t core_num;
  int32_t max_dtype;
};

// tiling info
struct TilingInfo {
  int32_t key;
  int32_t block_nparts;
  int32_t block_axis;
  int32_t ub_factor;
  int32_t ub_axis;
};

const std::unordered_map<std::string, int32_t> kDtypeSizeMap{{"float16", 2}, {"float32", 4}};

int64_t GetDtypeSize(std::string& dtype) {
  // element nums in one block
  int32_t dtype_size = kDtypeSizeMap.at(dtype);
  return dtype_size;
}

void DimValPut(const int32_t dim_val, const int32_t range_l, const int32_t range_r, OpRunInfo& run_info) {
  if (range_l < range_r) {
    ByteBufferPut(run_info.tiling_data, dim_val);
  }
}

bool WriteTilingData(const std::string& op_type,
                     const nlohmann::json& op_info, const CompileInfo& compile_info,
                     TilingInfo& tiling_info, OpRunInfo& run_info,
                     std::vector<int64_t> input_features_shape,
                     std::vector<int64_t> input_labels_shape,
                     std::array<int64_t, MAX_DIM_LEN>& output_shape) {
  GELOGD("op [%s] tiling ub_size:%lld", op_type.c_str(), compile_info.ub_size);
  GELOGD("op [%s] tiling core_num:%lld", op_type.c_str(), compile_info.core_num);
  GELOGD("op [%s] tiling max_dtype:%lld", op_type.c_str(), compile_info.max_dtype);

  GELOGD("op [%s] tiling key:%lld", op_type.c_str(), tiling_info.key);
  GELOGD("op [%s] tiling ub_factor:%lld", op_type.c_str(), tiling_info.ub_factor);
  GELOGD("op [%s] tiling ub_axis:%lld", op_type.c_str(), tiling_info.ub_axis);
  GELOGD("op [%s] tiling block_nparts:%lld", op_type.c_str(), tiling_info.block_nparts);
  GELOGD("op [%s] tiling block_axis:%lld", op_type.c_str(), tiling_info.block_axis);

  run_info.block_dim = tiling_info.block_nparts;

  const int32_t& dim_var_0_0 = op_info["ori_shape"]["features_shape0"];
  const int32_t& dim_var_0_1 = op_info["ori_shape"]["features_shape1"];
  const int32_t& dim_var_1_0 = op_info["ori_shape"]["labels_shape0"];
  const int32_t& dim_var_1_1 = op_info["ori_shape"]["labels_shape1"];

  const int32_t& range_0_0_l = op_info["range"]["features_range0_l"];
  const int32_t& range_0_0_r = op_info["range"]["features_range0_r"];
  const int32_t& range_0_1_l = op_info["range"]["features_range1_l"];
  const int32_t& range_0_1_r = op_info["range"]["features_range1_r"];
  const int32_t& range_1_0_l = op_info["range"]["labels_range0_l"];
  const int32_t& range_1_0_r = op_info["range"]["labels_range0_r"];
  const int32_t& range_1_1_l = op_info["range"]["labels_range1_l"];
  const int32_t& range_1_1_r = op_info["range"]["labels_range1_r"];

  bool case_no_unknown_1 = dim_var_0_0 > 0 && dim_var_0_1 > 0 && dim_var_1_0 > 0 && dim_var_1_1 > 0;

  bool case_one_unknown_1 = dim_var_0_0 < 0 && dim_var_0_1 > 0 && dim_var_1_0 > 0 && dim_var_1_1 > 0;
  bool case_one_unknown_2 = dim_var_0_0 > 0 && dim_var_0_1 < 0 && dim_var_1_0 > 0 && dim_var_1_1 > 0;
  bool case_one_unknown_3 = dim_var_0_0 > 0 && dim_var_0_1 > 0 && dim_var_1_0 < 0 && dim_var_1_1 > 0;
  bool case_one_unknown_4 = dim_var_0_0 > 0 && dim_var_0_1 > 0 && dim_var_1_0 > 0 && dim_var_1_1 < 0;

  bool case_two_unknown_1 = dim_var_0_0 < 0 && dim_var_0_1 < 0 && dim_var_1_0 > 0 && dim_var_1_1 > 0;
  bool case_two_unknown_2 = dim_var_0_0 < 0 && dim_var_0_1 > 0 && dim_var_1_0 < 0 && dim_var_1_1 > 0;
  bool case_two_unknown_3 = dim_var_0_0 < 0 && dim_var_0_1 > 0 && dim_var_1_0 > 0 && dim_var_1_1 < 0;
  bool case_two_unknown_4 = dim_var_0_0 > 0 && dim_var_0_1 < 0 && dim_var_1_0 < 0 && dim_var_1_1 > 0;
  bool case_two_unknown_5 = dim_var_0_0 > 0 && dim_var_0_1 < 0 && dim_var_1_0 > 0 && dim_var_1_1 < 0;
  bool case_two_unknown_6 = dim_var_0_0 > 0 && dim_var_0_1 > 0 && dim_var_1_0 < 0 && dim_var_1_1 < 0;

  bool case_three_unknown_1 = dim_var_0_0 < 0 && dim_var_0_1 < 0 && dim_var_1_0 < 0 && dim_var_1_1 > 0;
  bool case_three_unknown_2 = dim_var_0_0 < 0 && dim_var_0_1 < 0 && dim_var_1_0 > 0 && dim_var_1_1 < 0;
  bool case_three_unknown_3 = dim_var_0_0 < 0 && dim_var_0_1 > 0 && dim_var_1_0 < 0 && dim_var_1_1 < 0;
  bool case_three_unknown_4 = dim_var_0_0 > 0 && dim_var_0_1 < 0 && dim_var_1_0 < 0 && dim_var_1_1 < 0;

  bool case_four_unknown_1 = dim_var_0_0 < 0 && dim_var_0_1 < 0 && dim_var_1_0 < 0 && dim_var_1_1 < 0;

  int32_t tiling_key = static_cast<int32_t>(tiling_info.key);
  run_info.tiling_key = tiling_key;

  if (case_no_unknown_1) {
    GELOGI("op [%s]: case_no_unknown_1 running", op_type.c_str());
  } else if (case_one_unknown_1) {
    GELOGI("op [%s]: case_one_unknown_1 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
  } else if (case_one_unknown_2) {
    GELOGI("op [%s]: case_one_unknown_2 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
  } else if (case_one_unknown_3) {
    GELOGI("op [%s]: case_one_unknown_3 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
  } else if (case_one_unknown_4) {
    GELOGI("op [%s]: case_one_unknown_4 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
  } else if (case_two_unknown_1) {
    GELOGI("op [%s]: case_two_unknown_1 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
    DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
  } else if (case_two_unknown_2) {
    GELOGI("op [%s]: case_two_unknown_2 running", op_type.c_str());
    if (range_0_0_l > 1 && range_1_0_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
    }
  } else if (case_two_unknown_3) {
    GELOGI("op [%s]: case_two_unknown_3 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
    DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
  } else if (case_two_unknown_4) {
    GELOGI("op [%s]: case_two_unknown_4 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
    DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
  } else if (case_two_unknown_5) {
    GELOGI("op [%s]: case_two_unknown_5 running", op_type.c_str());
    if (range_0_1_l > 1 && range_1_1_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    }
  } else if (case_two_unknown_6) {
    GELOGI("op [%s]: case_two_unknown_6 running", op_type.c_str());
    DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
    DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
  } else if (case_three_unknown_1) {
    GELOGI("op [%s]: case_three_unknown_1 running", op_type.c_str());
    if (range_0_0_l > 1 && range_1_0_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    }
  } else if (case_three_unknown_2) {
    GELOGI("op [%s]: case_three_unknown_2 running", op_type.c_str());
    if (range_0_1_l > 1 && range_1_1_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    }
  } else if (case_three_unknown_3) {
    GELOGI("op [%s]: case_three_unknown_3 running", op_type.c_str());
    if (range_0_0_l > 1 && range_1_0_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    }
  } else if (case_three_unknown_4) {
    GELOGI("op [%s]: case_three_unknown_4 running", op_type.c_str());
    if (range_0_1_l > 1 && range_1_1_l > 1) {
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    }
  } else if (case_four_unknown_1) {
    GELOGI("op [%s]: case_four_unknown_1 running", op_type.c_str());
    if (range_0_0_l > 1 && range_0_1_l > 1 && range_1_0_l > 1 && range_1_1_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    } else if (range_0_1_l > 1 && range_1_1_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
    } else if (range_0_0_l > 1 && range_1_0_l > 1) {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    } else {
      DimValPut(static_cast<int32_t>(input_features_shape[0]), range_0_0_l, range_0_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[0]), range_1_0_l, range_1_0_r, run_info);
      DimValPut(static_cast<int32_t>(input_features_shape[1]), range_0_1_l, range_0_1_r, run_info);
      DimValPut(static_cast<int32_t>(input_labels_shape[1]), range_1_1_l, range_1_1_r, run_info);
    }
  }

  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(tiling_info.block_nparts));
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(tiling_info.ub_factor));

  return true;
}


bool CompletedShapes(std::array<std::array<int64_t, MAX_DIM_LEN>, 2>& input_shapes,
                     std::array<int64_t, MAX_DIM_LEN>& output_shape,
                     const size_t input_num, size_t& dim_len,
                     const std::string& op_type, const TeOpParas& op_paras) {
  for (size_t i = 0; i < input_num; i++) {
    OP_TILING_CHECK(op_paras.inputs[i].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, " input tensor cannot be empty"), return false);
    // init all dim to 1
    input_shapes[i].fill(1ll);
  }
  output_shape.fill(1ll);
  for (size_t i = 0; i < input_num; i++) {
    size_t cur_dim_len = op_paras.inputs[i].tensor[0].shape.size();
    size_t start_index = dim_len - cur_dim_len;
    for (size_t j = 0; j < cur_dim_len; j++) {
      input_shapes[i][start_index] = op_paras.inputs[i].tensor[0].shape[j];
      start_index++;
    }
  }
  for (size_t i = 0; i < dim_len; i++) {
    int64_t max_output = input_shapes[0][i];
    output_shape[i] = input_shapes[0][i];
    for (size_t j = 1; j < input_num; j++) {
      bool verify_broadcast = input_shapes[j][i] != 1 &&
          (input_shapes[j][i] != max_output && max_output != 1);
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


int32_t GetMaxUbCount(std::string& out_type, CompileInfo& compile_info, int32_t shape_len) {
  int32_t total_size = compile_info.ub_size;
  int32_t dtype_size = out_type == "float16" ? 2 : 4;
  total_size = total_size / dtype_size;

  int32_t total_width = 0;
  if (out_type == "float32") {
      total_width = 6;
  } else {
      if (shape_len == 4) {
          total_width = 10;
      } else {
          total_width = 12;
      }
  }
  int32_t align_to = 128;

  int32_t max_bound = total_width * align_to;
  int32_t max_ub_count = total_size / max_bound * align_to;

  return max_ub_count;
}


int32_t GetNpartFactor(int32_t n_h_w, int32_t dtype_size, int32_t block_dim) {
  int32_t nparts_factor = block_dim;
  if (n_h_w * dtype_size >= block_dim * 32) {
      nparts_factor = block_dim;
  } else {
      nparts_factor = n_h_w * dtype_size / 32;
  }
  // softmax_cross_logits_nd is a special batch size for nd format
  int32_t softmax_cross_logits_nd = 2105352;
  // softmax_cross_logits_nhw is a special batch size for nhwc format
  int32_t softmax_cross_logits_nhw = 11842605;
  if (block_dim == 30 && nparts_factor != 0) {
      if (n_h_w == softmax_cross_logits_nd) {
          while (n_h_w % nparts_factor != 0) {
              nparts_factor--;
          }
      }
  }
  if (block_dim == 32 && nparts_factor != 0) {
      if (n_h_w == softmax_cross_logits_nhw) {
          while (n_h_w % nparts_factor != 0) {
              nparts_factor--;
          }
      }
  }
  return nparts_factor;
}


void GetUbTiling2D(int32_t shape_nhw, int32_t shape_c, int32_t& nparts_factor, int32_t block_tiling_inner_loop,
                   int32_t min_num_size_one_core, int32_t max_ub_count, int32_t& split_size) {
  int32_t temp_size = shape_c;
  int32_t bound_size = max_ub_count;
  int32_t threshold_size = 128;
  int32_t one_block_size = 32;
  int32_t byte_size_fp32 = 4;

  if (shape_c > threshold_size && shape_c * byte_size_fp32 % one_block_size != 0) {
      if (shape_nhw / nparts_factor >= min_num_size_one_core) {
          split_size = 1;
          return;
      }
      split_size = 1;
      nparts_factor = 1;
      return;
  }
  nparts_factor = nparts_factor <= 1 ? 1 : nparts_factor;
  split_size = 1;
  for (int i = block_tiling_inner_loop; i > 0; i--) {
      if (temp_size * i <= bound_size) {
          split_size = i;
          while (block_tiling_inner_loop % split_size != 0) {
              split_size--;
          }
          break;
      }
  }
}

void CalNdKey(TilingInfo& tiling_info, bool is_special_pattern,
              std::array<std::array<int64_t, MAX_DIM_LEN>, 2>& input_shapes) {
  int64_t key = tiling_info.key;
  if (is_special_pattern) {
      key += 2000000;
      std::array<int64_t, MAX_DIM_LEN> feature_shape = input_shapes[0];
      std::array<int64_t, MAX_DIM_LEN> lable_shape = input_shapes[1];
      for (int i = 0; i < 2; i++) {
           if (i == 0) {
              if (feature_shape[i] == lable_shape[i]) {
                  key += 100000;
              } else if (feature_shape[i] != lable_shape[i] &&
                         (feature_shape[i] == 1 || lable_shape[i] == 1)) {
                  key += 200000;
              }
          } else if (i == 1) {
              // for common_reduce
              if (feature_shape[i] == lable_shape[i] && feature_shape[i] != 1) {
                  key += 30000;
              // for broadcast_reduce
              } else if (feature_shape[i] != lable_shape[i] &&
                         (feature_shape[i] == 1 || lable_shape[i] == 1)) {
                  key += 40000;
              }
          }
      }
  } else {
      key = 0;
  }
  tiling_info.key = key;
}

bool DoNdTiling(const std::string& op_type, const nlohmann::json& op_info,
                CompileInfo& compile_info, TilingInfo& tiling_info,
                std::array<std::array<int64_t, MAX_DIM_LEN>, 2>& input_shapes,
                std::string& out_type,
                std::array<int64_t, MAX_DIM_LEN>& output_shape) {
  GELOGI("op [%s]: DoTiling func running", op_type.c_str());
  const int64_t multi_core_threshold = 1024;
  bool need_multi_core = true;
  int32_t n_h_w = input_shapes[0][0];
  int32_t c_size = input_shapes[0][1];
  int32_t threshold_size = 512;
  int32_t block_split_inner_size = n_h_w;
  int32_t ub_axis = 0;
  int32_t ub_factor = 0;
  int32_t block_axis = 0;
  int32_t block_nparts = 0;
  int32_t dtype_size = GetDtypeSize(out_type);
  if (n_h_w < multi_core_threshold) {
    need_multi_core = false;
  }

  if (need_multi_core) {
    int32_t shape_len = 2;
    int32_t max_ub_count = GetMaxUbCount(out_type, compile_info, shape_len);
    int32_t nparts_factor = GetNpartFactor(n_h_w, dtype_size, compile_info.core_num);
    nparts_factor = nparts_factor <= 1 ? 1 : nparts_factor;

    int32_t split_factor = 1;
    int32_t min_num_size_one_core = out_type == "float16" ? 16 : 8;
    if (n_h_w > c_size) {
        threshold_size = 512;
        block_split_inner_size = n_h_w / nparts_factor;
        while (block_split_inner_size > 0 && block_split_inner_size < min_num_size_one_core) {
            nparts_factor--;
            block_split_inner_size = n_h_w / nparts_factor;
        }
        nparts_factor = nparts_factor <= 1 ? 1 : nparts_factor;
        if (block_split_inner_size < min_num_size_one_core) {
            nparts_factor = 1;
            block_split_inner_size = n_h_w / nparts_factor;
        } else {
            while (block_split_inner_size >= min_num_size_one_core &&
              block_split_inner_size * c_size * dtype_size < threshold_size &&
              nparts_factor > 1) {
                nparts_factor--;
                if (nparts_factor > 0) {
                    block_split_inner_size = n_h_w / nparts_factor;
                }
              }
        }
        nparts_factor = nparts_factor <= 1 ? 1 : nparts_factor;
    } else {
        block_split_inner_size = n_h_w / nparts_factor;
        threshold_size = 8*1024;
        if (block_split_inner_size * c_size * dtype_size < threshold_size) {
            nparts_factor = 1;
            block_split_inner_size = n_h_w / nparts_factor;
        }
    }
    nparts_factor = nparts_factor <= 1 ? 1 : nparts_factor;
    GetUbTiling2D(n_h_w, c_size, nparts_factor, block_split_inner_size, min_num_size_one_core, max_ub_count, split_factor);
  }
  ub_axis = 0;
  ub_factor = output_shape[0];
  block_axis = 0;
  block_nparts = 1;
  int32_t bound_size = compile_info.ub_size / dtype_size / 10;

  if (c_size > bound_size) {
      VECTOR_INNER_ERR_REPORT_TILIING("SoftmaxCrossEntropyWithLogitsTiling", "not supported shape");
      return false;
  } else if ((c_size * (32 / dtype_size) < bound_size) &&
             (n_h_w >= (32 / dtype_size))) {
      // for open multi-core
      block_nparts = (n_h_w * dtype_size) >= (compile_info.core_num * 32) ?
                     compile_info.core_num : n_h_w * dtype_size / 32;
      int32_t block_tiling_inner_loop = n_h_w / block_nparts;
      ub_factor = min(bound_size / c_size, block_tiling_inner_loop);
  } else {
      // for cannot open multi-core scene
      block_nparts = 1;
      ub_factor = bound_size / c_size;
  }

  tiling_info.block_nparts = block_nparts;
  tiling_info.block_axis = block_axis;
  tiling_info.ub_factor = ub_factor;
  tiling_info.ub_axis = ub_axis;

  const std::vector<bool>& flag_info = op_info["flag_info"];
  bool is_special_pattern = flag_info[3];
  CalNdKey(tiling_info, is_special_pattern, input_shapes);

  return true;
}


bool GetPreCompileParams(const std::string& op_type, const nlohmann::json& op_info,
                         CompileInfo& compile_info) {
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info is null");
    return false;
  }
  const auto& common_info = op_info["common_info"];

  // core num
  if (common_info.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is null");
    return false;
  }
  compile_info.core_num = common_info["core_num"].get<std::int32_t>();

  // ub size
  if (common_info.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_size is null");
    return false;
  }
  compile_info.ub_size = common_info["ub_size"].get<std::int32_t>();

  compile_info.max_dtype = 4;

  GELOGD("op [%s] GetPreCompileParams, core_num[%d].", op_type.c_str(), compile_info.core_num);
  GELOGD("op [%s] GetPreCompileParams, ub_size[%d].", op_type.c_str(), compile_info.ub_size);
  return true;
}

bool SoftmaxCrossEntropyWithLogitsTiling(const std::string& op_type, const TeOpParas& op_paras,
                                         const nlohmann::json& op_info, OpRunInfo& run_info) {
  GELOGI("op [%s]: tiling running", op_type.c_str());

  auto input_features = op_paras.inputs[0].tensor[0];
  auto input_labels = op_paras.inputs[1].tensor[0];

  std::string out_type = op_paras.outputs[0].tensor[0].dtype;

  std::vector<int64_t> input_features_shape = input_features.shape;
  std::vector<int64_t> input_labels_shape = input_labels.shape;

  size_t dim_len = input_features_shape.size() > input_labels_shape.size() ?
                   input_features_shape.size() : input_labels_shape.size();
  std::array<std::array<int64_t, MAX_DIM_LEN>, 2> input_shapes{};

  OP_TILING_CHECK((op_info.find("flag_info") == op_info.end()),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [flag_info]"), return false);
  if (kDtypeSizeMap.count(input_features.dtype) == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SoftmaxCrossEntropyWithLogitsTiling", "Invalid input dtype");
    return false;
  }
  const std::vector<bool>& flag_info = op_info["flag_info"];
  std::array<int64_t, MAX_DIM_LEN> output_shape;
  bool ret = CompletedShapes(input_shapes, output_shape, input_num, dim_len, op_type, op_paras);
  // not use pure template
  CompileInfo compile_info;
  TilingInfo tiling_info;

  ret = ret && GetPreCompileParams(op_type, op_info, compile_info);

  int64_t key = 110000000;
  tiling_info.key = key;
  if (dim_len == 2) {
    ret = ret && DoNdTiling(op_type, op_info, compile_info, tiling_info, input_shapes, out_type, output_shape);
  }
  ret = ret && WriteTilingData(op_type, op_info, compile_info, tiling_info, run_info,
                               input_features_shape, input_labels_shape, output_shape);
  return ret;
}
REGISTER_OP_TILING_FUNC_BUFFERED(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsTiling);
}  // namespace optiling