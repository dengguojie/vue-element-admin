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
 * \file reduce_tiling.cpp
 * \brief tiling function of op
 */
#include <cmath>
#include <string>

#include <nlohmann/json.hpp>

#include "register/op_tiling.h"

#include "vector_tiling.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "op_log.h"

namespace optiling {

const int32_t SMALL_SHAPE_THRESHOLD = 1024;
const int32_t REDUCE_MEAN_COF_FP32 = 1;
const int32_t REDUCE_MEAN_COF_FP16 = 2;
const int32_t FUSED_NON_REDUCE_AXIS = 0;
const int32_t FUSED_REDUCE_AXIS = 1;
const int32_t NORMAL_TILLING = 0;
const int32_t DEFAULT_TILLING = 1;
const int32_t ATOMIC_TILLING = 2;

const uint8_t LAST_AXIS_INIT = 0;
const uint8_t LAST_AXIS_REDUCE = 1;
const uint8_t LAST_AXIS_NONREDUCE = 2;

bool IsInVector(std::vector<int32_t>& input_vector, int32_t value) {
  for (uint32_t i = 0; i < input_vector.size(); i++) {
    if (input_vector[i] == value) {
      return true;
    }
  }
  return false;
}

int64_t GetShapeMul(std::vector<int64_t>& shape, int32_t axis_index) {
  int64_t result = 1;
  for (uint32_t i = axis_index + 1; i < shape.size(); i++) {
    if (shape[i] != 0) {
      result = result * shape[i];
    }
  }
  return result;
}

int64_t GetAlignShapeMul(std::vector<int64_t>& output_shape, int32_t block_tilling_axis, bool is_last_axis_reduce,
                         int32_t block_size) {
  int64_t result = 1;
  for (uint32_t i = block_tilling_axis + 1; i < output_shape.size(); i++) {
    if (output_shape[i] == 0) {
      continue;
    }
    if (i == output_shape.size() - 1 && is_last_axis_reduce == false) {
      result = result * ((output_shape[i] + block_size - 1) / block_size * block_size);
    } else {
      result = result * output_shape[i];
    }
  }
  return result;
}

int64_t GetReorderInputShapeMul(std::vector<int64_t>& reorder_input_shape, int32_t axis_index,
                                std::vector<int32_t>& fused_block_tilling_axis, int32_t block_tilling_axis_in_reorder,
                                int64_t block_tilling_factor, int32_t block_size) {
  int64_t result = 1;

  for (uint32_t i = axis_index + 1; i < reorder_input_shape.size(); i++) {
    if (IsInVector(fused_block_tilling_axis, i)) {
      continue;
    }

    if (i == (uint32_t)block_tilling_axis_in_reorder) {
      if (i == reorder_input_shape.size() - 1) {
        result = result * ((block_tilling_factor + block_size - 1) / block_size * block_size);
      } else {
        result = result * block_tilling_factor;
      }
    } else {
      result = result * reorder_input_shape[i];
    }
  }
  return result;
}

int32_t GetRealBlockTillingAxis(std::vector<int64_t>& shape, int32_t idx) {
  int32_t zero_cnt = 0;
  for (int32_t i = 0; i < idx; i++) {
    if (shape[i] == 0) {
      zero_cnt += 1;
    }
  }
  return idx - zero_cnt;
}

int32_t GetBlockDim(std::vector<int64_t>& output_shape, int32_t block_tilling_axis, int64_t block_tilling_factor) {
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= block_tilling_axis; i++) {
    if (output_shape[i] != 0) {
      if (i == block_tilling_axis) {
        block_dim = (int32_t)((output_shape[i] + block_tilling_factor - 1) / block_tilling_factor) * block_dim;
      } else {
        block_dim = (int32_t)output_shape[i] * block_dim;
      }
    }
  }

  return block_dim;
}

int32_t GetAtomicBlockDim(std::vector<int64_t>& input_shape, std::vector<int32_t>& reduce_axis,
                          int32_t block_tilling_axis, int64_t block_tilling_factor) {
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= block_tilling_axis; i++) {
    if (IsInVector(reduce_axis, i)) {
      if (i == block_tilling_axis) {
        block_dim = (int32_t)((input_shape[i] + block_tilling_factor - 1) / block_tilling_factor) * block_dim;
        break;
      } else {
        block_dim = (int32_t)input_shape[i] * block_dim;
      }
    }
  }

  return block_dim;
}

int32_t GetBlockSize(std::string dtypeUB) {
  int32_t block_size = 0;

  if (dtypeUB == "float32" || dtypeUB == "int32" || dtypeUB == "uint32") {
    block_size = 8;
  } else if (dtypeUB == "float16" || dtypeUB == "int16" || dtypeUB == "uint16") {
    block_size = 16;
  } else if (dtypeUB == "int8" || dtypeUB == "uint8") {
    block_size = 32;
  }

  return block_size;
}

int32_t CalcPattern(std::vector<int64_t>& input_shape, std::vector<int32_t>& reduce_axis) {
  int32_t pattern = 0;
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      pattern += 2 * pow(2, input_shape.size() - 1 - i);
    } else {
      pattern += pow(2, input_shape.size() - 1 - i);
    }
  }

  return pattern;
}

bool GetBlockTillingInfo(std::vector<int64_t>& output_shape, int64_t max_ub_count, int32_t core_num,
                         bool is_last_axis_reduce, int32_t block_size, int32_t& block_tilling_axis, int64_t& block_tilling_factor) {
  if (block_size == 0) {
    GELOGD("block size is zero.");
    return false;
  }

  int32_t left_block_dim = 1;
  for (uint32_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] == 0 || output_shape[i] == 1) {
      continue;
    }

    int64_t right_inner_ub_count = GetAlignShapeMul(output_shape, i, is_last_axis_reduce, block_size);
    if (right_inner_ub_count > max_ub_count) {
      left_block_dim = (int32_t)left_block_dim * output_shape[i];
      continue;
    }

    int64_t max_block_tilling_factor = max_ub_count / right_inner_ub_count;
    int64_t right_total_num = GetShapeMul(output_shape, i);

    if (right_total_num <= block_size && left_block_dim * output_shape[i] < core_num) {
      if (left_block_dim > 1) {
        int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
        if (cur_block_factor <= output_shape[i]) {
          for (; cur_block_factor <= output_shape[i]; cur_block_factor++) {
            int64_t tail_block_factor = output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
            int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
            if (tail_block_tilling_inner_ddr_count >= block_size) {
              block_tilling_axis = i;
              block_tilling_factor = cur_block_factor;
              return true;
            }
          }
        } else {
          block_tilling_axis = i;
          block_tilling_factor = output_shape[i];
          return true;
        }
      } else {
        int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
        block_tilling_axis = i;
        block_tilling_factor = cur_block_factor;
        return true;
      }
    } else if (right_total_num <= block_size || left_block_dim * output_shape[i] >= core_num) {
      for (int32_t tilling_core_num = core_num; tilling_core_num <= left_block_dim * output_shape[i];
           tilling_core_num += core_num) {
        if (left_block_dim > tilling_core_num) {
          continue;
        }
        int64_t cur_block_dim = tilling_core_num / left_block_dim;
        int64_t cur_block_factor = (output_shape[i] + cur_block_dim - 1) / cur_block_dim;
        if (cur_block_factor > max_block_tilling_factor) {
          continue;
        }

        if (left_block_dim > 1) {
          int64_t tail_block_factor = output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
          int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
          if (tail_block_tilling_inner_ddr_count < block_size) {
            for (cur_block_dim = cur_block_dim - 1; cur_block_dim >= 1; cur_block_dim--) {
              cur_block_factor = (output_shape[i] + cur_block_dim - 1) / cur_block_dim;
              tail_block_factor = output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
              tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
              if (tail_block_tilling_inner_ddr_count >= block_size) {
                block_tilling_axis = i;
                block_tilling_factor = cur_block_factor;
                return true;
              }
            }
          } else {
            block_tilling_axis = i;
            block_tilling_factor = cur_block_factor;
            return true;
          }

        } else {
          int64_t block_tilling_inner_ddr_count = cur_block_factor * right_total_num;
          if (block_tilling_inner_ddr_count < block_size) {
            cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
          }

          block_tilling_axis = i;
          block_tilling_factor = cur_block_factor;
          return true;
        }
      }
    }

    left_block_dim = (int32_t)left_block_dim * output_shape[i];
  }

  GE_LOGE("GetBlockTillingInfo error.");
  return false;
}

void ProcessReorderAxis(bool is_last_axis_reduce, int32_t block_tilling_axis, std::vector<int64_t>& input_shape,
                        std::vector<int32_t>& reduce_axis, std::vector<int64_t>& reorder_input_shape,
                        std::map<int32_t, int32_t>& reorder_pos_ref_input,
                        std::vector<int32_t>& fused_block_tilling_axis, int32_t fused_type) {
  int32_t reorder_pos = 0;
  if (is_last_axis_reduce) {
    for (int32_t i = 0; i < (int32_t)input_shape.size(); i++) {
      if (!IsInVector(reduce_axis, i)) {
        if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tilling_axis) {
          fused_block_tilling_axis.push_back(reorder_pos);
        }

        reorder_input_shape.push_back(input_shape[i]);
        reorder_pos_ref_input[reorder_pos] = i;
        reorder_pos += 1;
      }
    }

    for (int32_t i = 0; i < (int32_t)input_shape.size(); i++) {
      if (IsInVector(reduce_axis, i)) {
        if (fused_type == FUSED_REDUCE_AXIS && i < block_tilling_axis) {
          fused_block_tilling_axis.push_back(reorder_pos);
        }

        reorder_input_shape.push_back(input_shape[i]);
        reorder_pos_ref_input[reorder_pos] = i;
        reorder_pos += 1;
      }
    }
  } else {
    int32_t last_reduce_axis_idx = 0;
    for (uint32_t i = 0; i < reduce_axis.size(); i++) {
      if (reduce_axis[i] > last_reduce_axis_idx) {
        last_reduce_axis_idx = reduce_axis[i];
      }
    }

    // order non reduce axis
    for (int32_t i = 0; i < last_reduce_axis_idx + 1; i++) {
      if (!IsInVector(reduce_axis, i)) {
        if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tilling_axis) {
          fused_block_tilling_axis.push_back(reorder_pos);
        }

        reorder_input_shape.push_back(input_shape[i]);
        reorder_pos_ref_input[reorder_pos] = i;
        reorder_pos += 1;
      }
    }

    // order reduce axis
    for (int32_t i = 0; i < last_reduce_axis_idx + 1; i++) {
      if (IsInVector(reduce_axis, i)) {
        if (fused_type == FUSED_REDUCE_AXIS && i < block_tilling_axis) {
          fused_block_tilling_axis.push_back(reorder_pos);
        }

        reorder_input_shape.push_back(input_shape[i]);
        reorder_pos_ref_input[reorder_pos] = i;
        reorder_pos += 1;
      }
    }
    // order last non axis, maybe several axis
    for (int32_t i = last_reduce_axis_idx + 1; i < (int32_t)input_shape.size(); i++) {
      if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tilling_axis) {
        fused_block_tilling_axis.push_back(reorder_pos);
      }

      reorder_input_shape.push_back(input_shape[i]);
      reorder_pos_ref_input[reorder_pos] = i;
      reorder_pos += 1;
    }
  }

  return;
}

bool GetUbTillingInfo(const std::string& opType, std::vector<int64_t>& input_shape,
                      std::vector<int64_t>& reorder_input_shape, std::map<int32_t, int32_t>& reorder_pos_ref_input,
                      std::vector<int32_t>& fused_block_tilling_axis, int64_t max_ub_count, int32_t block_size,
                      int32_t block_tilling_axis, int64_t block_tilling_factor, int32_t& ub_tilling_axis,
                      int64_t& ub_tilling_factor) {
  if (block_size == 0) {
    GE_LOGE("GetUbTillingInfo input paras error.");
    return false;
  }

  int32_t block_tilling_axis_in_reorder = -1;
  for (uint32_t i = 0; i < reorder_pos_ref_input.size(); i++) {
    if (reorder_pos_ref_input[i] == block_tilling_axis) {
      block_tilling_axis_in_reorder = i;
      break;
    }
  }

  int64_t load_mul = 1;
  for (int32_t i = 0; i < (int32_t)reorder_input_shape.size(); i++) {
    if (IsInVector(fused_block_tilling_axis, i)) {
      continue;
    }

    load_mul = GetReorderInputShapeMul(reorder_input_shape, i, fused_block_tilling_axis, block_tilling_axis_in_reorder,
                                       block_tilling_factor, block_size);
    if (load_mul <= max_ub_count) {
      ub_tilling_axis = reorder_pos_ref_input[i];
      ub_tilling_factor = (max_ub_count / load_mul);

      int64_t max_ub_tilling_factor = input_shape[ub_tilling_axis];
      if (i == block_tilling_axis_in_reorder) {
        max_ub_tilling_factor = block_tilling_factor;
      }
      if (ub_tilling_factor > max_ub_tilling_factor) {
        ub_tilling_factor = max_ub_tilling_factor;
      }

      return true;
    }
  }

  return false;
}

void GetNotMulCoreBlockTilling(std::vector<int64_t>& input_shape, std::vector<int32_t>& reduce_axis,
                               int32_t& block_tilling_axis, int64_t& block_tilling_factor) {
  if (input_shape.size() <= 0) {
    return;
  }
  block_tilling_axis = 0;
  block_tilling_factor = input_shape[0];
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (!IsInVector(reduce_axis, i)) {
      block_tilling_axis = i;
      block_tilling_factor = input_shape[i];
      return;
    }
  }
  return;
}

bool SetRunInfo(const std::string& opType, OpRunInfo& run_info, const nlohmann::json& op_info, int32_t block_dim,
                std::vector<int64_t>& input_shape, int32_t first_tilling_axis, int64_t first_tilling_factor,
                int32_t second_tilling_axis, int64_t second_tilling_factor, int32_t set_reduce_mean_cof_flag,
                float reduce_mean_cof, std::string pattern_str, int32_t tilling_type,
                int32_t default_tilling_key = -1) {
  GELOGI("start set run info.");

  run_info.block_dim = block_dim;

  std::unordered_map<std::string, int32_t> var_names;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    std::string name = "dim_" + std::to_string(i) + "_0";
    var_names[name] = input_shape[i];
  }

  if (tilling_type == NORMAL_TILLING) {
    int32_t block_tilling_axis = first_tilling_axis;
    int64_t block_tilling_factor = first_tilling_factor;
    int32_t ub_tilling_axis = second_tilling_axis;
    int64_t ub_tilling_factor = second_tilling_factor;

    const auto& reduce_tilling_key_map = op_info[pattern_str]["reduce_tiling_key_map"];

    // get tilling key
    int32_t tilling_key = -1;
    for (const auto& tilling_key_item : reduce_tilling_key_map.items()) {
      auto tilling_key_item_value = tilling_key_item.value();
      bool is_find_key = tilling_key_item_value.count("block_split_axis") > 0 &&
                         tilling_key_item_value["block_split_axis"] == block_tilling_axis &&
                         tilling_key_item_value.count("ub_split_axis") > 0 &&
                         tilling_key_item_value["ub_split_axis"] == ub_tilling_axis;

      if (is_find_key) {
        tilling_key = std::stoi(tilling_key_item.key());
        break;
      }
    }

    if (tilling_key == -1) {
      GE_LOGE("get tilling key error.");
      return false;
    }

    // put values into OpRunInfo
    ByteBufferPut(run_info.tiling_data, tilling_key);

    // put input shape
    const auto& all_vars = op_info["_vars"][std::to_string(tilling_key)];
    for (const auto& var : all_vars) {
      if (var_names.count(var) > 0) {
        ByteBufferPut(run_info.tiling_data, var_names[var]);
        GELOGD("input shape:%d", var_names[var]);
      }
    }

    if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP32) {
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
    } else if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP16) {
      fe::fp16_t reduce_mean_cof_fp16;
      reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
    }
    ByteBufferPut(run_info.tiling_data, (int32_t)block_tilling_factor);
    ByteBufferPut(run_info.tiling_data, (int32_t)ub_tilling_factor);

    GELOGD("block dim:%d", block_dim);
    GELOGD("tiling key:%d", tilling_key);
    GELOGD("block tilling axis:%d", block_tilling_axis);
    GELOGD("block tilling factor:%d", block_tilling_factor);
    GELOGD("ub tilling axis:%d", ub_tilling_axis);
    GELOGD("ub tilling factor:%d", ub_tilling_factor);

  } else if (tilling_type == DEFAULT_TILLING) {
    int32_t res_ub_tilling_axis = first_tilling_axis;
    int64_t res_ub_tilling_factor = first_tilling_factor;
    int32_t input_ub_tilling_axis = first_tilling_axis;
    int64_t input_ub_tilling_factor = second_tilling_factor;

    if (default_tilling_key == -1) {
      GE_LOGE("get tilling key error.");
      return false;
    }

    // put values into OpRunInfo
    ByteBufferPut(run_info.tiling_data, default_tilling_key);

    // put input shape
    const auto& all_vars = op_info["_vars"][std::to_string(default_tilling_key)];
    for (const auto& var : all_vars) {
      if (var_names.count(var) > 0) {
        ByteBufferPut(run_info.tiling_data, var_names[var]);
        GELOGD("input shape:%d", var_names[var]);
      }
    }

    if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP32) {
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
    } else if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP16) {
      fe::fp16_t reduce_mean_cof_fp16;
      reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
    }
    ByteBufferPut(run_info.tiling_data, (int32_t)res_ub_tilling_factor);
    ByteBufferPut(run_info.tiling_data, (int32_t)input_ub_tilling_factor);

    GELOGD("block dim:1");
    GELOGD("tiling key:%d", default_tilling_key);
    GELOGD("res ub tilling axis:%d", res_ub_tilling_axis);
    GELOGD("res ub tilling factor:%d", res_ub_tilling_factor);
    GELOGD("input ub tilling axis:%d", input_ub_tilling_axis);
    GELOGD("input ub tilling factor:%d", input_ub_tilling_factor);

  } else if (tilling_type == ATOMIC_TILLING) {
    int32_t block_tilling_axis = first_tilling_axis;
    int64_t block_tilling_factor = first_tilling_factor;
    int32_t ub_tilling_axis = second_tilling_axis;
    int64_t ub_tilling_factor = second_tilling_factor;

    const auto& atomic_tilling_key_map = op_info[pattern_str]["atomic_tiling_key_map"];

    // get tilling key
    int32_t tilling_key = -1;
    for (const auto& tilling_key_item : atomic_tilling_key_map.items()) {
      auto tilling_key_item_value = tilling_key_item.value();
      bool is_find_key = tilling_key_item_value.count("block_split_axis") > 0 &&
                         tilling_key_item_value["block_split_axis"] == block_tilling_axis &&
                         tilling_key_item_value.count("ub_split_axis") > 0 &&
                         tilling_key_item_value["ub_split_axis"] == ub_tilling_axis;

      if (is_find_key) {
        tilling_key = std::stoi(tilling_key_item.key());
        break;
      }
    }

    if (tilling_key == -1) {
      GE_LOGE("get tilling key error.");
      return false;
    }

    // put values into OpRunInfo
    ByteBufferPut(run_info.tiling_data, tilling_key);

    // put input shape
    const auto& all_vars = op_info["_vars"][std::to_string(tilling_key)];
    for (const auto& var : all_vars) {
      if (var_names.count(var) > 0) {
        ByteBufferPut(run_info.tiling_data, var_names[var]);
        GELOGD("input shape:%d", var_names[var]);
      }
    }

    if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP32) {
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
    } else if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP16) {
      fe::fp16_t reduce_mean_cof_fp16;
      reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      ByteBufferPut(run_info.tiling_data, (uint16_t)0);
    }
    ByteBufferPut(run_info.tiling_data, (int32_t)block_tilling_factor);
    ByteBufferPut(run_info.tiling_data, (int32_t)ub_tilling_factor);

    GELOGD("block dim:%d", block_dim);
    GELOGD("tiling key:%d", tilling_key);
    GELOGD("block tilling axis:%d", block_tilling_axis);
    GELOGD("block tilling factor:%d", block_tilling_factor);
    GELOGD("ub tilling axis:%d", ub_tilling_axis);
    GELOGD("ub tilling factor:%d", ub_tilling_factor);
  }

  return true;
}

bool ProcessNormalTilling(const std::string& opType, OpRunInfo& run_info, const nlohmann::json& op_info,
                          std::vector<int64_t>& input_shape, std::vector<int64_t>& output_shape,
                          std::vector<int32_t>& reduce_axis, int32_t core_num, int64_t max_ub_count, int32_t block_size,
                          bool is_last_axis_reduce, int64_t total_output_count, bool is_keep_dims,
                          int32_t set_reduce_mean_cof_flag, float reduce_mean_cof, std::string pattern_str) {
  GELOGI("process normal tilling.");

  if (input_shape.size() == 0) {
    GE_LOGE("input shape size error.");
    return false;
  }
  int32_t block_dim = 1;
  int32_t block_tilling_axis = 0;
  int64_t block_tilling_factor = input_shape[0];
  int32_t ub_tilling_axis = 0;
  int64_t ub_tilling_factor = input_shape[0];
  if (input_shape.size() == 1) {
    block_dim = 1;
    ub_tilling_axis = 0;
    ub_tilling_factor = (max_ub_count / block_size) * block_size;
    if (input_shape[0] < ub_tilling_factor) {
      ub_tilling_factor = input_shape[0];
    }
  } else {
    bool is_find_block_tilling =
        GetBlockTillingInfo(output_shape, max_ub_count, core_num, is_last_axis_reduce,
                            block_size, block_tilling_axis, block_tilling_factor);
    if (!is_find_block_tilling) {
      block_dim = 1;
      GetNotMulCoreBlockTilling(input_shape, reduce_axis, block_tilling_axis, block_tilling_factor);
      if (total_output_count > max_ub_count) {
        return false;
      }
    } else {
      block_dim = GetBlockDim(output_shape, block_tilling_axis, block_tilling_factor);
    }

    GELOGD("block tilling finished. block tilling axis: %d, block tilling factor: %d", block_tilling_axis,
           block_tilling_factor);

    // input shape reorder
    std::vector<int64_t> reorder_input_shape;
    std::map<int32_t, int32_t> reorder_pos_ref_input;
    std::vector<int32_t> fused_block_tilling_axis;
    ProcessReorderAxis(is_last_axis_reduce, block_tilling_axis, input_shape, reduce_axis, reorder_input_shape,
                       reorder_pos_ref_input, fused_block_tilling_axis, FUSED_NON_REDUCE_AXIS);

    // align
    if (block_size != 0 && reorder_input_shape.size() > 0 && reorder_input_shape.back() % block_size != 0) {
      reorder_input_shape.back() = (reorder_input_shape.back() + block_size - 1) / block_size * block_size;
    }

    ub_tilling_axis = block_tilling_axis;
    ub_tilling_factor = block_tilling_factor;

    bool is_find_ub_tilling = GetUbTillingInfo(opType, input_shape, reorder_input_shape, reorder_pos_ref_input,
                                               fused_block_tilling_axis, max_ub_count, block_size, block_tilling_axis,
                                               block_tilling_factor, ub_tilling_axis, ub_tilling_factor);
    if (!is_find_ub_tilling) {
      GE_LOGE("Get ub tilling info error.");
      return false;
    }

    if (is_keep_dims == 0) {
      block_tilling_axis = GetRealBlockTillingAxis(output_shape, block_tilling_axis);
    }
  }

  bool ret = SetRunInfo(opType, run_info, op_info, block_dim, input_shape, block_tilling_axis, block_tilling_factor,
                        ub_tilling_axis, ub_tilling_factor, set_reduce_mean_cof_flag, reduce_mean_cof, pattern_str,
                        NORMAL_TILLING, -1);

  return ret;
}

bool ProcessDefaultTilling(const std::string& opType, OpRunInfo& run_info, const nlohmann::json& op_info,
                           std::vector<int64_t>& input_shape, std::vector<int64_t>& output_shape,
                           std::vector<int32_t>& reduce_axis, bool is_last_axis_reduce, int64_t max_ub_count,
                           int32_t block_size, int32_t set_reduce_mean_cof_flag, float reduce_mean_cof,
                           std::string pattern_str) {
  GELOGI("process default tilling.");

  const auto& reduce_tilling_key_map = op_info[pattern_str]["reduce_tiling_key_map"];
  int32_t tilling_key = -1;
  int32_t res_ub_tilling_axis = -1;
  int32_t input_ub_tilling_axis = -1;
  for (const auto& tilling_key_item : reduce_tilling_key_map.items()) {
    auto tilling_key_item_value = tilling_key_item.value();
    bool is_find_key =
        tilling_key_item_value.count("res_ub_split_axis") > 0 && tilling_key_item_value.count("ub_split_axis") > 0;
    if (is_find_key) {
      tilling_key = std::stoi(tilling_key_item.key());
      res_ub_tilling_axis = tilling_key_item_value["res_ub_split_axis"];
      input_ub_tilling_axis = tilling_key_item_value["ub_split_axis"];
      break;
    }
  }

  if (tilling_key == -1) {
    GE_LOGE("cant find tilling key.");
    return false;
  }

  int64_t res_ub_tilling_factor = 0;
  int64_t input_ub_tilling_factor = 0;

  if (block_size == 0) {
    GE_LOGE("block size is zero.");
    return false;
  }

  if (input_ub_tilling_axis >= (int32_t)input_shape.size() || input_ub_tilling_axis < 0) {
    GE_LOGE("input ub tilling axis is error.");
    return false;
  }

  int32_t res_ub_tilling_calc_axis = -1;
  int32_t zero_cnt = 0;
  for (int32_t i = 0; i < (int32_t)output_shape.size(); i++) {
    if (output_shape[i] == 0) {
      zero_cnt += 1;
      continue;
    }

    if (i - zero_cnt == res_ub_tilling_axis) {
      res_ub_tilling_calc_axis = i;
      break;
    }
  }

  if (res_ub_tilling_calc_axis >= (int32_t)output_shape.size() || res_ub_tilling_calc_axis < 0) {
    GE_LOGE("res ub tilling calc axis is error.");
    return false;
  }

  bool is_tilling_same_axis = false;
  if (res_ub_tilling_calc_axis == input_ub_tilling_axis) {
    is_tilling_same_axis = true;
  }

  int64_t res_load_mul = GetAlignShapeMul(output_shape, res_ub_tilling_calc_axis, is_last_axis_reduce, block_size);
  res_ub_tilling_factor = max_ub_count / res_load_mul;
  if (res_ub_tilling_factor > output_shape[res_ub_tilling_calc_axis]) {
    res_ub_tilling_factor = output_shape[res_ub_tilling_calc_axis];
  }

  std::vector<int64_t> reorder_input_shape;
  std::map<int32_t, int32_t> reorder_pos_ref_input;
  std::vector<int32_t> fused_res_ub_tilling_axis;
  ProcessReorderAxis(is_last_axis_reduce, res_ub_tilling_axis, input_shape, reduce_axis, reorder_input_shape,
                     reorder_pos_ref_input, fused_res_ub_tilling_axis, FUSED_NON_REDUCE_AXIS);

  int32_t res_ub_tilling_axis_in_reorder = -1;
  for (uint32_t i = 0; i < reorder_pos_ref_input.size(); i++) {
    if (reorder_pos_ref_input[i] == res_ub_tilling_axis) {
      res_ub_tilling_axis_in_reorder = res_ub_tilling_axis;
      break;
    }
  }

  int64_t input_load_mul = GetReorderInputShapeMul(reorder_input_shape, reorder_pos_ref_input[input_ub_tilling_axis],
                                                   fused_res_ub_tilling_axis, res_ub_tilling_axis_in_reorder,
                                                   res_ub_tilling_factor, block_size);
  input_ub_tilling_factor = max_ub_count / input_load_mul;
  if (is_tilling_same_axis) {
    input_ub_tilling_factor =
        input_ub_tilling_factor > res_ub_tilling_factor ? res_ub_tilling_factor : input_ub_tilling_factor;
  } else {
    input_ub_tilling_factor = input_ub_tilling_factor > input_shape[input_ub_tilling_axis]
                                  ? input_shape[input_ub_tilling_axis]
                                  : input_ub_tilling_factor;
  }

  bool ret = SetRunInfo(opType, run_info, op_info, 1, input_shape, res_ub_tilling_axis, res_ub_tilling_factor,
                        input_ub_tilling_axis, input_ub_tilling_factor, set_reduce_mean_cof_flag, reduce_mean_cof,
                        pattern_str, DEFAULT_TILLING, tilling_key);

  return ret;
}

bool GetAtomicBlockTillingInfo(std::vector<int64_t>& input_shape, std::vector<int32_t>& reduce_axis, int32_t core_num,
                               int32_t& block_tilling_axis, int64_t& block_tilling_factor) {
  bool is_find_block_tiling = false;
  int64_t left_mul = 1;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      is_find_block_tiling = true;
      block_tilling_axis = i;
      block_tilling_factor = 1;

      if (left_mul * input_shape[i] >= core_num) {
        block_tilling_axis = i;
        int64_t block_tilling_factor_outer = core_num / left_mul;
        block_tilling_factor = (input_shape[i] + block_tilling_factor_outer - 1) / block_tilling_factor_outer;
        return is_find_block_tiling;
      }
      left_mul = left_mul * input_shape[i];
    }
  }

  return is_find_block_tiling;
}

bool ProcessAtomicTilling(const std::string& opType, OpRunInfo& run_info, const nlohmann::json& op_info,
                          std::vector<int64_t>& input_shape, std::vector<int64_t>& output_shape,
                          std::vector<int32_t>& reduce_axis, int32_t core_num, int64_t max_ub_count, int32_t block_size,
                          bool is_last_axis_reduce, int32_t set_reduce_mean_cof_flag, float reduce_mean_cof,
                          std::string pattern_str) {
  GELOGI("process atomic tilling.");

  int32_t block_tilling_axis = 0;
  int64_t block_tilling_factor = 0;
  bool is_find_block_tilling =
      GetAtomicBlockTillingInfo(input_shape, reduce_axis, core_num, block_tilling_axis, block_tilling_factor);
  if (!is_find_block_tilling) {
    return false;
  }

  int32_t block_dim = GetAtomicBlockDim(input_shape, reduce_axis, block_tilling_axis, block_tilling_factor);

  // input shape reorder
  std::vector<int64_t> reorder_input_shape;
  std::map<int32_t, int32_t> reorder_pos_ref_input;
  std::vector<int32_t> fused_block_tilling_axis;
  ProcessReorderAxis(is_last_axis_reduce, block_tilling_axis, input_shape, reduce_axis, reorder_input_shape,
                     reorder_pos_ref_input, fused_block_tilling_axis, FUSED_REDUCE_AXIS);

  // align
  if (block_size != 0 && reorder_input_shape.size() > 0 && reorder_input_shape.back() % block_size != 0) {
    reorder_input_shape.back() = (reorder_input_shape.back() + block_size - 1) / block_size * block_size;
  }

  int32_t ub_tilling_axis = 0;
  int64_t ub_tilling_factor = 0;
  bool is_find_ub_tilling = GetUbTillingInfo(opType, input_shape, reorder_input_shape, reorder_pos_ref_input,
                                             fused_block_tilling_axis, max_ub_count, block_size, block_tilling_axis,
                                             block_tilling_factor, ub_tilling_axis, ub_tilling_factor);
  if (!is_find_ub_tilling) {
    return false;
  }

  bool ret = SetRunInfo(opType, run_info, op_info, block_dim, input_shape, block_tilling_axis, block_tilling_factor,
                        ub_tilling_axis, ub_tilling_factor, set_reduce_mean_cof_flag, reduce_mean_cof, pattern_str,
                        ATOMIC_TILLING, -1);

  return ret;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] op_paras: inputs/outputs/atts of the op
 * @param [in] op_compile_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ReduceTiling(const std::string& opType, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info) {
  using namespace nlohmann;
  GELOGI("op [%s]: tiling running", opType.c_str());

  // get input shape
  std::vector<int64_t> input_shape_ori;
  if (op_paras.inputs.size() > 0 && op_paras.inputs[0].tensor.size() > 0) {
    input_shape_ori = op_paras.inputs[0].tensor[0].shape;
  } else {
    GE_LOGE("op [%s] : input shape error.", opType.c_str());
    return false;
  }

  if (input_shape_ori.size() <= 0) {
    GE_LOGE("op [%s] : input shape error.", opType.c_str());
    return false;
  }

  std::vector<int64_t> input_shape;
  std::vector<int32_t> reduce_axis;
  std::string pattern_str;
  if (op_info.count("reduce_axis_unknown") > 0 && op_info["reduce_axis_unknown"].get<std::int32_t>() == 1) {
    GELOGI("op [%s]: reduce axis unknown.", opType.c_str());

    // get original reduce axis
    std::vector<int32_t> reduce_axis_ori;
    if (op_paras.const_inputs.find("axes") != op_paras.const_inputs.end()) {
      TeConstTensorData reduce_axis_info = op_paras.const_inputs.at("axes");
      const ge::Tensor& ge_tensor = std::get<2>(reduce_axis_info);
      ge::DataType const_inputs_dtype = ge_tensor.GetTensorDesc().GetDataType();

      if (const_inputs_dtype == ge::DT_INT32) {
        const int32_t* reduce_axis_info_ptr = reinterpret_cast<const int32_t*>(std::get<0>(reduce_axis_info));
        size_t reduce_axis_info_size = std::get<1>(reduce_axis_info);
        GELOGD("reduce axis info size:%d", reduce_axis_info_size);

        for (size_t i = 0; i < reduce_axis_info_size / 4; i++) {
          int32_t reduce_axis_ori_tmp = *(reduce_axis_info_ptr + i);
          reduce_axis_ori.push_back(reduce_axis_ori_tmp);
          GELOGD("reduce axis ori:%d", reduce_axis_ori_tmp);
        }
      } else if (const_inputs_dtype == ge::DT_INT64) {
        const int64_t* reduce_axis_info_ptr = reinterpret_cast<const int64_t*>(std::get<0>(reduce_axis_info));
        size_t reduce_axis_info_size = std::get<1>(reduce_axis_info);
        GELOGD("reduce axis info size:%d", reduce_axis_info_size);

        for (size_t i = 0; i < reduce_axis_info_size / 8; i++) {
          int64_t reduce_axis_ori_tmp = *(reduce_axis_info_ptr + i);
          reduce_axis_ori.push_back((int32_t)reduce_axis_ori_tmp);
          GELOGD("reduce axis ori:%lld", reduce_axis_ori_tmp);
        }
      } else {
        GE_LOGE("op[%s] ReduceTilingAxisUnknown: wrong reduce axis data type.", opType.c_str());
        return false;
      }
    } else {
      GE_LOGE("op[%s] ReduceTilingAxisUnknown: get reduce axis failed.", opType.c_str());
      return false;
    }

    // convert reduce axis
    for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
      if (reduce_axis_ori[i] < 0) {
        reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
      }
    }

    // const input shape
    if (op_info.count("reduce_shape_known") > 0 && op_info["reduce_shape_known"].get<std::int32_t>() == 1) {
      // pattern key is tiling key
      int32_t tilingKey = CalcPattern(input_shape_ori, reduce_axis_ori);
      pattern_str = std::to_string(tilingKey);
      if (op_info.count("block_dim") == 0 || op_info["block_dim"].count(pattern_str) == 0) {
        GE_LOGE("op [%s] : can't find block dim or pattern str in compile info.", opType.c_str());
        return false;
      }

      int32_t block_dim = op_info["block_dim"][pattern_str].get<std::int32_t>();
      run_info.block_dim = block_dim;
      ByteBufferPut(run_info.tiling_data, tilingKey);
      return true;
    }

    // get process input shape and reduce axis
    bool is_del_axis = false;
    for (size_t i = 0; i < input_shape_ori.size(); i++) {
      if (input_shape_ori[i] != 1 && IsInVector(reduce_axis_ori, i)) {
        is_del_axis = true;
        break;
      }
    }

    uint8_t last_axis_type = LAST_AXIS_INIT;
    for (size_t i = 0; i < input_shape_ori.size(); i++) {
      int64_t value = input_shape_ori[i];
      if ((is_del_axis && value != 1) || !is_del_axis) {
        if (last_axis_type == LAST_AXIS_INIT) {
          if (IsInVector(reduce_axis_ori, i)) {
            input_shape.push_back(1);
            reduce_axis.push_back(input_shape.size());
            input_shape.push_back(value);
            last_axis_type = LAST_AXIS_REDUCE;
          } else {
            input_shape.push_back(value);
            last_axis_type = LAST_AXIS_NONREDUCE;
          }
        } else if (last_axis_type == LAST_AXIS_NONREDUCE) {
          if (IsInVector(reduce_axis_ori, i)) {
            reduce_axis.push_back(input_shape.size());
            input_shape.push_back(value);
            last_axis_type = LAST_AXIS_REDUCE;
          } else {
            size_t end_idx = input_shape.size() - 1;
            input_shape[end_idx] = input_shape[end_idx] * value;
            last_axis_type = LAST_AXIS_NONREDUCE;
          }
        } else {
          if (IsInVector(reduce_axis_ori, i)) {
            size_t end_idx = input_shape.size() - 1;
            input_shape[end_idx] = input_shape[end_idx] * value;
            last_axis_type = LAST_AXIS_REDUCE;
          } else {
            input_shape.push_back(value);
            last_axis_type = LAST_AXIS_NONREDUCE;
          }
        }
      }
    }
    if (input_shape.size() > input_shape_ori.size()) {
      input_shape = input_shape_ori;
      reduce_axis = reduce_axis_ori;
    }

    // calc pattern key
    int32_t pattern = CalcPattern(input_shape, reduce_axis);
    pattern_str = std::to_string(pattern);

  } else {
    GELOGI("op [%s]: reduce axis known.", opType.c_str());

    // get process input shape by fused_rel_dic
    if (op_info.count("fused_rel_dic") > 0) {
      const auto& fused_rel_dic = op_info["fused_rel_dic"];
      for (const auto& fused_rel_dic_item : fused_rel_dic.items()) {
        int64_t input_shape_value = 1;
        std::vector<int32_t> fused_rel_dic_item_value = fused_rel_dic_item.value();
        for (uint32_t i = 0; i < fused_rel_dic_item_value.size(); i++) {
          uint32_t fused_axis = fused_rel_dic_item_value[i];
          input_shape_value *= input_shape_ori[fused_axis];
        }
        input_shape.push_back(input_shape_value);
      }
    } else {
      input_shape = input_shape_ori;
    }

    // get reduce pattern
    std::vector<int32_t> compute_pattern;
    if (op_info.count("compute_pattern") == 0) {
      GE_LOGE("op [%s] : can't find compute pattern.", opType.c_str());
      return false;
    }
    const auto& compute_pattern_tmp = op_info["compute_pattern"];
    for (const auto& pattern : compute_pattern_tmp) {
      compute_pattern.push_back(pattern);
    }
    if (compute_pattern.size() == 0) {
      GE_LOGE("op [%s] : can't find compute pattern info.", opType.c_str());
      return false;
    }
    pattern_str = std::to_string(compute_pattern[0]);

    // get process reduce axis
    const auto& reduce_axis_tmp = op_info[pattern_str]["reduce_info"]["reduce_axis"];
    for (const auto& axis : reduce_axis_tmp) {
      reduce_axis.push_back(axis);
    }

    // convert reduce axis
    for (size_t i = 0; i < reduce_axis.size(); i++) {
      if (reduce_axis[i] < 0) {
        reduce_axis[i] = input_shape.size() + reduce_axis[i];
      }
    }
  }

  GELOGI("get compute pattern: %s", pattern_str.c_str());
  if (op_info.count(pattern_str) == 0) {
    GE_LOGE("op [%s] : can't find pattern str in compile info.", opType.c_str());
    return false;
  }

  int64_t max_ub_count = op_info[pattern_str]["max_ub_count"].get<std::int64_t>();
  int32_t core_num = op_info[pattern_str]["core_num"].get<std::int32_t>();
  const std::string& dtypeUB = op_info[pattern_str]["reduce_info"]["dtype"].get<std::string>();
  const std::string& output_dtypeUB = op_info[pattern_str]["reduce_info"]["out_dtype"].get<std::string>();
  int32_t is_keep_dims = op_info[pattern_str]["reduce_info"]["keep_dims"].get<std::int32_t>();
  GELOGI("get base compile info success.");

  // special opinfo process
  float reduce_mean_cof = 1.0;
  int32_t set_reduce_mean_cof_flag = 0;
  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info["reduce_mean_cof_dtype"].get<std::string>();
    if (reduce_mean_cof_dtype == "float32") {
      set_reduce_mean_cof_flag = REDUCE_MEAN_COF_FP32;
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
    } else if (reduce_mean_cof_dtype == "float16") {
      set_reduce_mean_cof_flag = REDUCE_MEAN_COF_FP16;
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
    }
  }

  bool is_have_atomic_tilling_map = false;
  if (op_info[pattern_str].count("atomic_tiling_key_map") > 0) {
    is_have_atomic_tilling_map = true;
  }

  std::vector<int64_t> output_shape;
  int64_t total_output_count = 1;
  int64_t total_reduce_count = 1;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      if (is_keep_dims == 1) {
        output_shape.push_back(1);
      } else {
        output_shape.push_back(0);
      }
      total_reduce_count *= input_shape[i];
    } else {
      output_shape.push_back(input_shape[i]);
      total_output_count *= input_shape[i];
    }
  }

  int32_t calc_block_size = GetBlockSize(dtypeUB);
  if (calc_block_size == 0) {
    GE_LOGE("op [%s] : block size is zero.", opType.c_str());
    return false;
  }

  int32_t output_block_size = GetBlockSize(output_dtypeUB);
  if (output_block_size == 0) {
    GE_LOGE("op [%s] : block size is zero.", opType.c_str());
    return false;
  }

  int32_t block_size = calc_block_size > output_block_size ? calc_block_size : output_block_size;
  bool is_last_axis_reduce = IsInVector(reduce_axis, input_shape.size() - 1);

  bool is_atomic_tilling = is_have_atomic_tilling_map && total_output_count <= max_ub_count &&
                           total_reduce_count * total_output_count > SMALL_SHAPE_THRESHOLD &&
                           total_output_count < (int64_t)core_num * block_size / 2;

  if (is_atomic_tilling) {
    bool atomic_tilling_ret =
        ProcessAtomicTilling(opType, run_info, op_info, input_shape, output_shape, reduce_axis, core_num, max_ub_count,
                             block_size, is_last_axis_reduce, set_reduce_mean_cof_flag, reduce_mean_cof, pattern_str);
    if (atomic_tilling_ret) {
      return true;
    }
  }

  bool normal_tilling_ret =
      ProcessNormalTilling(opType, run_info, op_info, input_shape, output_shape, reduce_axis, core_num, max_ub_count,
                           block_size, is_last_axis_reduce, total_output_count, is_keep_dims, set_reduce_mean_cof_flag,
                           reduce_mean_cof, pattern_str);

  if (normal_tilling_ret) {
    return true;
  }

  bool default_tilling_ret =
      ProcessDefaultTilling(opType, run_info, op_info, input_shape, output_shape, reduce_axis, is_last_axis_reduce,
                            max_ub_count, block_size, set_reduce_mean_cof_flag, reduce_mean_cof, pattern_str);
  if (default_tilling_ret) {
    return true;
  }

  GE_LOGE("op [%s] : default tilling process error.", opType.c_str());
  return false;
}

REGISTER_OP_TILING_FUNC(ReduceSum, ReduceTiling);
REGISTER_OP_TILING_FUNC(ReduceMaxD, ReduceTiling);
REGISTER_OP_TILING_FUNC(ReduceSumD, ReduceTiling);
REGISTER_OP_TILING_FUNC(ReduceMeanD, ReduceTiling);

}  // namespace optiling
