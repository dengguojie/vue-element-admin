/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file bn_reduce.cpp
 * \brief tiling function of op
 */

#include "bn_reduce.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {
using namespace ge;

bool BNReduce::IsInVector(const std::vector<int32_t>& input, const int32_t value) {
  return std::find(input.begin(), input.end(), value) != input.end();
}

int32_t BNReduce::CalcPattern(const std::vector<int64_t>& input, const std::vector<int32_t>& axis) {
  int32_t pattern = 0;
  for (size_t i = 0; i < input.size(); i++) {
    if (IsInVector(axis, i)) {
      pattern += 2 << (input.size() - i - 1);
    } else {
      pattern += ((int)input.size() - 2 - (int)i) >= 0 ? 2 << (input.size() - 2 - i) : 1;
    }
  }
  return pattern;
}

int32_t BNReduce::CalcConstPattern(const std::vector<int32_t>& reduce_axis) {
  // generate dict key according to reduce_axis
  // Init() make reduce axis sorted
  if (reduce_axis.size() == 0) {
    return -1;
  }

  int32_t dict_key = 0;
  for (auto& i : reduce_axis) {
    // dict_key: 1234 -> reduce [0,1,2,3]
    dict_key = 10 * dict_key + i + 1;
  }

  return dict_key;
}

int64_t BNReduce::GetReorderInputShapeMul(const int32_t axis_index, const int32_t block_tiling_axis_in_reorder) {
  int64_t result = 1;

  for (uint32_t i = axis_index + 1; i < reorderInfo.reorder_input_shape.size(); i++) {
    if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }

    if (i != static_cast<uint32_t>(block_tiling_axis_in_reorder)) {
      result = result * reorderInfo.reorder_input_shape[i];
      continue;
    }

    if (i == reorderInfo.reorder_input_shape.size() - 1) {
      result = result * ((tilingInfo.block_tiling_factor + block_size - 1) / block_size * block_size);
    } else {
      result = result * tilingInfo.block_tiling_factor;
    }
  }
  return result;
}

int32_t BNReduce::CalcTilingKey() {
  using namespace std;
  int db = 0;
  int shape_type = 0;
  pattern = CalcPattern(input_shape, reduce_axis);
  vector<int> pos = {db, shape_type, tilingInfo.block_tiling_axis, tilingInfo.ub_tiling_axis, pattern,
                     (int)is_customised, (int)is_fuse_hn};

  vector<int> coefficient = {1000000000, 10000000, 1000000, 100000, 100, 10, 1};

  int32_t key = 0;
  for (size_t i = 0; i < coefficient.size(); i++) {
    key += pos[i] * coefficient[i];
  }
  if (is_customised) {
    compileInfo.atomic = false;
  }
  key = compileInfo.atomic ? key * 1 : -1 * key;
  return key;
}

void BNReduce::EliminateOne() {
  for (auto item : reduce_axis_ori) {
    reduce_flag[item] = 1;
  }

  size_t skip_count = 0;
  size_t pos_a = 0;
  size_t pos_r = 0;
  for (size_t i = 0; i < input_shape_ori.size(); i++) {
    if (input_shape_ori[i] == 1) {
      skip_count++;
      continue;
    }

    normalize_shape[pos_a] = input_shape_ori[i];
    pos_a++;
    if (reduce_flag[i] == 1) {
      normalize_axis[pos_r] = i - skip_count;
      pos_r++;
    }
  }

  if (pos_a < normalize_shape.size()) {
    normalize_shape.resize(pos_a);
  }
  // reset flag
  for (auto item : reduce_axis_ori) {
    reduce_flag[item] = 0;
  }
  // sort axis
  normalize_axis.resize(pos_r);
  if (normalize_shape.empty()) {
    normalize_shape.emplace_back(1);
  }

  OP_LOGD(op_type, "normalize_shape = %s, normalize_axis=%s",
          (ge::DebugString(normalize_shape)).c_str(), (ge::DebugString(normalize_axis)).c_str());
}

bool BNReduce::ConstInputProcPost() {
  // runtime
  pattern = CalcConstPattern(reduce_axis_ori);
  std::string pattern_str = std::to_string(pattern);

  if (op_info.find("_block_dims") == op_info.end() || op_info.find("_atomic_flags") == op_info.end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "_block_dims or _atomic_flags not exist in compile info.");
    return false;
  }

  if (op_info.at("_block_dims").find(pattern_str) == op_info.at("_block_dims").end() ||
      op_info.at("_atomic_flags").find(pattern_str) == op_info.at("_atomic_flags").end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern_str:%s not exist in compile info.", pattern_str.c_str());
    return false;
  }

  run_info.block_dim = op_info.at("_block_dims").at(pattern_str).get<std::int32_t>();
  run_info.clear_atomic = op_info.at("_atomic_flags").at(pattern_str).get<bool>();

  ByteBufferPut(run_info.tiling_data, pattern);
  run_info.tiling_key = pattern;
  OP_LOGD(op_type, "tiling_key:%d", run_info.tiling_key);
  return true;
}

bool BNReduce::FusedReduceAxis() {
  /* model: reduce_axis_unknown (eg: reduce_sum)
   * if input_shape_ori is [-1], model is R
   * if input_shape_ori is [-1, -1], model is AR, RA
   * if input_shape_ori is [-1, -1, -1], model is AR, ARA, RAR
   * if input_shape_ori is [-1, -1, -1, -1], model is AR, ARA, ARAR, RARA
   * Special Regulation:
   * if after fused, model is RA while len(input_shape_ori) is 3, model will be ARA by padding "1".
   * if after fused, model is R while len(input_shape_ori) isn't 1, model will be AR by padding "1".
   * if after fused, model is A, model will be RA by padding "1".
   * */
  vector<int32_t> pos(normalize_shape.size());
  for (auto item : normalize_axis) {
    pos[item] = 1;
  }

  // Fused serial axises which in same type.
  size_t first = 0;
  size_t second = 0;
  int64_t value_input = 0;
  int32_t value_axis = 0;
  size_t length = normalize_shape.size();
  bool cond_0 = false;
  bool cond_1 = false;

  size_t capacity_shape = 0;
  size_t capacity_axis = 0;

  // Deal Model
  if (normalize_axis.size() == 0) {
    // model is A, A -> RA (only one RA)
    input_shape[0] = 1;
    reduce_axis[0] = 0;
    capacity_shape++;
    capacity_axis++;
  } else if (normalize_axis[0] == 0) {
    // model is Rx, Rx -> ARx
    input_shape[0] = 1;
    capacity_shape++;
  }

  while (second <= length) {
    if (second <= length - 1 and pos[first] == pos[second]) {
      // look for unequal idx
      second += 1;
      continue;
    }

    // fused serial axises
    value_input = std::accumulate(normalize_shape.begin() + first, normalize_shape.begin() + second, 1,
                                  std::multiplies<int64_t>());
    input_shape[capacity_shape] = value_input;
    capacity_shape++;

    // cond_0: [first, second) is serial reduce_axises
    // cond_1: [first: ] is serial reduce_axises.
    cond_0 = second <= length - 1 and pos[second] == 0;
    cond_1 = second == length and pos[second - 1] == 1;
    if (cond_0 or cond_1) {
      value_axis = capacity_shape - 1;
      reduce_axis[capacity_axis] = value_axis;
      capacity_axis++;
    }
    first = second;
    second += 1;
  }

  input_shape.resize(capacity_shape);
  reduce_axis.resize(capacity_axis);

  OP_LOGD(op_type, "input_shape = %s, reduce_axis = %s",
          (ge::DebugString(input_shape)).c_str(), (ge::DebugString(reduce_axis)).c_str());
  return true;
}

bool BNReduce::GetCompileInfo() {
  if (op_info.find("_common_info") == op_info.end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "_common_info not exist in compile info.");
    return false;
  }

  std::vector<int32_t> info = op_info["_common_info"];
  const uint32_t info_item_count = 6;
  if (info.size() < info_item_count) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "common info item count = %lu, is not equal to %d.", info.size(), info_item_count);
    return false;
  }

  compileInfo.max_ub_count = info[0];
  compileInfo.core_num = info[1];
  compileInfo.is_keep_dims = (bool)info[2];
  compileInfo.reduce_block_size = info[3];
  compileInfo.atomic =  (bool)info[4];
  compileInfo.customised =  (bool)info[5];

  block_size = compileInfo.reduce_block_size;

  if (compileInfo.max_ub_count <= 0 || compileInfo.core_num <= 0 || compileInfo.reduce_block_size <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                    "invalid compile info: max_ub_count = %ld, core_num = %d, reduce_block_size = %d ",
                                    compileInfo.max_ub_count, compileInfo.core_num, compileInfo.reduce_block_size);
    return false;
  }

  OP_LOGD(op_type, "max_ub_count = %lld, core_num = %d, atomic = %d, is_customised = %d ",
          compileInfo.max_ub_count, compileInfo.core_num,
          compileInfo.atomic, compileInfo.customised);
  return true;
}

bool BNReduce::ChooseAtomic() {
  int64_t total_output_count = 1;
  int64_t total_reduce_count = 1;

  output_shape.resize(input_shape.size());

  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      output_shape[i] = 1;
      total_reduce_count *= input_shape[i];
    } else {
      output_shape[i] = input_shape[i];
      total_output_count *= input_shape[i];
    }
  }

  // 1: Check if atomic is enabled
  // 2: Check if output is large enough (> SMALL_SHAPE_THRESHOLD)
  // 3: Check normal atomic rules
  // 4: (Priority) Check if it is outermost_reduce and is larger than or equal to core_num
  // Layer 0 (Required)
  bool atomic_available = compileInfo.atomic;
  bool is_outermost_reduce = std::find(reduce_axis.begin(), reduce_axis.end(), 1) != reduce_axis.end() && input_shape[0] == 1;
  // Layer 1
  compileInfo.atomic = total_output_count <= compileInfo.max_ub_count &&
                         total_output_count * total_reduce_count > SMALL_SHAPE_THRESHOLD &&
                         total_reduce_count > static_cast<int64_t>(compileInfo.core_num / 2);
  // Layer 2
  compileInfo.atomic = compileInfo.atomic ||
                           (is_outermost_reduce && input_shape[1] >= compileInfo.core_num &&
                           compileInfo.max_ub_count > SMALL_SHAPE_THRESHOLD * 4);

  bool except_flag = false;
  int32_t n_size = static_cast<int32_t>(input_shape_ori[0]);
  int32_t c1_size = static_cast<int32_t>(input_shape_ori[1]);
  int32_t h_size = static_cast<int32_t>(input_shape_ori[2]);
  int32_t w_size = static_cast<int32_t>(input_shape_ori[3]);
  int32_t c0_size = static_cast<int32_t>(input_shape_ori[4]);

  if ((n_size * c1_size) == 1 && (h_size * w_size * c0_size) <= static_cast<int32_t>(compileInfo.max_ub_count * 4)) {
    except_flag = true;
  } else if (n_size != 1 && c1_size != 1 && (h_size * w_size) <= H_W_THRESHOLD && (h_size * w_size) > 1) {
    except_flag = true;
  } else if (n_size == 1 && c1_size != 1 &&
             (n_size * c1_size * h_size * w_size) <= static_cast<int32_t>(compileInfo.max_ub_count)) {
    except_flag = true;
  } else if (n_size != 1 && c1_size == 1 && (n_size * c1_size * h_size * w_size) <= compileInfo.max_ub_count &&
             h_size % (compileInfo.core_num / 2) != 0 && w_size % (compileInfo.core_num / 2) != 0) {
    except_flag = true;
  }

  // Final
  compileInfo.atomic = atomic_available && compileInfo.atomic && !except_flag;
  OP_LOGD(op_type, "change atomic = %d", compileInfo.atomic);
  return true;
}

bool BNReduce::GetUbTilingInfo() {
  // rewrite ub_tiling_factor, ub_tiling_axis
  int32_t block_tiling_axis_in_reorder = -1;
  for (uint32_t i = 0; i < reorderInfo.reorderPos_oriPos.size(); i++) {
    if (reorderInfo.reorderPos_oriPos[i] == tilingInfo.block_tiling_axis) {
      block_tiling_axis_in_reorder = i;
      break;
    }
  }

  int64_t load_mul = 1;
  for (int32_t i = 0; i < static_cast<int32_t>(reorderInfo.reorder_input_shape.size()); i++) {
    if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }

    load_mul = GetReorderInputShapeMul(i, block_tiling_axis_in_reorder);
    if (load_mul <= compileInfo.max_ub_count) {
      tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[i];
      tilingInfo.ub_tiling_factor = (compileInfo.max_ub_count / load_mul);

      int64_t max_ub_tiling_factor = input_shape[tilingInfo.ub_tiling_axis];
      if (i == block_tiling_axis_in_reorder) {
        max_ub_tiling_factor = tilingInfo.block_tiling_factor;
      }
      if (tilingInfo.ub_tiling_factor > max_ub_tiling_factor) {
        tilingInfo.ub_tiling_factor = max_ub_tiling_factor;
      }
      if (tilingInfo.ub_tiling_axis == tilingInfo.block_tiling_axis &&
            tilingInfo.block_tiling_factor / tilingInfo.ub_tiling_factor == 1) {
        tilingInfo.ub_tiling_factor = tilingInfo.block_tiling_factor % 2 == 0 ?
            tilingInfo.block_tiling_factor / 2 : tilingInfo.ub_tiling_factor;
      }
      OP_LOGD(op_type, "ub axis = %d, factor = %lld",
              tilingInfo.ub_tiling_axis, tilingInfo.ub_tiling_factor);
      return true;
    }
  }

  OP_LOGD(op_type, "try to get ub tiling info failed.");
  return false;
}

void BNReduce::ProcessReorderAxis(const int32_t fused_type) {
  /* InputShape: a0,r0,a1,r1,a2,r2,r3,a3
   * |---> block_tiling_axis(NormalReduce)
   * |---> core = a0*a1
   * |--->fused_block_tiling_axis
   * ReorderShape: |a0,a1,a2|r0,r1,r2,r3|a3
   * |---> last_reduce_axis_idx
   * */
  int32_t block_tiling_axis = tilingInfo.block_tiling_axis;
  int32_t last_reduce_axis_idx = reduce_axis.back();
  reorderInfo.reorder_input_shape.resize(input_shape.size());
  reorderInfo.reorderPos_oriPos.resize(input_shape.size());

  int num_r = reduce_axis.size();
  int num_a = input_shape.size() - num_r;
  for (auto item : reduce_axis) {
    reduce_flag[item] = 1;
  }

  int pos_r = num_a - ((int)input_shape.size() - (last_reduce_axis_idx + 1));
  int pos_a = 0;

  // [0: last_reduce_axis_idx]
  for (int32_t i = 0; i <= last_reduce_axis_idx; i++) {
    if (reduce_flag[i] == 1) {
      if (fused_type == FUSED_REDUCE_AXIS && i < block_tiling_axis) {
        reorderInfo.fused_block_tiling_axis.emplace_back(pos_r);
      }
      reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_r] = i;
      pos_r++;
    } else {
      if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tiling_axis) {
        reorderInfo.fused_block_tiling_axis.emplace_back(pos_a);
      }
      reorderInfo.reorder_input_shape[pos_a] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_a] = i;
      pos_a++;
    }
  }

  // order last non axis, maybe several axis
  for (int32_t i = last_reduce_axis_idx + 1; i < static_cast<int32_t>(input_shape.size()); i++) {
    if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tiling_axis) {
      reorderInfo.fused_block_tiling_axis.emplace_back(pos_r);
    }
    reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
    reorderInfo.reorderPos_oriPos[pos_r] = i;
    pos_r++;
  }

  return;
}

bool BNReduce::GetAtomicBlockDim() {
  // reload block_dim
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tilingInfo.block_tiling_axis; i++) {
    if (IsInVector(reduce_axis, i)) {
      if (i == tilingInfo.block_tiling_axis) {
        block_dim = static_cast<int32_t>(((input_shape[i] + tilingInfo.block_tiling_factor - 1) /
                    tilingInfo.block_tiling_factor) * block_dim);
        break;
      } else {
        block_dim = static_cast<int32_t>(input_shape[i] * block_dim);
      }
    }
  }
  tilingInfo.block_dim = block_dim;
  OP_LOGD(op_type, "get atomic block dim = %d", tilingInfo.block_dim);
  return true;
}

bool BNReduce::GetAtomicBlockTilingInfo() {
  // rewrite block_tiling_axis, block_tiling_factor.
  bool is_find_block_tiling = false;
  int64_t left_mul = 1;
  int32_t core_num = compileInfo.core_num;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      is_find_block_tiling = true;
      tilingInfo.block_tiling_axis = i;
      tilingInfo.block_tiling_factor = 1;

      if (left_mul * input_shape[i] >= core_num) {
        tilingInfo.block_tiling_axis = i;
        int64_t block_tiling_factor_outer = core_num / left_mul;
        tilingInfo.block_tiling_factor =
          (input_shape[i] + block_tiling_factor_outer - 1) / block_tiling_factor_outer;
        OP_LOGD(op_type, "get atomic block info, block axis = %d, block factor = %lld",
                i, tilingInfo.block_tiling_factor);
        return true;
      }
      left_mul = left_mul * input_shape[i];
    }
  }

  return is_find_block_tiling;
}

bool BNReduce::ProcessAtomicTiling() {
  // init
  tilingInfo.block_dim = 0;
  tilingInfo.block_tiling_axis = 0;
  tilingInfo.block_tiling_factor = 0;
  tilingInfo.ub_tiling_axis = 0;
  tilingInfo.ub_tiling_factor = 0;

  // rewrite TilingInfo(block)
  if (!GetAtomicBlockTilingInfo()) {
    OP_LOGD(op_type, "process atomic tiling ,try to get atomic block tiling info failed");
    return false;
  }
  if (!GetAtomicBlockDim()) {
    OP_LOGD(op_type, "process atomic tiling ,try to get atomic block dim failed");
    return false;
  }

  // rewrite ReorderInfo
  ProcessReorderAxis(FUSED_REDUCE_AXIS);

  // align
  if (reorderInfo.reorder_input_shape.size() > 0 && reorderInfo.reorder_input_shape.back() % block_size != 0) {
    reorderInfo.reorder_input_shape.back() =
      (reorderInfo.reorder_input_shape.back() + block_size - 1) / block_size * block_size;
  }

  // rewrite TilingInfo(ub)
  return GetUbTilingInfo();
}

bool BNReduce::Init() {
  if (op_paras.inputs.size() <= 0 || op_paras.inputs[0].tensor.size() <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error.");
    return false;
  }

  input_shape_ori = op_paras.inputs[0].tensor[0].shape;
  OP_LOGD(op_type, "Init input_shape_ori = %s", (ge::DebugString(input_shape_ori)).c_str());

  // Get ori reduce aixs
  reduce_axis_ori = op_info.at("ori_axis").get<std::vector<int32_t>>();
  OP_LOGD(op_type, "Init get ori reduce axis = %s", (ge::DebugString(reduce_axis_ori)).c_str());

  // Convert reduce axis (-1 -> length+1)
  // CHECK AXIS VALUE
  int32_t max_value = static_cast<int32_t>(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is illegal.");
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }

  // Discard "1" and default sorted
  EliminateOne();
  compileInfo.is_const = op_info.count("_reduce_shape_known") > 0 && op_info.at("_reduce_shape_known").get<bool>();
  compileInfo.is_const_post = op_info.count("_const_shape_post") > 0 && op_info.at("_const_shape_post").get<bool>();

  return true;
}

bool BNReduce::WriteTilingData() {
  if (compileInfo.is_const_post) {
    // runtime
    return ConstInputProcPost();
  }

  if (compileInfo.is_const) {
    // compile
    ByteBufferPut(run_info.tiling_data, tilingInfo.block_tiling_axis);
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(tilingInfo.block_tiling_factor));
    ByteBufferPut(run_info.tiling_data, tilingInfo.ub_tiling_axis);
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(tilingInfo.ub_tiling_factor));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(is_customised));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(is_fuse_hn));

    OP_LOGD(op_type, "block/res_ub tilling axis:%d", tilingInfo.block_tiling_axis);
    OP_LOGD(op_type, "block/res_ub tilling factor:%lld", tilingInfo.block_tiling_factor);
    OP_LOGD(op_type, "ub/input_ub tilling axis:%d", tilingInfo.ub_tiling_axis);
    OP_LOGD(op_type, "ub/input_ub tilling factor:%lld", tilingInfo.ub_tiling_factor);
    OP_LOGD(op_type, "run_info.block_dim:%d", tilingInfo.block_dim);

    run_info.block_dim = tilingInfo.block_dim;
    return true;
  }

  // tiling_key
  run_info.block_dim = tilingInfo.block_dim;
  int32_t tiling_key = CalcTilingKey();
  run_info.tiling_key = static_cast<int32_t>(tiling_key);
  OP_LOGD(op_type, "tiling_key:%d, %s", run_info.tiling_key, std::to_string(run_info.tiling_key).c_str());

  // pure dma_copy, must skip "1".
  uint32_t offset = 0;
  if (input_shape[0] == 1 && reduce_axis[0] == 0) {
    offset = 1;
  }

  if (!is_customised) {
    for (uint32_t i = offset; i < input_shape.size(); i++) {
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(input_shape[i]));
      OP_LOGD(op_type, "input shape:%lld", input_shape[i]);
    }
  } else {
    std::string str_key = std::to_string(tiling_key);
    if (op_info.find("_vars") == op_info.end()) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "_vars not exist in compile info");
      return false;
    }

    const auto& all_vars = op_info.at("_vars").at(str_key);
    const string dim_flag = "_dim_";
    const uint32_t dim_index_pos = dim_flag.size();
    for (const auto& var : all_vars) {
      const std::string& var_str = var.get<std::string>();
      if (var_str.compare(0, dim_index_pos, dim_flag) == 0) {
        uint32_t dim_index = std::atoi(var_str.substr(dim_index_pos, 1).c_str());
        if (dim_index >= input_shape_ori.size()) {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, " write tiling data dim index = %d, is out of input shape size[%lu]",
                                          dim_index, input_shape_ori.size());
          return false;
        }
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(input_shape_ori[dim_index]));
        OP_LOGD(op_type, "input shape[%d]: %lld", dim_index, input_shape_ori[dim_index]);
      }
    }
  }

  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(tilingInfo.block_tiling_factor));
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(tilingInfo.ub_tiling_factor));
  OP_LOGD(op_type, "block/res_ub tilling axis:%d", tilingInfo.block_tiling_axis);
  OP_LOGD(op_type, "block/res_ub tilling factor:%lld", tilingInfo.block_tiling_factor);
  OP_LOGD(op_type, "ub/input_ub tilling axis:%d", tilingInfo.ub_tiling_axis);
  OP_LOGD(op_type, "ub/input_ub tilling factor:%lld", tilingInfo.ub_tiling_factor);

  return true;
}

int64_t BNReduce::CustomisedGetNearestFactor(const int64_t dim, const int64_t split_size) {
  int64_t nearest_factor = split_size;
  while(dim % nearest_factor != 0) {
    nearest_factor--;
  }

  int64_t res = split_size;
  if (split_size / nearest_factor < 2) {
      res = nearest_factor;
  }

  return res;
}

bool BNReduce::CustomisedGetUBTiling() {
  int64_t block_tiling_axis = 2;
  int64_t block_tiling_inner_loop = input_shape_ori[2];

  int64_t n_size = input_shape_ori[0];
  int64_t c1_size = input_shape_ori[1];
  int64_t h_size = input_shape_ori[2];
  int64_t w_size = input_shape_ori[3];
  int64_t c0_size = input_shape_ori[4];

  int64_t hwc0_size = input_shape_ori[2] * input_shape_ori[3] * input_shape_ori[4];
  const int32_t core_num = compileInfo.core_num;
  const int32_t max_ub_count = compileInfo.max_ub_count;

  if(((max_ub_count / hwc0_size) >= 1) &&
       ((c1_size >= core_num && c1_size % core_num == 0) || (n_size >= core_num && n_size % core_num == 0))) {
    int64_t ub_split_axis = 0;
    int64_t ub_split_inner = 1;

    int32_t n_inner;

    if (c1_size >= core_num && c1_size % core_num == 0) {
      n_inner = n_size;
    }  else {
      n_inner = n_size / core_num;
    }

    for (int32_t i = n_inner; i > 0; i--) {
      if (n_inner % i != 0) {
        continue;
      }

      if (h_size * w_size * c0_size * i > max_ub_count) {
        continue;
      }

      ub_split_inner = i;
      break;
    }

    tilingInfo.ub_tiling_axis = ub_split_axis;
    tilingInfo.ub_tiling_factor = ub_split_inner;
    OP_LOGI(op_type, "customised tiling ub axis = %d, ub factor = %d",
            tilingInfo.ub_tiling_axis, tilingInfo.ub_tiling_factor);
    return true;
  }

  int64_t bound_size = max_ub_count;
  int64_t split_axis = block_tiling_axis;
  int64_t temp_size = 1;

  bool need_split = false;
  for (int i = 4; i >= 2; i--) {
    temp_size = temp_size * input_shape_ori[i];
    if (temp_size >= bound_size) {
      split_axis = i;
      temp_size = temp_size / input_shape_ori[i];
      need_split = true;
      break;
    }
  }

  int64_t split_size = 1;
  if (need_split) {
    for (int64_t i = 2; i <= input_shape_ori[split_axis]; i++) {
      if ((temp_size * i) == bound_size) {
        split_size = i;
        break;
      }
      if ((temp_size * i) > bound_size) {
        split_size = i - 1;
        split_size = CustomisedGetNearestFactor(input_shape_ori[split_axis], split_size);
        break;
      }
    }
  } else {
    split_size = block_tiling_inner_loop;
  }

  if (split_axis == 2 && split_size > block_tiling_inner_loop)  {
    split_size = block_tiling_inner_loop;
  }

  tilingInfo.ub_tiling_axis = split_axis;
  tilingInfo.ub_tiling_factor = split_size;

  OP_LOGD(op_type, "customised tiling ub axis = %d, ub factor = %lld",
          tilingInfo.ub_tiling_axis, tilingInfo.ub_tiling_factor);
  return true;
}

bool BNReduce::CustomisedGetBlockTiling() {
  OP_LOGD(op_type, "enter CustomisedGetBlockTiling");

  const int64_t ub_split_axis = tilingInfo.ub_tiling_axis;
  const int64_t ub_split_inner = tilingInfo.ub_tiling_factor;
  const int32_t core_num = compileInfo.core_num;

  int64_t c1_size = input_shape_ori[1];
  int64_t threshold = 16;
  int64_t block_split_axis;
  int64_t block_split_factor;
  int64_t outer_loop = input_shape_ori[ub_split_axis] / ub_split_inner;
  int64_t half_core_num = core_num / 2;
  int64_t batch = input_shape_ori[0];
  const int32_t max_ub_count = compileInfo.max_ub_count;

  if (c1_size >= core_num && input_shape_ori[2] * input_shape_ori[3] > threshold) {
    block_split_axis = 1;
    block_split_factor = (c1_size + core_num - 1) / core_num;

    bool is_mte3_opt = ub_split_axis == 0 && block_split_factor * 16 < max_ub_count;
    is_fuse_hn = is_mte3_opt;

    tilingInfo.block_dim = (c1_size + block_split_factor - 1) / block_split_factor;
  } else if (ub_split_axis == 2 &&
            outer_loop >= core_num &&
            input_shape_ori[2] % core_num == 0 &&
            input_shape_ori[0] < core_num) {
    block_split_axis = 2;
    block_split_factor = core_num;

    int32_t inner_loop = input_shape_ori[2] / core_num;
    GetClosedFactor(inner_loop);
    tilingInfo.block_dim = core_num;
  } else if (ub_split_axis == 2 &&
            input_shape_ori[2] >= half_core_num &&
            input_shape_ori[2] % half_core_num == 0 &&
            input_shape_ori[0] < core_num && input_shape_ori[0] == 2) {
    block_split_axis = 2;
    block_split_factor = core_num;

    int32_t inner_loop = input_shape_ori[2] / half_core_num;
    GetClosedFactor(inner_loop);
    is_fuse_hn = true;

    tilingInfo.block_dim = core_num;
  } else if (batch >= core_num && input_shape_ori[2] * input_shape_ori[3] > threshold)  {
    block_split_axis = 0;
    block_split_factor = core_num;

    int64_t h_size = input_shape_ori[2];
    int64_t c0_size = input_shape_ori[3];

    bool is_c1_too_big = (c1_size * c0_size > compileInfo.max_ub_count) && \
                         ((ub_split_axis == 2 and ub_split_inner == h_size) || ub_split_axis == 0);

    is_fuse_hn = is_c1_too_big;
    tilingInfo.block_dim = core_num;
  } else {
    block_split_axis = 4;
    tilingInfo.block_dim = c1_size;

    saved_customised_ub_factor = tilingInfo.ub_tiling_factor;
    saved_customised_ub_axis = tilingInfo.ub_tiling_axis;

    OP_LOGD(op_type, "customised tiling - return without result");
    return false;
  }

  tilingInfo.block_tiling_axis = block_split_axis;
  tilingInfo.block_tiling_factor = block_split_factor;

  OP_LOGD(op_type, "customised tiling - block axis = %d, block factor = %lld",
      tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);
  return true;
}

int64_t BNReduce::GetClosedFactor(const int64_t inner_loop)  {
  std::vector<int32_t> factors;
  int64_t ub_factor = tilingInfo.ub_tiling_factor;

  int64_t sqrt_n = static_cast<int64_t>(sqrt(inner_loop));
  for (int64_t i = 1; i < (sqrt_n + 1); i++) {
    if (inner_loop % i == 0) {
      int64_t tmp = inner_loop / i;
      factors.push_back(i);
      if (tmp != i) {
            factors.push_back(tmp);
      }
    }
  }

  std::sort(factors.begin(), factors.end());

  uint32_t index = 0;
  bool is_find = false;
  for (uint32_t i = 0; i < factors.size(); i++) {
    if (factors[i] > ub_factor) {
      index = i;
      is_find = true;
      break;
    }
  }

  if (is_find) {
    if (index > 0) {
      index = index - 1;
    }
  } else {
    index = factors.size() - 1;
  }

  tilingInfo.ub_tiling_factor = factors[index];
  return tilingInfo.ub_tiling_factor;
}

bool BNReduce::DoDefaultTiling()  {
  OP_LOGI(op_type, "bn training reduce customised default tiling running");
  is_customised = true;
  run_info.clear_atomic = false;

  reduce_axis = reduce_axis_ori;
  input_shape = input_shape_ori;

  tilingInfo.ub_tiling_axis = saved_customised_ub_axis;
  tilingInfo.ub_tiling_factor = saved_customised_ub_factor;

  tilingInfo.block_dim = input_shape_ori[1];
  tilingInfo.block_tiling_axis = 4;
  tilingInfo.block_tiling_factor = input_shape_ori[1];
  OP_LOGI(op_type, "customised tiling default , block axis= %d, block factor = %d",
          tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);

  return true;
}

bool BNReduce::DoCustomisedTiling()  {
  std::vector<int32_t> info = op_info["_common_info"];
  bool can_customised = (bool)info[5];
  if (!can_customised) {
    OP_LOGD(op_type, "tiling running customised tiling disabled");
    return false;
  }

  OP_LOGI(op_type, "bn training reduce customised tiling running");
  is_customised = true;
  run_info.clear_atomic = false;

  reduce_axis = reduce_axis_ori;
  input_shape = input_shape_ori;

  bool ret = true;
  ret = ret && CustomisedGetUBTiling();
  ret = ret && CustomisedGetBlockTiling();
  return ret;

}

bool BNReduce::DoGeneralTiling()  {
  is_customised = false;
  (void)FusedReduceAxis();

  bool ret = true;
  (void)ChooseAtomic();

  if (compileInfo.atomic) {
    run_info.clear_atomic = true;
    ret = ProcessAtomicTiling();
  } else {
    run_info.clear_atomic = false;
    ret = false;
  }

  return ret;
}
bool BNReduce::DoTiling() {
  OP_LOGD(op_type, "tiling running");
  bool ret = true;
  is_customised = false;

  ret = ret && Init();
  if (compileInfo.is_const && compileInfo.is_const_post) {
    return true;
  }

  ret = ret && GetCompileInfo();

  ret = ret && DoCustomisedTiling();        //  try customised tiling first
  if (!ret) {
    ret = DoGeneralTiling();                // if customised tiling failed, than do atomic tiling next
  }

  if (!ret) {
    ret = DoDefaultTiling();                // if atomic tiling failed, than do default tiling
  }

  return ret;
}

bool BNReduceTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info) {
  OP_LOGD(op_type, "tiling running");
  BNReduce reduce(op_type, op_paras, op_info, run_info);
  bool ret = reduce.DoTiling();
  ret = ret && reduce.WriteTilingData();
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(BNTrainingReduce, BNReduceTiling);
}  // namespace optiling
