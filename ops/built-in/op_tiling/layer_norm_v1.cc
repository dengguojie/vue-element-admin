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
#include "layer_norm_v1.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

#include "../fusion_pass/common/fp16_t.hpp"
#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"
#include "vector_tiling_log.h"

namespace optiling {
struct TilingParams {
  /* data */
  int32_t block_dim;
  int32_t block_tiling_axis;
  int32_t block_tiling_axis_1 = 0;
  int32_t block_factor;
  int32_t block_factor_1 = 1;
  int32_t ub_tiling_axis;
  int32_t ub_factor;
  int32_t ub_tiling_axis_reduce;
  int32_t ub_fuse_factor;
};

struct CompileInfo {
  /* data */
  bool is_const = false;
  bool is_const_post = false;
  bool atomic = false;
  bool is_keep_dims = false;
  bool is_normal = true;
  bool is_fuse_axis = false;
  int64_t max_ub_count;
  int32_t core_num;
  int32_t min_block_size;
};

bool IsInVector(std::vector<int32_t> input, const int32_t value) {
  for (int32_t i = 0; i < static_cast<int32_t>(input.size()); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

int32_t CalcPatternKey(std::vector<int64_t> input, const std::vector<int32_t> &reduce_axis,
                       const int32_t block_split_axis, const int32_t ub_split_axis_index_reduce,
                       const int32_t ub_split_axis, bool is_normal, const bool is_fuse_axis) {
  int32_t pattern = 0;

  for (size_t i = 0; i < input.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      pattern += NUM_THR * pow(NUM_TW, (input.size() - i - 1));
    } else {
      pattern += pow(NUM_TW, (input.size() - 1 - i));
    }
  }

  pattern += block_split_axis * NUM_BSA + ub_split_axis * NUM_USA + ub_split_axis_index_reduce;

  if (!is_normal) {
    pattern = pattern * NUM_TW;
  }
  if (!is_fuse_axis) {
    pattern = pattern * NUM_THR;
  }

  return pattern;
}

int32_t CalcTilingKey(const CompileInfo &commoninfo, const std::vector<int64_t> &input_x,
                      const TilingParams &tilingparams, const std::vector<int32_t> &reduce_axis) {
  int32_t key = 0;
  int32_t block_split_axis = tilingparams.block_tiling_axis;
  int32_t block_split_axis_1 = tilingparams.block_tiling_axis_1;
  int32_t ub_split_axis = tilingparams.ub_tiling_axis;
  int32_t ub_split_axis_index_reduce = tilingparams.ub_tiling_axis_reduce;
  bool is_normal = commoninfo.is_normal;
  bool is_fuse_axis = commoninfo.is_fuse_axis;
  int32_t pattern = CalcPatternKey(input_x, reduce_axis, block_split_axis, ub_split_axis_index_reduce, ub_split_axis,
                                   is_normal, is_fuse_axis);
  std::vector<int32_t> val = {1000000000, 10000000, 1000000, 100000, 10000, 1000};
  std::vector<int32_t> pos = {0, 0, block_split_axis, ub_split_axis, pattern, block_split_axis_1};
  for (size_t i = 0; i < pos.size(); i++) {
    key += pos[i] * val[i];
  }
  int32_t range_key;
  if (input_x[input_x.size() - 1] == LAST_DIM_RANGE_ONE) {
    range_key = LAST_DIM_RANGE_ONE_KEY;
  } else if (input_x[input_x.size() - 1] <= LAST_DIM_RANGE_NUM1) {
    range_key = LAST_DIM_RANGE_NUM1_KEY;
  } else if (input_x[input_x.size() - 1] <= LAST_DIM_RANGE_NUM2) {
    range_key = LAST_DIM_RANGE_NUM2_KEY;
  } else {
    range_key = LAST_DIM_RANGE_OTHER_KEY;
  }

  return key + range_key;
}

bool GetCompileInfo(const string &op_type, const layerNormOpInfo &op_info, CompileInfo &compileinfo,
                    const std::vector<int32_t> &reduce_axis, std::vector<int64_t> input_shape,
                    utils::OpRunInfo &run_info) {
  std::vector<int32_t> common_info;

  OP_TILING_CHECK(op_info.common_info.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [common_info]"), return false);
  common_info = op_info.common_info;

  compileinfo.core_num = common_info[CI_CORE_NUM_INDEX];
  compileinfo.is_keep_dims = (bool)common_info[CI_CORE_KENEP_DIM_INDEX];
  compileinfo.min_block_size = common_info[CI_MIN_BLOCK_SIZE];
  compileinfo.atomic = (bool)common_info[CI_ATOMIC];

  V_OP_TILING_CHECK(
      compileinfo.min_block_size > 0,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "min_block_size is %d that is illegal", compileinfo.min_block_size),
      return false);

  V_OP_TILING_CHECK(compileinfo.core_num > 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is %d that is illegal", compileinfo.core_num),
                    return false);

  if (!(op_info.reduce_mean_cof_dtype.empty())) {
    float reduce_mean_cof = 1.0;
    for (uint32_t i = 0; i < input_shape.size(); i++) {
      reduce_mean_cof = IsInVector(reduce_axis, i) ? reduce_mean_cof / input_shape[i] : reduce_mean_cof;
    }
    const string &reduce_mean_cof_dtype = op_info.reduce_mean_cof_dtype;
    if (reduce_mean_cof_dtype == "float32") {
      run_info.AddTilingData((float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "fp32 reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      run_info.AddTilingData((fe::fp16_t)reduce_mean_cof);
      run_info.AddTilingData((uint16_t)0);
      OP_LOGD(op_type.c_str(), "fp16 reduce mean cof:%f", reduce_mean_cof);
    }
  }
  return true;
}

bool CheckWorkspaceCase(std::vector<int64_t> input_x, const std::vector<int32_t> &reduce_axis,
                        const int32_t max_ub_size) {
  int32_t reduce_shape_size = 1;
  // add last dim is not reduce axis judge NCHW->5HD NC1HWC0 reduce_axis[3]
  for (int32_t j = static_cast<int32_t>(input_x.size() - 1); j >= 0; j--) {
    if (IsInVector(reduce_axis, j)) {
      break;
    }
    reduce_shape_size *= input_x[j];
  }
  // judge
  if (reduce_shape_size > max_ub_size) {
    OP_LOGI("CheckWorkspaceCase: normal case.");
    return false;
  }

  for (size_t j = 0; j < reduce_axis.size(); j++) {
    int32_t reduce_index = reduce_axis[j];
    reduce_shape_size *= input_x[reduce_index];
  }
  if (reduce_shape_size > max_ub_size) {
    OP_LOGI("CheckWorkspaceCase: workspace case.");
    return true;
  }
  OP_LOGI("CheckWorkspaceCase: normal case.");
  return false;
}

bool CheckExceedUbSize(const int32_t block_inner, std::vector<int64_t> input_x, size_t i, const int32_t max_ub_size) {
  // judge exceed ub_size  and do workspace
  int32_t shape_size = block_inner;
  for (size_t j = i + 1; j < input_x.size(); j++) {
    shape_size *= input_x[j];
  }
  if (shape_size > max_ub_size) {
    OP_LOGI("CheckExceedUbSize: true");
    return true;
  }
  OP_LOGI("CheckExceedUbSize: false");
  return false;
}

void CalcUbFactor(const int32_t ub_mul_num, const int32_t block, const int32_t split_num, int32_t &ub_factor,
                  bool &is_open_multi_core) {
  int32_t min_uf = (block + ub_mul_num - 1) / ub_mul_num;
  for (int32_t uf = ub_factor; uf >= min_uf; uf--) {
    int32_t tail_region = split_num % uf;

    if (!(split_num % uf) || (tail_region * ub_mul_num >= block)) {
      ub_factor = uf;
      is_open_multi_core = true;
      break;
    }
  }
}

std::vector<int32_t> GetUbTilingData(const int32_t block_inner, size_t i, std::vector<int64_t> input_x,
                                     const int32_t max_ub_size, const std::vector<int32_t> &reduce_axis,
                                     const int32_t block) {
  int32_t x_size = 1;
  int32_t ub_tiling_axis = i;
  int32_t ub_factor = 1;
  int32_t ub_outer = 1;
  int32_t ub_mul_num = 1;
  int32_t axis_num = 1;
  bool is_workspacecase = false;
  bool is_open_multi_core = false;

  for (int32_t j = static_cast<int32_t>(input_x.size() - 1); j > static_cast<int32_t>(i); j--) {
    axis_num = 1;
    x_size *= input_x[j];
    if (!IsInVector(reduce_axis, j)) {
      ub_mul_num *= input_x[j];
      axis_num = input_x[j];
    }
    if (x_size <= max_ub_size) {
      continue;
    }
    ub_tiling_axis = j;
    ub_mul_num = ub_mul_num / axis_num;
    ub_factor = max_ub_size / (x_size / input_x[j]);
    if ((ub_factor * ub_mul_num < block) && !IsInVector(reduce_axis, j)) {
      ub_tiling_axis = j + 1;
      ub_factor = input_x[ub_tiling_axis];
      ub_mul_num = !IsInVector(reduce_axis, ub_tiling_axis) ? ub_mul_num / ub_factor : ub_mul_num;
    }
    if (IsInVector(reduce_axis, ub_tiling_axis)) {
      is_workspacecase = true;
    } else {
      // check input_x[j]%uf memery overflow
      CalcUbFactor(ub_mul_num, block, input_x[ub_tiling_axis], ub_factor, is_open_multi_core);
    }
    break;
  }

  if ((x_size <= max_ub_size) && (x_size * block_inner > max_ub_size)) {
    int32_t split_num = block_inner;
    ub_factor = max_ub_size / x_size;
    if ((ub_factor * ub_mul_num < block) && !IsInVector(reduce_axis, i)) {
      ub_tiling_axis = i + 1;
      ub_factor = input_x[ub_tiling_axis];
      split_num = input_x[ub_tiling_axis];
      ub_mul_num = ub_mul_num / axis_num;
    }
    if (IsInVector(reduce_axis, ub_tiling_axis)) {
      is_workspacecase = true;
    } else {
      // check input_x[j]%uf memery overflow
      CalcUbFactor(ub_mul_num, block, split_num, ub_factor, is_open_multi_core);
    }
  }
  ub_outer = input_x[ub_tiling_axis] % ub_factor == 0 ? input_x[ub_tiling_axis] / ub_factor
                                                      : (input_x[ub_tiling_axis] + ub_factor) / ub_factor;
  std::vector<int32_t> res = {ub_tiling_axis, ub_factor, is_workspacecase, is_open_multi_core, ub_outer};
  return res;
}

void GetUbTilingDataAfterFuse(const int32_t block_axis_1, const int32_t block_factor_1, const int32_t ub_tiling_axis,
                              std::vector<int64_t> input_x, const int32_t ub_factor, TilingParams &tilingparams) {
  if (block_axis_1 == ub_tiling_axis) {
    int32_t new_ub_factor = (input_x[block_axis_1] + block_factor_1 - 1) / block_factor_1;
    if (new_ub_factor < ub_factor) {
      tilingparams.ub_factor = new_ub_factor;
    }
  }
}

int32_t GetUbReduceAxis(std::vector<int64_t> input_x, const int32_t max_ub_size,
                        const std::vector<int32_t> &reduce_axis) {
  int32_t reduce_shape_size = 1;
  // add last dim is not reduce axis judge
  for (int32_t j = static_cast<int32_t>(input_x.size() - 1); j >= 0; j--) {
    if (IsInVector(reduce_axis, j)) {
      break;
    }
    reduce_shape_size *= input_x[j];
  }
  for (int32_t i = static_cast<int32_t>(reduce_axis.size()) - 1; i >= 0; i--) {
    reduce_shape_size *= input_x[reduce_axis[i]];
    if (reduce_shape_size > max_ub_size) {
      OP_LOGI("In workspace case, ub axis must be reduce axis--> true");
      return i;
    }
  }
  OP_LOGI("In workspace case, ub axis must be reduce axis--> false");

  return 0;
}

int32_t GetUnblockAxisOutputMul(const int32_t block_axis, std::vector<int64_t> input_x,
                                const std::vector<int32_t> &reduce_axis) {
  int32_t mul_num = 1;
  for (size_t i = 0; i < input_x.size(); i++) {
    if (!IsInVector(reduce_axis, i) && (static_cast<size_t>(block_axis) != i)) {
      mul_num *= input_x[i];
    }
  }
  return mul_num;
}

std::vector<int32_t> GetBlockTilingData(const int32_t block_axis, int32_t block_factor, const int32_t ub_tiling_axis,
                                        const int32_t ub_outer, std::vector<int64_t> input_x,
                                        const std::vector<int32_t> &reduce_axis, const int32_t core_num) {
  int32_t block_axis_1 = block_axis + 1;
  block_factor = input_x[block_axis];
  int32_t block_factor_1 = ub_outer;
  int32_t block_axis_1_size = 1;
  int32_t fuse_axis = input_x[block_axis];
  std::vector<int32_t> cm_core_num_list;
  cm_core_num_list.push_back(core_num);
  for (int32_t i = core_num - 1; i > 1; i--) {
    if (!(core_num % i)) {
      cm_core_num_list.push_back(i);
    }
  }
  cm_core_num_list.push_back(1);
  // fuse only consecutive axis, so |block_axis - block_axis_1|=1
  if (IsInVector(reduce_axis, block_axis_1) || (block_axis_1 > ub_tiling_axis)) {
    std::vector<int32_t> res = {block_axis, block_factor, block_axis_1, block_factor_1};
    return res;
  }
  if (block_axis_1 == ub_tiling_axis) {
    fuse_axis *= ub_outer;
    block_axis_1_size = ub_outer;
  } else {
    fuse_axis *= input_x[block_axis_1];
    block_axis_1_size = input_x[block_axis_1];
  }

  if (!(fuse_axis % core_num)) {
    for (size_t k = 0; k < cm_core_num_list.size(); k++) {
      if (!(input_x[block_axis] % cm_core_num_list[k])) {
        block_factor = cm_core_num_list[k];
        block_factor_1 = core_num / cm_core_num_list[k];
        break;
      } else if (!(block_axis_1_size % cm_core_num_list[k])) {
        block_factor_1 = cm_core_num_list[k];
        block_factor = core_num / cm_core_num_list[k];
        break;
      }
    }
  }

  std::vector<int32_t> res = {block_axis, block_factor, block_axis_1, block_factor_1};
  return res;
}

void GetTilingDataAllReduceAxis(std::vector<int64_t> input_x, TilingParams &tilingparams,
                                const std::vector<int32_t> &reduce_axis, const int32_t max_ub_size,
                                CompileInfo &compileinfo, const int32_t block) {
  // all_reduce
  tilingparams.block_dim = 1;
  tilingparams.block_tiling_axis = 0;
  tilingparams.block_factor = input_x[0];
  // judge exceed ub_size and do workspace
  bool isworkspace = CheckWorkspaceCase(input_x, reduce_axis, max_ub_size);
  bool isexceedub = CheckExceedUbSize(input_x[0], input_x, 0, max_ub_size);
  if (isworkspace && isexceedub) {
    // workspace
    // not open and open multi-core block
    std::vector<int32_t> ubtilingdata = GetUbTilingData(input_x[0], 0, input_x, max_ub_size, reduce_axis, block);
    tilingparams.ub_tiling_axis = ubtilingdata[0];
    tilingparams.ub_factor = ubtilingdata[1];
    int32_t ub_tiling_axis_reduce = GetUbReduceAxis(input_x, max_ub_size, reduce_axis);
    tilingparams.ub_tiling_axis_reduce = ub_tiling_axis_reduce;
    tilingparams.ub_fuse_factor = input_x[0];
    compileinfo.is_normal = false;
  } else {
    // normal case
    tilingparams.ub_tiling_axis = 0;
    tilingparams.ub_factor = input_x[0];
    tilingparams.ub_tiling_axis_reduce = 0;
    tilingparams.ub_fuse_factor = 0;
    compileinfo.is_normal = true;
  }
}

void CalcUbFuseFactor(const int32_t block_axis, std::vector<int64_t> input_x, TilingParams &tilingparams,
                      const int32_t core_num, const int32_t max_ub_size, const int32_t block,
                      const int32_t block_inner) {
  /***
   * ub_fuse_factor condition:
   * open multi-core block
   * block_inner < input_x[0]                  /
   * ub_fuse_factor > block                    /
   * block_inner > block                       / --> if true:
   * ub_fuse_factor = max([block,、、、， block_inner]) block_inner %
   * ub_fuse_factor > block      /      else:ub_fuse_factor = block_inner
   * ub_fuse_factor < ub_size                  /
   *
   * not open:
   * ub_fuse_factor = 0
   ***/
  tilingparams.ub_fuse_factor = block_inner;
  if ((block_inner < input_x[block_axis]) && (block_inner > block) && (input_x[block_axis] > core_num)) {
    for (int32_t n = block_inner; n > block; n--) {
      if (n < max_ub_size && block_inner % n > block) {
        tilingparams.ub_fuse_factor = n;
        break;
      }
    }
  }
}

void TilingCommonSplitUB(const int32_t ub_block_inner, const int32_t block_axis, std::vector<int64_t> input_x,
                         TilingParams &tilingparams, const std::vector<int32_t> &reduce_axis, const int32_t core_num,
                         const int32_t max_ub_size, CompileInfo &compileinfo, const int32_t block,
                         int32_t &block_inner) {
  // open multi-core block and split ub axis
  std::vector<int32_t> ubtilingdata =
      GetUbTilingData(ub_block_inner, block_axis, input_x, max_ub_size, reduce_axis, block);
  tilingparams.ub_tiling_axis = ubtilingdata[0];
  tilingparams.ub_factor = ubtilingdata[1];
  bool normal2workspace = ubtilingdata[2];
  if (normal2workspace) {
    tilingparams.ub_tiling_axis_reduce = 0;
    // judge ub_fuse_factor*a1*a2 ... < max_ub_size
    int32_t fuse_num = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(input_x.size()); i++) {
      if (i == block_axis) {
        fuse_num *= block_inner;
      } else {
        fuse_num *= IsInVector(reduce_axis, i) ? 1 : input_x[i];
      }
    }
    bool reset_switch = false;
    while (fuse_num > max_ub_size) {
      fuse_num = fuse_num / TILING_DIVIDE_2;
      block_inner = block_inner / TILING_DIVIDE_2;
      reset_switch = true;
    }
    if (reset_switch) {
      tilingparams.block_factor = block_inner;
      tilingparams.block_dim = (input_x[block_axis] % block_inner == 0)
                                   ? input_x[block_axis] / block_inner
                                   : (input_x[block_axis] + block_inner - 1) / block_inner;
    }
    tilingparams.ub_fuse_factor = block_inner;
    compileinfo.is_normal = false;
  } else {
    bool is_open_multi_core = ubtilingdata[3];
    if (!is_open_multi_core) {
      tilingparams.block_dim = 1;
      tilingparams.block_tiling_axis = block_axis;
      tilingparams.block_tiling_axis_1 = block_axis;
      tilingparams.block_factor_1 = 1;
      tilingparams.block_factor = input_x[block_axis];
    } else if (tilingparams.ub_tiling_axis != static_cast<int32_t>(block_axis)) {
      std::vector<int32_t> blocktilingdata = GetBlockTilingData(block_axis, block_inner, ubtilingdata[0],
                                                                ubtilingdata[4U], input_x, reduce_axis, core_num);
      tilingparams.block_dim = blocktilingdata[1] * blocktilingdata[3U];
      tilingparams.block_tiling_axis = block_axis;
      tilingparams.block_tiling_axis_1 = blocktilingdata[2U];
      tilingparams.block_factor = blocktilingdata[1];
      tilingparams.block_factor_1 = blocktilingdata[3U];
      GetUbTilingDataAfterFuse(blocktilingdata[2U], blocktilingdata[3U], ubtilingdata[0], input_x, ubtilingdata[1],
                               tilingparams);
    }
    tilingparams.ub_tiling_axis_reduce = 0;
    tilingparams.ub_fuse_factor = 0;
    compileinfo.is_normal = true;
  }
}

void TilingCommonGenerate(const int32_t block_axis, std::vector<int64_t> input_x, TilingParams &tilingparams,
                          const std::vector<int32_t> &reduce_axis, const int32_t core_num, const int32_t max_ub_size,
                          CompileInfo &compileinfo, const int32_t block, int32_t &block_inner) {
  // judge exceed ub_size and do workspace
  int32_t ub_block_inner = (input_x[block_axis] > core_num) ? block_inner : block;
  bool isworkspace = CheckWorkspaceCase(input_x, reduce_axis, max_ub_size);
  bool isexceedub = CheckExceedUbSize(block_inner, input_x, block_axis, max_ub_size);
  if (isworkspace && isexceedub) {
    CalcUbFuseFactor(block_axis, input_x, tilingparams, core_num, max_ub_size, block, block_inner);
    // judge ub_fuse_factor*a1*a2 ... < max_ub_size
    int32_t fuse_num = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(input_x.size()); i++) {
      if (i == block_axis) {
        fuse_num *= block_inner;
      } else {
        fuse_num *= IsInVector(reduce_axis, i) ? 1 : input_x[i];
      }
    }
    bool reset_switch = false;
    while (fuse_num > max_ub_size) {
      fuse_num = fuse_num / TILING_DIVIDE_2;
      block_inner = block_inner / TILING_DIVIDE_2;
      ub_block_inner = block_inner;
      reset_switch = true;
    }
    if (reset_switch) {
      tilingparams.block_factor = block_inner;
      tilingparams.block_dim = (input_x[block_axis] % block_inner == 0)
                                   ? input_x[block_axis] / block_inner
                                   : (input_x[block_axis] + block_inner - 1) / block_inner;
      CalcUbFuseFactor(block_axis, input_x, tilingparams, core_num, max_ub_size, block, block_inner);
    }
    // workspace
    // not open and open multi-core block
    std::vector<int32_t> ubtilingdata =
        GetUbTilingData(ub_block_inner, block_axis, input_x, max_ub_size, reduce_axis, block);
    tilingparams.ub_tiling_axis = ubtilingdata[0];
    tilingparams.ub_factor = ubtilingdata[1];
    int32_t ub_tiling_axis_reduce = GetUbReduceAxis(input_x, max_ub_size, reduce_axis);
    tilingparams.ub_tiling_axis_reduce = ub_tiling_axis_reduce;
    compileinfo.is_normal = false;
  } else if (isexceedub && !isworkspace) {
    // open multi-core block and split ub axis
    TilingCommonSplitUB(ub_block_inner, block_axis, input_x, tilingparams, reduce_axis, core_num, max_ub_size,
                        compileinfo, block, block_inner);
  } else {
    // normal case
    tilingparams.ub_tiling_axis = block_axis;
    tilingparams.ub_factor = block_inner;
    tilingparams.ub_tiling_axis_reduce = 0;
    tilingparams.ub_fuse_factor = 0;
    compileinfo.is_normal = true;
  }
}

void GetTilingDataNonAllReduceND(std::vector<int64_t> input_x, TilingParams &tilingparams,
                                 const std::vector<int32_t> &reduce_axis, const int32_t core_num,
                                 const int32_t max_ub_size, CompileInfo &compileinfo, const int32_t block) {
  int32_t block_inner_core;
  for (size_t i = 0; i < input_x.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      continue;
    } else if (input_x[i] > core_num) {
      block_inner_core = (input_x[i] % core_num == 0) ? input_x[i] / core_num : (input_x[i] + core_num - 1) / core_num;
    } else {
      block_inner_core = 1;
    }
    int32_t mul_num = GetUnblockAxisOutputMul(i, input_x, reduce_axis);
    int32_t min_block_inner_core = (block % mul_num == 0) ? block / mul_num : (block + mul_num) / mul_num;
    block_inner_core = (block_inner_core < min_block_inner_core) ? min_block_inner_core : block_inner_core;
    int32_t block_inner = block_inner_core;
    tilingparams.block_tiling_axis = i;
    tilingparams.block_factor = block_inner;
    tilingparams.block_dim =
        (input_x[i] % block_inner == 0) ? input_x[i] / block_inner : (input_x[i] + block_inner - 1) / block_inner;
    TilingCommonGenerate(i, input_x, tilingparams, reduce_axis, core_num, max_ub_size, compileinfo, block, block_inner);
    break;
  }
}

void GetTilingDataNonAllReduceNZ(std::vector<int64_t> input_x, TilingParams &tilingparams,
                                 const std::vector<int32_t> &reduce_axis, const int32_t core_num,
                                 const int32_t max_ub_size, CompileInfo &compileinfo, const int32_t block) {
  // reorder input_x : set reduce axis together
  std::vector<int64_t> new_input_x;
  std::vector<int32_t> new_reduce_axis;
  std::vector<int32_t> old_input_x_index;
  int32_t index = 0;
  for (size_t i = 0; i < input_x.size(); i++) {
    if (IsInVector(reduce_axis, i) && (static_cast<int32_t>(i) == reduce_axis[reduce_axis.size() - 1])) {
      for (size_t j = 0; j < reduce_axis.size(); j++) {
        new_input_x.push_back(input_x[reduce_axis[j]]);
        old_input_x_index.push_back(reduce_axis[j]);
        new_reduce_axis.push_back(index);
        index++;
      }
    } else if (IsInVector(reduce_axis, i)) {
      continue;
    } else {
      new_input_x.push_back(input_x[i]);
      old_input_x_index.push_back(i);
      index++;
    }
  }
  // call ND tiling func as common func
  GetTilingDataNonAllReduceND(new_input_x, tilingparams, new_reduce_axis, core_num, max_ub_size, compileinfo, block);

  int32_t nz_block_tiling_axis = old_input_x_index[tilingparams.block_tiling_axis];
  int32_t nz_ub_tiling_axis = old_input_x_index[tilingparams.ub_tiling_axis];
  int32_t nz_block_tiling_axis_1 = old_input_x_index[tilingparams.block_tiling_axis_1];
  int32_t nz_block_dim = tilingparams.block_dim;
  int32_t nz_block_factor = tilingparams.block_factor;
  if (abs(nz_block_tiling_axis - nz_block_tiling_axis_1) > 1) {
    if (tilingparams.block_tiling_axis_1 &&
        (abs(tilingparams.block_tiling_axis - tilingparams.block_tiling_axis_1) == 1)) {
      nz_block_dim = (nz_block_factor < core_num) ? input_x[nz_block_tiling_axis] : nz_block_factor;
      nz_block_factor = input_x[nz_block_tiling_axis] / nz_block_dim;
    }
    nz_block_tiling_axis_1 = nz_block_tiling_axis;
  }
  // update new_index to old_index
  tilingparams.block_tiling_axis = nz_block_tiling_axis;
  tilingparams.ub_tiling_axis = nz_ub_tiling_axis;
  tilingparams.block_tiling_axis_1 = nz_block_tiling_axis_1;
  tilingparams.block_factor = nz_block_factor;
  tilingparams.block_dim = nz_block_dim;
}

void GetTilingData(std::vector<int64_t> input_x, TilingParams &tilingparams, const std::vector<int32_t> &reduce_axis,
                   const int32_t core_num, const int32_t max_ub_size, CompileInfo &compileinfo,
                   const ge::DataType &input_dtype, const string &input_format) {
  // std::vector<int32_t> tiling_params;
  int32_t block = 1;
  if (input_dtype == ge::DT_FLOAT) {
    block = BLOCK_NUM_32;
  } else {
    block = BLOCK_NUM_16;
  }
  if (input_x.size() == reduce_axis.size()) {
    // all_reduce
    GetTilingDataAllReduceAxis(input_x, tilingparams, reduce_axis, max_ub_size, compileinfo, block);
  } else if (input_format == "FRACTAL_NZ") {
    // non all reduce, input_format=="FRACTAL_NZ"
    GetTilingDataNonAllReduceNZ(input_x, tilingparams, reduce_axis, core_num, max_ub_size, compileinfo, block);
  } else {
    // non all reduce, input_format!="FRACTAL_NZ"
    GetTilingDataNonAllReduceND(input_x, tilingparams, reduce_axis, core_num, max_ub_size, compileinfo, block);
  }
}

std::vector<int32_t> split_shape(std::vector<int64_t> shape_split, int32_t dim_split) {
  // split shape
  int32_t batch_num = 1;
  int32_t data_num = 1;
  for (int32_t i = 0; i < dim_split; i++) {
    batch_num *= shape_split[i];
  }
  for (size_t i = dim_split; i < shape_split.size(); i++) {
    data_num *= shape_split[i];
  }
  std::vector<int32_t> res = {batch_num, data_num};
  return res;
}

void LayerNormTikTiling(const std::vector<int64_t> &input_x, int32_t begin_norm_axis, const ge::DataType &gm_type,
                        int32_t ub_max_byte, int32_t core_num, const string &tik_mode, utils::OpRunInfo &run_info,
                        bool atomic_clean_diff_shape) {
  int32_t tiling_mode = 1;
  std::vector<int32_t> res = split_shape(input_x, begin_norm_axis);
  int32_t batch_num = res[0];
  int32_t data_num = res[1];
  int32_t block_size = 32;
  int32_t const_vector_proc_byte = 256;
  ge::DataType ub_type = ge::DT_FLOAT;
  map<ge::DataType, int32_t> const_dtype_byte = {
      {ge::DT_FLOAT16, 2},
      {ge::DT_FLOAT, 4},
  };
  map<ge::DataType, int32_t> const_per_block_num = {
      {ge::DT_FLOAT16, 16},
      {ge::DT_FLOAT, 8},
  };
  int32_t ub_size;

  if (batch_num <= core_num) {
    ub_size = ub_max_byte;
  } else {
    ub_size = ub_max_byte / TILING_DIVIDE_2;
  }

  // expand tensor:batch_mean_ub, batch_mean_square_ub, batch_variance_ub,
  // work_tensro_ub
  int32_t expand_tensor_num = 4;
  int32_t expand_size = expand_tensor_num * block_size;
  int32_t ub_size_remain = ub_size - expand_size;

  int32_t ub_data_size = const_dtype_byte[ub_type];
  int32_t ub_repeat_data_num = const_vector_proc_byte / const_dtype_byte[ub_type];
  int32_t gm_data_size = const_dtype_byte[gm_type];
  // count tensor: input_data_ub, input_data_square_ub
  int32_t count_tensor_num = 2;
  int32_t each_data_size = count_tensor_num * ub_data_size + ub_data_size / ub_repeat_data_num;
  if (ub_type != gm_type) {
    each_data_size += gm_data_size;
  }

  int32_t data_num_max = ub_size_remain / each_data_size;
  int32_t align_num = ub_repeat_data_num;
  int32_t data_num_align = (data_num_max + align_num - 1) / align_num * align_num;
  int32_t const_vector_proc_max_rpt = 255;
  data_num_max = const_vector_proc_max_rpt * align_num;
  int32_t mode_split_n_max_num = min({data_num_align, data_num_max});
  if (data_num <= mode_split_n_max_num) {
    tiling_mode = 0;
  }

  int32_t each_core_batch_num = (batch_num + core_num - 1) / core_num;
  int32_t loop_times = (batch_num + each_core_batch_num - 1) / each_core_batch_num;
  int32_t last_core_batch_num = batch_num - each_core_batch_num * (loop_times - 1);
  if ((!atomic_clean_diff_shape) && (data_num < const_per_block_num[gm_type])) {
    each_core_batch_num = batch_num;
    loop_times = 1;
    last_core_batch_num = batch_num;
  }

  // tiling_data
  int32_t gm_block_num = block_size / ub_data_size;
  int32_t workspace_mean = (batch_num + gm_block_num) * ub_data_size;
  int32_t workspace_sync = core_num * TILING_FACTOR_8;
  int32_t non_y_workspace = workspace_mean > workspace_sync ? workspace_mean : workspace_sync;
  non_y_workspace = (non_y_workspace + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE;
  std::vector<int64_t> workspaces;
  if ((ub_type != gm_type) && atomic_clean_diff_shape) {
    workspaces = {batch_num * data_num * ub_data_size, non_y_workspace, non_y_workspace, non_y_workspace};
  } else if ((ub_type != gm_type) && (!atomic_clean_diff_shape)) {
    workspaces = {non_y_workspace, non_y_workspace, non_y_workspace};
  }
  for (auto ws : workspaces) {
    run_info.AddWorkspace(ws);
  }
  if (tik_mode == TSCONST) {
    run_info.SetBlockDim(loop_times);
  } else {
    run_info.SetBlockDim(core_num);
  }
  run_info.SetClearAtomic(true);

  run_info.AddTilingData((int32_t)tiling_mode);
  run_info.AddTilingData((int32_t)batch_num);
  run_info.AddTilingData((int32_t)data_num);
  run_info.AddTilingData((int32_t)loop_times);
  run_info.AddTilingData((int32_t)each_core_batch_num);
  run_info.AddTilingData((int32_t)last_core_batch_num);
  float reduce_mean_cof = 1.0 / data_num;
  run_info.AddTilingData((float)reduce_mean_cof);
}

void WriteByteBuffer(const string &mode, const int32_t tiling_key, const int32_t workspace_sub1,
                     TilingParams &tilingparams, CompileInfo &compileinfo, utils::OpRunInfo &run_info) {
  std::vector<int64_t> workspaces = {workspace_sub1};
  for (auto ws : workspaces) {
    run_info.AddWorkspace(ws);
  }
  run_info.SetBlockDim(tilingparams.block_dim);
  run_info.SetTilingKey(tiling_key);

  if (mode == TSCONST) {
    run_info.AddTilingData((int32_t)tilingparams.block_tiling_axis);
    run_info.AddTilingData((int32_t)tilingparams.block_tiling_axis_1);
    run_info.AddTilingData((int32_t)tilingparams.ub_tiling_axis);
    run_info.AddTilingData((int32_t)tilingparams.ub_tiling_axis_reduce);
    run_info.AddTilingData((int32_t)tilingparams.block_factor);
    run_info.AddTilingData((int32_t)tilingparams.block_factor_1);
    run_info.AddTilingData((int32_t)tilingparams.ub_factor);
    run_info.AddTilingData((int32_t)tilingparams.ub_fuse_factor);

    if (compileinfo.is_normal) {
      run_info.AddTilingData((int32_t)1);
    } else {
      run_info.AddTilingData((int32_t)0);
    }
  } else {
    run_info.AddTilingData((int32_t)tilingparams.block_factor);
    run_info.AddTilingData((int32_t)tilingparams.block_factor_1);
    run_info.AddTilingData((int32_t)tilingparams.ub_factor);
    run_info.AddTilingData((int32_t)tilingparams.ub_fuse_factor);
  }
}

bool LayerNormTilingV1(const string &op_type, const ge::Operator &op_paras, const layerNormOpInfo &op_info,
                       utils::OpRunInfo &run_info) {
  OP_LOGI(op_type.c_str(), "LayerNormTilingV1 running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  auto input0_desc = operator_info->MutableInputDesc(0);
  std::vector<int64_t> input_x = input0_desc->MutableShape().GetDims();
  ge::DataType input_dtype = op_paras.GetInputDesc(0).GetDataType();
  const int32_t core_num = op_info.core_num;
  const string input_format = op_info.input_format;
  int32_t begin_norm_axis = op_info.begin_norm_axis;
  int32_t begin_params_axis = op_info.begin_params_axis;
  int32_t len_input_x = input_x.size();
  if (begin_norm_axis < 0) {
    begin_norm_axis += len_input_x;
  }
  if (begin_params_axis < 0) {
    begin_params_axis += len_input_x;
  }

  bool if_tik_support = op_info.is_tik_support;
  if (if_tik_support) {
    OP_LOGI(op_type.c_str(), "LayerNormTikTiling running.");
    string tik_mode;
    auto tik_iter_num = op_info.tik_mode;
    if (!(tik_iter_num.empty())) {
      tik_mode = op_info.tik_mode;
    } else {
      tik_mode = TSDYNAMIC;
    }
    bool atomic_clean_diff_shape = op_info.atomic_clean_diff_shape;
    int32_t ub_max_byte = op_info.ub_max_byte;
    LayerNormTikTiling(input_x, begin_norm_axis, input_dtype, ub_max_byte, core_num, tik_mode, run_info,
                       atomic_clean_diff_shape);
  } else {
    OP_LOGI(op_type.c_str(), "LayerNormDslTiling running.");
    string mode;
    auto iter_num = op_info.mode;
    if (!(iter_num.empty())) {
      mode = op_info.mode;
    } else {
      mode = TSDYNAMIC;
    }
    std::vector<int32_t> reduce_axis = op_info.reduce_axis;

    int32_t max_ub_size;
    int32_t block = 1;
    if (input_dtype == ge::DT_FLOAT) {
      max_ub_size = op_info.max_ub_size_normal_fp32;
      block = BLOCK_NUM_32;
    } else {
      max_ub_size = op_info.max_ub_size_normal_fp16;
      block = BLOCK_NUM_16;
    }

    int32_t workspace_sub1 = TYPE_SIZE_16;
    bool is_support_vexp = op_info.is_support_vexp;
    if (is_support_vexp || (input_dtype == ge::DT_FLOAT)) {
      workspace_sub1 = TYPE_SIZE_32;
    }

    for (uint32_t i = 0; i < input_x.size(); i++) {
      workspace_sub1 *= input_x[i];
      run_info.AddTilingData((int32_t)input_x[i]);
      OP_LOGD(op_type.c_str(), "input_x shape:%d.", input_x[i]);
    }

    // update last dim align block
    std::vector<int64_t> old_input_x = input_x;
    if ((input_x[input_x.size() - 1] != LAST_DIM_RANGE_ONE) && (input_x.size() > 1)) {
      input_x[input_x.size() - 1] = (input_x[input_x.size() - 1] + block - 1) / block * block;
    }

    TilingParams tilingparams;
    CompileInfo compileinfo;
    GetTilingData(input_x, tilingparams, reduce_axis, core_num, max_ub_size, compileinfo, input_dtype, input_format);
    // tiling_key
    int32_t tiling_key = CalcTilingKey(compileinfo, old_input_x, tilingparams, reduce_axis);
    // rwrite ByteBuffer
    WriteByteBuffer(mode, tiling_key, workspace_sub1, tilingparams, compileinfo, run_info);
    bool compileflag = GetCompileInfo(op_type, op_info, compileinfo, reduce_axis, old_input_x, run_info);
    if (!compileflag) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo failed.");
    }
  }
  OP_LOGI(op_type.c_str(), "LayerNormTilingV1 end.");
  return true;
}
}  // namespace optiling
