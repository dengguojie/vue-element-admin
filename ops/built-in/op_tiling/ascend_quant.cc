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
 * \file ascend_quant_tiling.cpp
 * \brief tiling function of op
 */
#include <map>
#include <nlohmann/json.hpp>

#include "error_log.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "external/graph/operator.h"
#include "../op_proto/util/error_util.h"

namespace {
  constexpr int32_t ALIGN_BLOCK = 32;
  constexpr int32_t BASE_DIM_NUM = 4;
  constexpr int32_t SUBTRACTOR_TWO = 2;
  constexpr int32_t SUBTRACTOR_THREE = 3;
  constexpr int32_t SUBTRACTOR_FOUR = 4;
  constexpr int64_t BASE_NUMBER = 2;
  constexpr int64_t FACTOR_TWO = 2;
  constexpr int64_t NC1HWC0_INDEX_C1 = 1;
  constexpr int64_t NC1HWC0_INDEX_H = 2;
  constexpr int64_t NC1HWC0_INDEX_W = 3;
  constexpr int64_t NC1HWC0_INDEX_C0 = 4;
}  // namespace

namespace optiling {
struct CompileInfo {
  int64_t max_ub_count{0};
  int32_t core_num{1};
};

struct TilingParams {
  int32_t block_dim{0};
  int32_t block_tiling_axis{-1};
  int64_t block_factor{1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_factor{1};
  int32_t is_fuse_block{1};
};

int32_t CalcPatternKey(std::vector<int64_t>& shape,
                       int32_t block_tiling_axis,
                       int32_t ub_tiling_axis) {
  int32_t pattern = 0;
  for (size_t i = 0; i < shape.size(); i++) {
    pattern += pow(BASE_NUMBER, (shape.size() - 1 - i));
  }
  pattern += block_tiling_axis * 100 + ub_tiling_axis * 10;

  return pattern;
}

int32_t GetBlockDim(std::vector<int64_t>& out_shape,
                    int32_t tiling_axis,
                    int64_t tiling_n_parts) {
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tiling_axis; i++) {
    if (out_shape[i] != 0) {
      if (i == tiling_axis) {
        block_dim = (int32_t)tiling_n_parts * block_dim;
      } else {
        block_dim = (int32_t)out_shape[i] * block_dim;
      }
    }
  }

  return block_dim;
}

bool GetCompileInfo(const std::string& op_type,
                    const ge::Operator& op_paras,
                    const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info,
                    CompileInfo& compile_info) {
  std::vector<int32_t> info = op_info["common_info"];
  compile_info.max_ub_count = info[0];
  compile_info.core_num = info[1];

  return true;
}

int32_t GetBlockSize(const ge::DataType& dtype) {
  int32_t block_size = 0;
  int32_t type_size = GetSizeByDataType(dtype);
  if (type_size > 0) {
    block_size = ALIGN_BLOCK / type_size;
  }

  return block_size;
}

void GetUbTilingData(TilingParams& param,
                     int32_t block_tiling_axis,
                     int64_t block_factor,
                     std::vector<int64_t>& out_shape,
                     int64_t max_ub_size) {
  int64_t ub_size = 1;
  int32_t ub_tiling_axis = 0;
  int64_t ub_outer = 1;
  int64_t ub_factor = 1;
  bool is_split_block_factor = true;

  for (int64_t j = out_shape.size() - 1; j > block_tiling_axis; j--) {
    ub_size *= out_shape[j];
    if (ub_size > max_ub_size) {
      ub_tiling_axis = j;
      ub_outer = ub_size % max_ub_size == 0 ? ub_size / max_ub_size : (ub_size + max_ub_size - 1) / max_ub_size;
      ub_factor = out_shape[j] % ub_outer == 0 ? out_shape[j] / ub_outer : (out_shape[j] + ub_outer - 1) / ub_outer;
      is_split_block_factor = false;
      break;
    }
  }

  if (is_split_block_factor) {
    ub_size *= block_factor;
    ub_tiling_axis = block_tiling_axis;
    if (ub_size > max_ub_size) {
      ub_outer = ub_size % max_ub_size == 0 ? ub_size / max_ub_size : (ub_size + max_ub_size - 1) / max_ub_size;
    }
    ub_factor = block_factor % ub_outer == 0 ? block_factor / ub_outer : (block_factor + ub_outer - 1) / ub_outer;
  }

  param.ub_tiling_axis = ub_tiling_axis;
  param.ub_factor = ub_factor;
}

int32_t CalcTilingKey(std::vector<int64_t>& shape,
                      TilingParams& param) {
  int32_t key = 0;
  int32_t block_tiling_axis = param.block_tiling_axis;
  int32_t ub_tiling_axis = param.ub_tiling_axis;
  int32_t is_fuse_block = param.is_fuse_block;
  int32_t pattern = CalcPatternKey(shape, block_tiling_axis, ub_tiling_axis);
  std::vector<int32_t> val = {1000000000, 10000000, 1000000, 100000, 10000, 1000};
  std::vector<int32_t> pos = {0, is_fuse_block, 0, block_tiling_axis, ub_tiling_axis, pattern};
  for (size_t i = 0; i < pos.size(); i++) {
    key += pos[i] * val[i];
  }

  return key;
}

static void PrintTilingParams(const TilingParams& param) {
  OP_LOGD("ascend_quant_tiling",
          "(block_dim,block_tiling_axis,block_factor,ub_tiling_axis,ub_factor):(%d,%d,%d,%d,%d)",
          param.block_dim, param.block_tiling_axis, param.block_factor, param.ub_tiling_axis, param.ub_factor);
}

void GetTilingData(TilingParams& param,
                   std::vector<int64_t>& out_shape,
                   CompileInfo& compile_info,
                   const ge::DataType& input_dtype) {
  int64_t core_limit = out_shape.size() - 1;
  int32_t block_tiling_axis = core_limit - 1;
  int64_t block_factor = 1;
  int32_t n_parts = out_shape[core_limit - 1];
  int64_t core_size = 1;
  int32_t core_num = compile_info.core_num;
  int32_t is_fuse_block = 1;

  for (int64_t i = 0; i < core_limit; i++) {
    core_size *= out_shape[i];
    if (out_shape[i] >= core_num && i == 1) {
      is_fuse_block = 0;
      block_tiling_axis = i;
      n_parts = core_num;
      block_factor = (out_shape[i] + core_num - 1) / core_num;
      break;
    }
    if (core_size >= core_num) {
      int64_t left_block_dim = core_size / out_shape[i];
      int64_t cur_block_dim = core_num / left_block_dim;
      int64_t cur_block_factor = (out_shape[i] + cur_block_dim - 1) / cur_block_dim;
      n_parts = cur_block_dim;
      block_tiling_axis = i;
      block_factor = cur_block_factor;
      break;
    }
  }

  param.block_tiling_axis = block_tiling_axis;
  param.is_fuse_block = is_fuse_block;
  if (is_fuse_block > 0) {
    param.block_factor = n_parts;
    param.block_dim = GetBlockDim(out_shape, block_tiling_axis, n_parts);
  } else {
    param.block_factor = n_parts;
    param.block_dim = n_parts;
  }

  int32_t dtype_size = GetBlockSize(input_dtype);
  OP_TILING_CHECK(dtype_size == 0,
      VECTOR_INNER_ERR_REPORT_TILIING("AscendQuant", "input_dtype=%s,size be zero.", to_string(input_dtype).c_str()),
      return);
  int64_t max_ub_size = compile_info.max_ub_count / dtype_size;
  GetUbTilingData(param, block_tiling_axis, block_factor, out_shape, max_ub_size);
}

bool AscendQuantTiling(const std::string& op_type,
                       const ge::Operator& op_paras,
                       const nlohmann::json& op_info,
                       utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "AscendQuantTiling running.");

  Format input_format = op_paras.GetInputDesc(0).GetFormat();
  ge::Shape input_x = op_paras.GetInputDesc(0).GetShape();
  const ge::DataType input_dtype = op_paras.GetInputDesc(0).GetDataType();
  OP_TILING_CHECK(input_format != FORMAT_NC1HWC0 && input_format != FORMAT_FRACTAL_NZ,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                          "input format only support NC1HWC0,FRACTAL_NZ, but got %s.", to_string(input_format).c_str()),
                  return false);

  CompileInfo compile_info;
  bool compile_flag = GetCompileInfo(op_type, op_paras, op_info, run_info, compile_info);
  if (!compile_flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo failed.");
  }

  std::vector<int64_t> input_y;
  std::vector<int64_t> input_x_new;
  int64_t c0 = 32;
  if (input_format == FORMAT_NC1HWC0) {
    int64_t c1 = input_x.GetDim(NC1HWC0_INDEX_C1) % FACTOR_TWO == 0
                     ? input_x.GetDim(NC1HWC0_INDEX_C1) / FACTOR_TWO
                     : (input_x.GetDim(NC1HWC0_INDEX_C1) + 1) / FACTOR_TWO;
    int64_t hw = input_x.GetDim(NC1HWC0_INDEX_H) * input_x.GetDim(NC1HWC0_INDEX_W);

    input_y.push_back(input_x.GetDim(0));
    input_y.push_back(c1);
    input_y.push_back(hw);
    input_y.push_back(c0);

    input_x_new.push_back(input_x.GetDim(0));
    input_x_new.push_back(input_x.GetDim(1));
    input_x_new.push_back(hw);
    input_x_new.push_back(input_x.GetDim(NC1HWC0_INDEX_C0));
  } else {
    int64_t batch = 1;
    if (input_x.GetDimNum() > BASE_DIM_NUM) {
      for (size_t i = 0; i < input_x.GetDimNum() - BASE_DIM_NUM; i++) {
        batch *= input_x.GetDim(i);
      }
    }
    int64_t c1_index = input_x.GetDimNum() - SUBTRACTOR_FOUR;
    int64_t h_index = input_x.GetDimNum() - SUBTRACTOR_THREE;
    int64_t w_index = input_x.GetDimNum() - SUBTRACTOR_TWO;
    int64_t c1 = input_x.GetDim(c1_index) % FACTOR_TWO == 0 ? input_x.GetDim(c1_index) / FACTOR_TWO
                                                            : (input_x.GetDim(c1_index) + 1) / FACTOR_TWO;
    int64_t hw = input_x.GetDim(h_index) * input_x.GetDim(w_index);

    input_y.push_back(batch);
    input_y.push_back(c1);
    input_y.push_back(hw);
    input_y.push_back(c0);

    input_x_new.push_back(batch);
    input_x_new.push_back(input_x.GetDim(c1_index));
    input_x_new.push_back(hw);
    input_x_new.push_back(input_x.GetDim(input_x.GetDimNum() - 1));
  }

  TilingParams tiling_params;
  GetTilingData(tiling_params, input_y, compile_info, input_dtype);

  // tiling_key
  int32_t tiling_key = CalcTilingKey(input_y, tiling_params);
  run_info.SetBlockDim(tiling_params.block_dim);
  run_info.SetTilingKey(tiling_key);
  const auto &var_index_list = op_info["var_index_list"];
  for (const auto &index : var_index_list) {
    if (index < input_x_new.size()) {
      run_info.AddTilingData((int32_t)input_x_new[index]);
      OP_LOGD(op_type.c_str(), "input_x_new shape:%d", input_x_new[index]);
    }
  }
  run_info.AddTilingData((int32_t)tiling_params.block_factor);
  run_info.AddTilingData((int32_t)tiling_params.ub_factor);
  OP_LOGD(op_type.c_str(), "block factor=%d", tiling_params.block_factor);
  OP_LOGD(op_type.c_str(), "ub factor=%d", tiling_params.ub_factor);

  PrintTilingParams(tiling_params);

  OP_LOGI(op_type.c_str(), "AscendQuantTiling end.");
  return true;
}

// register tiling interface of AscendQuant op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(AscendQuant, AscendQuantTiling);
}  // namespace optiling
